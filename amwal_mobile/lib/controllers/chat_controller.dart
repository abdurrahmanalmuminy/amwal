import 'dart:convert';
import 'package:amwal_mobile/models/message.dart';
import 'package:amwal_mobile/models/mock_data.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:async';

class ChatController {
  final ValueNotifier<List<MessageClass>> messages = ValueNotifier([]);

  // The streaming endpoint is different
  final String endpoint =
      'https://amwal-agent-1032189486836.me-central1.run.app/chat-stream';

  bool _isStreaming = false;
  http.Client? _client;
  StreamSubscription? _streamSubscription;

  // The sendMessage function now accepts an optional mockData parameter.
  Future<void> sendMessage(String userInput) async {
    if (_isStreaming) {
      // Prevent sending new messages while a stream is active
      return;
    }

    _isStreaming = true;
    final start = DateTime.now();

    // Add user message to the list
    messages.value = [
      ...messages.value,
      MessageClass(
        role: 'user',
        content: userInput,
        mockData: mockData.toJson(),
      ),
    ];

    // Add an empty assistant message to be populated by the stream
    final assistantMessageIndex = messages.value.length;
    messages.value = [
      ...messages.value,
      MessageClass(role: 'assistant', content: ""),
    ];

    _client = http.Client();

    try {
      // Construct the payload with both message and optional mock_data.
      final Map<String, dynamic> payload = {
        'message': userInput,
        'mock_data': mockData,
      };

      final request = http.Request('POST', Uri.parse(endpoint))
        ..headers['Content-Type'] = 'application/json'
        ..body = jsonEncode(payload); // Encode the updated payload

      final streamedResponse = await _client!.send(request);

      // Variable to hold the complete streamed reply
      String fullReply = "";

      // Listen to the byte stream
      _streamSubscription = streamedResponse.stream.listen(
        (data) {
          final chunk = utf8.decode(data);
          // Parse the SSE format
          final lines = chunk.split('\n\n');
          for (final line in lines) {
            if (line.isEmpty) continue;
            final dataString = line.replaceFirst('data: ', '');
            try {
              final jsonChunk = jsonDecode(dataString);
              if (jsonChunk.containsKey('token')) {
                // Append new token to the assistant's message
                fullReply += jsonChunk['token'] as String;
                messages.value = messages.value.map((msg) {
                  if (msg.role == 'assistant' && msg.content.isEmpty) {
                    return MessageClass(role: 'assistant', content: fullReply);
                  } else if (msg.role == 'assistant' &&
                      fullReply.isNotEmpty &&
                      msg == messages.value[assistantMessageIndex]) {
                    // Update the existing assistant message
                    return MessageClass(role: 'assistant', content: fullReply);
                  }
                  return msg;
                }).toList();
              } else if (jsonChunk['done'] == true) {
                // Stream is complete
                _isStreaming = false;
                _streamSubscription?.cancel();
                _streamSubscription = null;
                _client?.close();
                _client = null;

                // Finalize the message with the duration
                messages.value = messages.value.map((msg) {
                  if (msg.role == 'assistant' && msg.content == fullReply) {
                    return MessageClass(
                      role: 'assistant',
                      content: fullReply,
                      responseDuration: DateTime.now().difference(start),
                    );
                  }
                  return msg;
                }).toList();
                break; // Exit the loop
              }
            } catch (e) {
              if (kDebugMode) {
                print("Error parsing stream chunk: $e");
                print("Chunk content: $dataString");
              }
            }
          }
        },
        onError: (error) {
          _isStreaming = false;
          _streamSubscription?.cancel();
          _streamSubscription = null;
          _client?.close();
          _client = null;
          messages.value = [
            ...messages.value,
            MessageClass(role: 'assistant', content: '❌ ما قدرت أوصل للسيرفر.'),
          ];
          if (kDebugMode) {
            print("Stream error: $error");
          }
        },
        onDone: () {
          _isStreaming = false;
          _streamSubscription?.cancel();
          _streamSubscription = null;
          _client?.close();
          _client = null;
          if (fullReply.isEmpty) {
            messages.value = messages.value.map((msg) {
              if (msg.role == 'assistant' && msg.content.isEmpty) {
                return MessageClass(
                  role: 'assistant',
                  content: '⚠️ صار خطأ! حاول مرة ثانية.',
                );
              }
              return msg;
            }).toList();
          }
        },
      );
    } catch (e) {
      _isStreaming = false;
      _streamSubscription?.cancel();
      _streamSubscription = null;
      _client?.close();
      _client = null;
      messages.value = [
        ...messages.value..removeLast(),
        MessageClass(role: 'assistant', content: '❌ ما قدرت أوصل للسيرفر.'),
      ];
      if (kDebugMode) {
        print("HTTP request error: $e");
      }
    }
  }

  void clearChat() {
    messages.value = [];
  }
}
