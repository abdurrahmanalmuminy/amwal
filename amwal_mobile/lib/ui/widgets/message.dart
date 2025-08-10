import 'package:amwal_mobile/models/message.dart';
import 'package:amwal_mobile/ui/theme/colors.dart';
import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:uicons/uicons.dart';

class Message extends StatefulWidget {
  final MessageClass messageClass;

  const Message({super.key, required this.messageClass});

  @override
  State<Message> createState() => _MessageState();
}

class _MessageState extends State<Message> with SingleTickerProviderStateMixin {
  final tts = FlutterTts();
  late AnimationController _controller;
  late Animation<double> _animation;

  // Check if the assistant message is still streaming (content is empty)
  bool get _isStreaming =>
      widget.messageClass.role == 'assistant' &&
      widget.messageClass.content.isEmpty;

  @override
  void initState() {
    super.initState();
    // Animation controller for the loading dots when streaming
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    )..repeat(reverse: true);

    _animation = Tween<double>(begin: 0.3, end: 1.0).animate(_controller);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
    tts.stop();
  }

  @override
  Widget build(BuildContext context) {
    final msg = widget.messageClass;

    Widget buildHumanMessage() {
      return Row(
        children: [
          Container(
            constraints: const BoxConstraints(maxWidth: 350),
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
            decoration: BoxDecoration(
              color: AppColors.primaryColor.withValues(alpha: 0.1),
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(20),
                bottomLeft: Radius.circular(20),
                bottomRight: Radius.circular(20),
              ),
            ),
            child: Text(msg.content),
          ),
        ],
      );
    }

    Widget buildAIMessage() {
      // Logic for the loading animation when streaming
      if (_isStreaming) {
        return FadeTransition(
          opacity: _animation,
          child: const ListTile(
            contentPadding: EdgeInsets.zero,
            leading: Icon(Icons.auto_awesome, color: AppColors.primaryColor),
            title: Text(
              "ثواني أفكر شوي...",
            ), // Three dots as a loading indicator
          ),
        );
      }

      // Logic for the final, complete AI message
      return ListTile(
        isThreeLine: true,
        contentPadding: EdgeInsets.zero,
        leading: const Icon(Icons.auto_awesome, color: AppColors.primaryColor),
        title: msg.responseDuration != null
            ? Opacity(
                opacity: 0.5,
                child: Text(
                  "رد عبدالرحمن خلال ${msg.responseDuration!.inSeconds} ثانية",
                  style: Theme.of(
                    context,
                  ).textTheme.bodySmall!.copyWith(height: 1),
                ),
              )
            : null,
        subtitle: Padding(
          padding: const EdgeInsets.only(top: 10),
          child: MarkdownBody(
            data: msg.content.trim(),
            styleSheet: MarkdownStyleSheet(
              p: Theme.of(context).textTheme.bodyMedium,
              h2: Theme.of(
                context,
              ).textTheme.titleMedium!.copyWith(fontWeight: FontWeight.bold),
              strong: const TextStyle(fontWeight: FontWeight.bold),
              listBullet: const TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
        ),
        // The TTS button is only available for a complete message
        trailing: IconButton(
          onPressed: () async {
            await tts.setLanguage("ar-SA");
            // The TTS voice setup should only be called once
            await tts.setVoice({
              "name": "Majed",
              "locale": "ar-001",
              "identifier": "com.apple.voice.compact.ar-001.Maged",
            });
            await tts.setSpeechRate(0.65);
            await tts.speak(msg.content);
          },
          icon: Icon(UIcons.regularRounded.volume),
          iconSize: 18,
        ),
      );
    }

    // Main build method
    return msg.isUser ? buildHumanMessage() : buildAIMessage();
  }
}
