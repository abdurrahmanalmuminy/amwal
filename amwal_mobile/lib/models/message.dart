class MessageClass {
  final String role; // 'user' or 'assistant'
  final String content;
  final Map<String, dynamic>? mockData;
  final Duration? responseDuration;

  MessageClass({
    required this.role,
    required this.content,
    this.mockData,
    this.responseDuration,
  });

  bool get isUser => role == "user";
}
