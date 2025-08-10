import 'package:flutter/material.dart';

class Goal extends StatelessWidget {
  const Goal({super.key});

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: Text("🚗 شراء سيارة"),
      subtitle: LinearProgressIndicator(
        minHeight: 10,
        value: 0.25,
        borderRadius: BorderRadius.circular(100),
        backgroundColor: Theme.of(context).chipTheme.backgroundColor,
      ),
    );
  }
}
