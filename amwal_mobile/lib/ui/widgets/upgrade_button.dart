import 'package:flutter/material.dart';
import 'package:uicons/uicons.dart';

class UpgradeButton extends StatelessWidget {
  const UpgradeButton({super.key});

  @override
  Widget build(BuildContext context) {
    return ShaderMask(
      shaderCallback: (Rect bounds) {
        return const LinearGradient(
          colors: [Color(0xFFFFED4F), Color(0xFFFFFDEB), Color(0xFFFFED4F)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ).createShader(bounds);
      },
      child: FilledButton.icon(
        style: ButtonStyle(
          foregroundColor: WidgetStatePropertyAll(Colors.black),
          backgroundColor: WidgetStatePropertyAll(Colors.white),
        ),
        onPressed: (){},
        label: Text("رقّي خطتك"),
        icon: Icon(UIcons.regularRounded.bolt),
      ),
    );
  }
}
