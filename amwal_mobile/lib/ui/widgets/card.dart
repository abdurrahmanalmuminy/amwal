import 'package:flutter/material.dart';

class CustomCard extends StatelessWidget {
  final Widget child;
  final bool? noShadow;
  const CustomCard({super.key, required this.child, this.noShadow});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        border: Border.all(width: 1, color: Theme.of(context).colorScheme.onSurface.withValues(alpha: 0.05)),
        borderRadius: BorderRadius.circular(20),
        boxShadow: noShadow != true ? [
          BoxShadow(
            offset: Offset(0, 1),
            color: Colors.black.withValues(alpha: 0.06),
            blurRadius: 16
          )
        ] : null
      ),
      child: child,
    );
  }
}