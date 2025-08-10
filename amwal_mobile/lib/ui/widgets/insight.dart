import 'package:amwal_mobile/models/insight.dart';
import 'package:amwal_mobile/ui/widgets/card.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/material.dart';

class Insight extends StatelessWidget {
  final InsightModel insight;
  const Insight({super.key, required this.insight});

  @override
  Widget build(BuildContext context) {
    return CustomCard(
      child: Container(
        width: 150,
        height: 60,
        padding: EdgeInsets.symmetric(horizontal: 12),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Icon(insight.icon, size: 18, color: insight.color),
            gap(width: 8),
            Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  insight.title,
                  style: Theme.of(
                    context,
                  ).textTheme.bodySmall,
                ),
                Text(insight.value, style: Theme.of(context).textTheme.titleLarge!.copyWith(height: 1)),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
