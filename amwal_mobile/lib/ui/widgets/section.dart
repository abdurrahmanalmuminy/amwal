import 'package:amwal_mobile/ui/widgets/card.dart';
import 'package:flutter/material.dart';

class Section extends StatelessWidget {
  final String title;
  final Widget child;
  final bool? hideMore;
  const Section({super.key, required this.title, required this.child, this.hideMore});

  @override
  Widget build(BuildContext context) {
    return CustomCard(
      child: Container(
        padding: EdgeInsets.all(15),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Row(
              children: [
                Text(
                  title,
                  style: Theme.of(context).textTheme.titleMedium!.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 10),
              child: child,
            ),
            hideMore != true ? TextButton(onPressed: (){}, child: Text("عرض المزيد")) : SizedBox(),
          ],
        ),
      ),
    );
  }
}
