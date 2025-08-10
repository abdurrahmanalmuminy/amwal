import 'package:amwal_mobile/ui/theme/colors.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/material.dart';
import 'package:uicons/uicons.dart';

class Transaction extends StatelessWidget {
  const Transaction({super.key});

  @override
  Widget build(BuildContext context) {
    return ListTile(
      contentPadding: EdgeInsets.zero,
      dense: true,
      leading: Container(
        padding: EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: AppColors.primaryColor.withValues(alpha: 0.1),
          border: Border.all(width: 1, color: AppColors.primaryColor),
          borderRadius: BorderRadius.circular(100),
        ),
        child: Icon(
          UIcons.regularRounded.utensils,
          size: 18,
          color: AppColors.primaryColor,
        ),
      ),
      title: Text(
        "مطعم بابا عبده",
        style: Theme.of(context).textTheme.titleSmall,
      ),
      subtitle: Text("24 يوليو في 10:46 مساءً"),
      trailing: SizedBox(
        width: 100,
        child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Text(
                  "-9.850",
                  style: Theme.of(context).textTheme.titleMedium!.copyWith(color: Colors.red),
                ),
                gap(width: 3),
                Image.asset(
                  'assets/images/riyal_symbol.png', // Change to your image path
                  width: 15,
                  color: Colors.red,
                ),
              ],
            ),
      ),
    );
  }
}
