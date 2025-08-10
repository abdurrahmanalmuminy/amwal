import 'package:amwal_mobile/models/cashflow.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/material.dart';
import 'package:uicons/uicons.dart';

class Cashflow extends StatelessWidget {
  final CashflowClass cashflow;
  const Cashflow({super.key, required this.cashflow});

  @override
  Widget build(BuildContext context) {
    Color color = cashflow.title == "الدخل" ? Colors.green : Colors.red;
    return Container(
      padding: EdgeInsets.symmetric(vertical: 15),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Text(
                cashflow.amount,
                style: Theme.of(context).textTheme.titleMedium,
              ),
              gap(width: 3),
              Image.asset(
                'assets/images/riyal_symbol.png', // Change to your image path
                width: 15,
                color: Theme.of(context).textTheme.titleMedium!.color,
              ),
            ],
          ),
          gap(height: 5),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(cashflow.title, style: TextStyle(color: color)),
              Icon(
                cashflow.title == "الدخل"
                    ? UIcons.regularRounded.arrow_small_up
                    : UIcons.regularRounded.arrow_small_down,
                size: 18,
                color: color,
              ),
            ],
          ),
        ],
      ),
    );
  }
}
