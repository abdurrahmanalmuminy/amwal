import 'package:amwal_mobile/models/cashflow.dart';
import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/cashflow.dart';
import 'package:amwal_mobile/ui/widgets/section.dart';
import 'package:amwal_mobile/ui/widgets/transaction.dart';
import 'package:amwal_mobile/ui/widgets/weekly_spending.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/material.dart';
import 'package:uicons/uicons.dart';

class Expenses extends StatefulWidget {
  const Expenses({super.key});

  @override
  State<Expenses> createState() => _ExpensesState();
}

class _ExpensesState extends State<Expenses> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(title: Text("النفقات"), automaticallyImplyLeading: false),
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage("assets/images/background.png"),
            fit: BoxFit.cover,
          ),
        ),
        child: ListView(
          children: [
            SafeArea(
              top: false,
              child: Padding(
                padding: Dimensions.bodyPadding,
                child: Column(
                  children: [
                    Section(
                      title: "التدفق النقدي",
                      hideMore: true,
                      child: Row(
                        children: [
                          Expanded(
                            child: Cashflow(
                              cashflow: CashflowClass(
                                title: "الدخل",
                                amount: "9.850",
                              ),
                            ),
                          ),
                          gap(width: 10),
                          Expanded(
                            child: Cashflow(
                              cashflow: CashflowClass(
                                title: "الإنفاق",
                                amount: "5,212",
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    gap(height: 20),
                    Section(
                      title: "نفقاتك الأسبوعية",
                      hideMore: true,
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        crossAxisAlignment: CrossAxisAlignment.center,
                        children: [
                          Container(
                            padding: EdgeInsets.symmetric(
                              horizontal: 15,
                              vertical: 10,
                            ),
                            decoration: BoxDecoration(
                              color: Theme.of(
                                context,
                              ).inputDecorationTheme.fillColor,
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Column(
                              children: [
                                Text("الأحد 20 يوليو - السبت 26 يوليو"),
                                Row(
                                  mainAxisSize: MainAxisSize.min,
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Text(
                                      "الإنفاق",
                                      style: TextStyle(color: Colors.red),
                                    ),
                                    Icon(
                                      UIcons.regularRounded.arrow_small_down,
                                      size: 18,
                                      color: Colors.red,
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                          SizedBox(height: 250, child: WeeklySpending()),
                        ],
                      ),
                    ),
                    gap(height: 20),
                    Section(
                      title: "المعاملات",
                      child: ListView(
                        padding: EdgeInsets.zero,
                        shrinkWrap: true,
                        physics: NeverScrollableScrollPhysics(),
                        children: [Transaction(), Transaction()],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
