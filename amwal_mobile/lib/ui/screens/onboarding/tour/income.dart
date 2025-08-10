import 'package:amwal_mobile/models/mock_data.dart';
import 'package:amwal_mobile/ui/screens/onboarding/tour/goal.dart';
import 'package:amwal_mobile/ui/theme/colors.dart';
import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class Income extends StatefulWidget {
  const Income({super.key});

  @override
  State<Income> createState() => _IncomeState();
}

class _IncomeState extends State<Income> {
  final TextEditingController income = TextEditingController();
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(),
      body: Container(
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage("assets/images/background.png"),
            fit: BoxFit.cover,
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: Dimensions.bodyPadding,
            child: SizedBox(
              width: double.infinity,
              child: Column(
                children: [
                  Text(
                    "كم دخلك الشهري؟",
                    style: Theme.of(context).textTheme.titleLarge,
                    textAlign: TextAlign.center,
                  ),
                  gap(height: 40),
                  SizedBox(
                    width: 150,
                    child: TextField(
                      controller: income,
                      textAlign: TextAlign.end,
                      keyboardType: TextInputType.number,
                      textAlignVertical: TextAlignVertical.center,
                      decoration: InputDecoration(
                        suffixIcon: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            SizedBox(height: 20, child: VerticalDivider()),
                            gap(width: 5),
                            Image.asset(
                              "assets/images/riyal_symbol.png",
                              width: 15,
                              color: AppColors.primaryColor,
                            ),
                            gap(width: 15),
                          ],
                        ),
                        hintText: "10,000",
                      ),
                    ),
                  ),
                  Expanded(child: SizedBox()),
                  SizedBox(
                    width: 220,
                    height: 60,
                    child: ElevatedButton(
                      onPressed: () {
                        try {
                          mockData.monthlyIncome = int.parse(income.text);
                        } catch (e) {
                          print(e.toString());
                        }
                        Navigator.of(context).push(
                          CupertinoPageRoute(
                            builder: (context) => const Goals(),
                          ),
                        );
                      },
                      child: Text("التالي"),
                    ),
                  ),
                  gap(height: 5),
                  TextButton(onPressed: () {}, child: Text("تخطي")),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
