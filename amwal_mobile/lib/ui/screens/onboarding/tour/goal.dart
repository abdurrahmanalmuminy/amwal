import 'package:amwal_mobile/models/mock_data.dart';
import 'package:amwal_mobile/ui/screens/onboarding/tour/expenses.dart';
import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class Goals extends StatefulWidget {
  const Goals({super.key});

  @override
  State<Goals> createState() => _GoalsState();
}

class _GoalsState extends State<Goals> {
  List<String> selectedGoals = [];
  @override
  Widget build(BuildContext context) {
    List<String> goals = [
      "🚗 شراء سيارة",
      "🏠 سداد ديون",
      "🛫 ادخار لسفر",
      "📈 استثمار طويل المدى",
      "💼 بناء مدخرات للطوارئ",
    ];

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
                    "وش هي أهدافك المالية حالياً؟",
                    style: Theme.of(context).textTheme.titleLarge,
                    textAlign: TextAlign.center,
                  ),
                  gap(height: 40),
                  Expanded(
                    child: ListView.separated(
                      itemCount: goals.length,
                      separatorBuilder: (_, __) => gap(height: 15),
                      itemBuilder: (context, index) {
                        final goal = goals[index];
                        return ChoiceChip(
                          padding: EdgeInsets.symmetric(
                            horizontal: 25,
                            vertical: 15,
                          ),
                          label: Text(
                            goal,
                            style: TextStyle(
                              fontSize: 16,
                              color:
                                  !selectedGoals.contains(goal) &&
                                      Theme.of(context).colorScheme.onSurface ==
                                          Colors.black
                                  ? Colors.black
                                  : null,
                            ),
                          ),
                          selected: selectedGoals.contains(goal),
                          onSelected: (_) {
                            setState(() {
                              selectedGoals.contains(goal)
                                  ? selectedGoals.remove(goal)
                                  : selectedGoals.add(goal);
                            });
                          },
                        );
                      },
                    ),
                  ),
                  SizedBox(
                    width: 220,
                    height: 60,
                    child: ElevatedButton(
                      onPressed: () {
                        mockData.financialGoals = selectedGoals;
                        Navigator.of(context).push(
                          CupertinoPageRoute(
                            builder: (context) => const Expenses(),
                          ),
                        );
                      },
                      child: Text("خطوة أخيرة..."),
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
