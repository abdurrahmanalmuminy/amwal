import 'package:amwal_mobile/models/mock_data.dart';
import 'package:amwal_mobile/ui/screens/onboarding/auth/phone_number.dart';
import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class Expenses extends StatefulWidget {
  const Expenses({super.key});

  @override
  State<Expenses> createState() => _ExpensesState();
}

class _ExpensesState extends State<Expenses> {
  List<String> selectedExpenses = [];
  @override
  Widget build(BuildContext context) {
    List<String> expenses = [
      "🏠 إيجار",
      "🚗 أقساط سيارة",
      "👨‍👩👧 مصروف عائلة",
      "📱 اشتراكات وخدمات",
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
                    "عندك التزامات شهرية؟",
                    style: Theme.of(context).textTheme.titleLarge,
                    textAlign: TextAlign.center,
                  ),
                  gap(height: 40),
                  Expanded(
                    child: ListView.separated(
                      itemCount: expenses.length,
                      separatorBuilder: (_, __) => gap(height: 15),
                      itemBuilder: (context, index) {
                        final expens = expenses[index];
                        return ChoiceChip(
                          padding: EdgeInsets.symmetric(
                            horizontal: 25,
                            vertical: 15,
                          ),
                          label: Text(
                            expens,
                            style: TextStyle(
                              fontSize: 16,
                              color:
                                  !selectedExpenses.contains(expens) &&
                                      Theme.of(context).colorScheme.onSurface ==
                                          Colors.black
                                  ? Colors.black
                                  : null,
                            ),
                          ),
                          selected: selectedExpenses.contains(expens),
                          onSelected: (_) {
                            setState(() {
                              selectedExpenses.contains(expens)
                                  ? selectedExpenses.remove(expens)
                                  : selectedExpenses.add(expens);
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
                        mockData.monthlyCommitments = selectedExpenses;
                        Navigator.of(context).push(
                          CupertinoPageRoute(
                            builder: (context) => const PhoneNumber(),
                          ),
                        );
                      },
                      child: Text("أنشئ حسابي"),
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
