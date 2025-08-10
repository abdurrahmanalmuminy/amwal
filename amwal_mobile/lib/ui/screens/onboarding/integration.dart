import 'package:amwal_mobile/ui/screens/onboarding/tour/name.dart';
import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:uicons/uicons.dart';

class Integration extends StatefulWidget {
  const Integration({super.key});

  @override
  State<Integration> createState() => _IntegrationState();
}

class _IntegrationState extends State<Integration> {
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
          top: false,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Expanded(flex: 3, child: SizedBox()),
              Image.asset(
                height: 450,
                width: double.infinity,
                Theme.of(context).colorScheme.onSurface == Colors.white
                    ? "assets/images/banks_dark.png"
                    : "assets/images/banks.png",
                fit: BoxFit.fitHeight,
              ),
              Expanded(child: SizedBox()),
              Padding(
                padding: Dimensions.bodyPadding,
                child: Column(
                  children: [
                    Text(
                      "يساعدك تطبيق أموال على ربط حساباتك المصرفية",
                      style: Theme.of(context).textTheme.titleLarge,
                      textAlign: TextAlign.center,
                    ),
                    Wrap(
                      spacing: 10,
                      children: [
                        ChoiceChip(
                          selected: true,
                          avatar: Icon(UIcons.solidRounded.shield_check),
                          label: Text("أمن"),
                          onSelected: (value) {},
                        ),
                        ChoiceChip(
                          selected: true,
                          avatar: Icon(UIcons.solidRounded.bolt),
                          label: Text("مؤتمت بالكامل"),
                          onSelected: (value) {},
                        ),
                      ],
                    ),
                    gap(height: 5),
                    Text(
                      "أموال يصنف معاملاتك، ويساعدك تتبع نفقاتك ويدير أموالك بشكل عام.",
                      style: Theme.of(context).textTheme.bodyMedium,
                      textAlign: TextAlign.center,
                    ),
                    gap(height: 40),
                    SizedBox(
                      width: 220,
                      height: 60,
                      child: ElevatedButton(
                        onPressed: () {
                          Navigator.of(context).push(
                            CupertinoPageRoute(
                              builder: (context) => const Name(),
                            ),
                          );
                        },
                        child: Text("اربط حسابك"),
                      ),
                    ),
                    gap(height: 5),
                    TextButton(
                      onPressed: () {},
                      child: Text("ارفع كشف الحساب"),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
