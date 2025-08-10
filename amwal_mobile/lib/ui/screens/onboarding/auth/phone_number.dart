import 'package:amwal_mobile/ui/screens/onboarding/auth/otp.dart';
import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class PhoneNumber extends StatefulWidget {
  const PhoneNumber({super.key});

  @override
  State<PhoneNumber> createState() => _PhoneNumberState();
}

class _PhoneNumberState extends State<PhoneNumber> {
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
                    "كم رقم جوالك",
                    style: Theme.of(context).textTheme.titleLarge,
                    textAlign: TextAlign.center,
                  ),
                  gap(height: 40),
                  SizedBox(
                    width: 225,
                    child: TextField(
                      textAlign: TextAlign.end,
                      textAlignVertical: TextAlignVertical.center,
                      decoration: InputDecoration(
                        suffixIcon: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            SizedBox(height: 20, child: VerticalDivider()),
                            gap(width: 5),
                            Text("🇸🇦 +966"),
                            gap(width: 15),
                          ],
                        ),
                        hintText: "5x xxx xxxx",
                      ),
                    ),
                  ),
                  Expanded(child: SizedBox()),
                  SizedBox(
                    width: 220,
                    height: 60,
                    child: ElevatedButton(
                      onPressed: () {
                        Navigator.of(context).push(
                          CupertinoPageRoute(
                            builder: (context) => const Otp(),
                          ),
                        );
                      },
                      child: Text("التحقق"),
                    ),
                  ),
                  gap(height: 22),
                  Text(
                    "باستخدامك لأموال فإنت توافق على الشروط والأحكام وسياسة الخصوصية",
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
