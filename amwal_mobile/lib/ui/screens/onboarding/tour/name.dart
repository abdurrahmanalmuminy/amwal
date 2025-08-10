import 'package:amwal_mobile/models/mock_data.dart';
import 'package:amwal_mobile/ui/screens/onboarding/tour/income.dart';
import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:uicons/uicons.dart';

class Name extends StatefulWidget {
  const Name({super.key});

  @override
  State<Name> createState() => _NameState();
}

class _NameState extends State<Name> {
  TextEditingController name = TextEditingController();
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
                    "وش اسمك؟",
                    style: Theme.of(context).textTheme.titleLarge,
                    textAlign: TextAlign.center,
                  ),
                  gap(height: 40),
                  SizedBox(
                    width: 225,
                    child: TextField(
                      controller: name,
                      textAlign: TextAlign.end,
                      textAlignVertical: TextAlignVertical.center,
                      decoration: InputDecoration(
                        suffixIcon: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            SizedBox(height: 20, child: VerticalDivider()),
                            gap(width: 5),
                            Icon(UIcons.solidRounded.user, size: 15),
                            gap(width: 15),
                          ],
                        ),
                        hintText: "عبدالرحمن",
                      ),
                    ),
                  ),
                  Expanded(child: SizedBox()),
                  SizedBox(
                    width: 220,
                    height: 60,
                    child: ElevatedButton(
                      onPressed: () {
                        mockData.name = name.text;
                        Navigator.of(context).push(
                          CupertinoPageRoute(
                            builder: (context) => const Income(),
                          ),
                        );
                      },
                      child: Text("اللي بعده"),
                    ),
                  ),
                  TextButton(
                    child: Text("لا تخاف، معلوماتك بأمان والامور طيبة"),
                    onPressed: () {},
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
