import 'package:amwal_mobile/ui/screens/onboarding/integration.dart';
import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:uicons/uicons.dart';

class Welcome extends StatelessWidget {
  const Welcome({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        leadingWidth: 100,
        leading: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextButton.icon(
              style: ButtonStyle(
                foregroundColor: WidgetStatePropertyAll(
                  Theme.of(context).textTheme.bodySmall!.color,
                ),
              ),
              onPressed: (){},
              icon: Icon(UIcons.regularRounded.globe),
              label: Text("عربي"),
            ),
          ],
        ),
      ),
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
              children: [
                Expanded(flex: 3, child: SizedBox()),
                Image.asset(
                  height: 450,
                  width: double.infinity,
                  Theme.of(context).colorScheme.onSurface == Colors.white
                      ? "assets/images/welcome_dark.png"
                      : "assets/images/welcome.png",
                  fit: BoxFit.fitHeight,
                ),
                Expanded(child: SizedBox()),
                Padding(
                  padding: Dimensions.bodyPadding,
                  child: Column(
                    children: [
                      Text(
                        "مستقبلك المالي\nيبدأ من هنا!",
                        style: Theme.of(context).textTheme.titleLarge,
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
                                builder: (context) => const Integration(),
                              ),
                            );
                          },
                          child: Text("خلينا نبدأ"),
                        ),
                      ),
                      gap(height: 5),
                      TextButton(onPressed: (){}, child: Text("عندك حساب؟")),
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
