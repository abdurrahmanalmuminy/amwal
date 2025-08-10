import 'package:amwal_mobile/ui/screens/navigation/navigation.dart';
import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:pinput/pinput.dart';

class Otp extends StatefulWidget {
  const Otp({super.key});

  @override
  State<Otp> createState() => _OtpState();
}

class _OtpState extends State<Otp> {
  @override
  Widget build(BuildContext context) {
    final defaultPinTheme = PinTheme(
      textStyle: Theme.of(context).textTheme.bodyLarge,
      width: 50,
      height: 50,
      decoration: BoxDecoration(
        color: Theme.of(context).chipTheme.backgroundColor,
        borderRadius: BorderRadius.circular(15),
      ),
    );

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
                    "التحقق",
                    style: Theme.of(context).textTheme.titleLarge,
                    textAlign: TextAlign.center,
                  ),
                  gap(height: 40),
                  Pinput(
                    length: 6,
                    defaultPinTheme: defaultPinTheme,
                    autofocus: true,
                    pinputAutovalidateMode: PinputAutovalidateMode.onSubmit,
                    showCursor: true,
                    onCompleted: (code) {
                      Navigator.of(context).push(
                        CupertinoPageRoute(
                          fullscreenDialog: true,
                          builder: (context) => const Navigation(),
                        ),
                      );
                    },
                  ),
                  gap(height: 40),
                  TextButton(onPressed: (){}, child: Text("ما وصلك الرمز؟")),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
