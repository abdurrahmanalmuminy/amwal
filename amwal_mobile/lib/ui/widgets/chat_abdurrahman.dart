import 'package:amwal_mobile/ui/screens/abdurrahman.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class ChatAbdurrahman extends StatelessWidget {
  final bool? hideAsk;
  const ChatAbdurrahman({super.key, this.hideAsk});

  @override
  Widget build(BuildContext context) {
    return ShaderMask(
      shaderCallback: (Rect bounds) {
        return LinearGradient(
          colors: [
            Color.fromARGB(255, 239, 228, 255),
            Color.fromRGBO(135, 153, 255, 1),
          ],
          begin: Alignment.topCenter,
          end: Alignment.bottomLeft,
        ).createShader(bounds);
      },
      child: SizedBox(
        width: double.infinity,
        height: 60,
        child: FilledButton.icon(
          style: ButtonStyle(
            backgroundColor: WidgetStatePropertyAll(Colors.white),
            foregroundColor: WidgetStatePropertyAll(Colors.black),
          ),
          onPressed: () {
            Navigator.of(context).push(
              CupertinoPageRoute(builder: (context) => const Abdurrahman()),
            );
          },
          label: hideAsk == true
              ? Text("كيف كان أدائي هذا الشهر؟")
              : Row(
                  children: [
                    Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "اسأل عبدالرحمن",
                          style: Theme.of(
                            context,
                          ).textTheme.bodySmall!.copyWith(color: Colors.black),
                        ),
                        Text(
                          "كيف كان أدائي هذا الشهر؟",
                          style: TextStyle(height: 1),
                        ),
                      ],
                    ),
                  ],
                ),
          icon: Icon(Icons.auto_awesome),
        ),
      ),
    );
  }
}
