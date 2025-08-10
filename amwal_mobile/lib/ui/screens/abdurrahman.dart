import 'dart:ui';

import 'package:amwal_mobile/controllers/chat_controller.dart';
import 'package:amwal_mobile/models/message.dart';
import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/message.dart';
import 'package:amwal_mobile/ui/widgets/upgrade_button.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/material.dart';
import 'package:uicons/uicons.dart';

class Abdurrahman extends StatefulWidget {
  const Abdurrahman({super.key});

  @override
  State<Abdurrahman> createState() => _AbdurrahmanState();
}

class _AbdurrahmanState extends State<Abdurrahman> {
  TextEditingController inputController = TextEditingController();
  ChatController controller = ChatController();
  void sendMessage(String message) async {
    final text = message.trim();
    if (text.isNotEmpty) {
      inputController.clear();
      await controller.sendMessage(text);
    }
  }

  @override
  Widget build(BuildContext context) {
    Widget suggestion(value) {
      return GestureDetector(
        onTap: () {
          sendMessage(value);
        },
        child: Container(
          padding: EdgeInsets.symmetric(horizontal: 20, vertical: 15),
          decoration: BoxDecoration(
            color: Theme.of(context).inputDecorationTheme.fillColor,
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(value),
        ),
      );
    }

    Widget buildMessageField() {
      return ClipRect(
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 2, sigmaY: 2),
          child: Container(
            decoration: BoxDecoration(
              color: Theme.of(context).bottomNavigationBarTheme.backgroundColor,
              borderRadius: BorderRadius.only(
                topLeft: Radius.circular(20),
                topRight: Radius.circular(20),
              ),
              border: Border(
                top: BorderSide(
                  width: 1,
                  color: Theme.of(
                    context,
                  ).colorScheme.onSurface.withValues(alpha: 0.05),
                ),
              ),
            ),
            child: SafeArea(
              top: false,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  TextField(
                    controller: inputController,
                    autofocus: true,
                    onSubmitted: (message) => sendMessage(message),
                    decoration: InputDecoration(
                      fillColor: Colors.transparent,
                      hintText: "وش اللي ببالك؟",
                    ),
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      IconButton(
                        onPressed: null,
                        icon: Icon(UIcons.regularRounded.plus, size: 18),
                      ),
                      Row(
                        children: [
                          IconButton(
                            onPressed: null,
                            icon: Icon(
                              UIcons.regularRounded.microphone,
                              size: 18,
                            ),
                          ),
                          IconButton(
                            onPressed: () {
                              sendMessage(inputController.text);
                            },
                            icon: Icon(
                              UIcons.regularRounded.paper_plane,
                              size: 18,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    }

    Widget chatAbdurrahman() {
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          ShaderMask(
            blendMode:
                BlendMode.srcIn, // ✅ ensures gradient replaces foreground color
            shaderCallback: (Rect bounds) {
              bool isDark =
                  Theme.of(context).colorScheme.onSecondary == Colors.black;
              return LinearGradient(
                colors: [
                  isDark ? Color(0xFFEFE4FF) : Color(0xFF7D3AEC),
                  isDark ? Color(0xFF8799FF) : Color(0xFF0026FF),
                ],
                begin: Alignment.topCenter,
                end: Alignment.bottomLeft,
              ).createShader(bounds);
            },
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Icon(
                  Icons.auto_awesome,
                  size: 35,
                  color: Colors.white,
                ), // ✅ white base
                gap(height: 10),
                Text(
                  "هذا عبدالرحمن..\nاسأله أي شي!",
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    color: Colors.white, // ✅ white base for gradient to apply
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                ),
                gap(height: 20),
                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  alignment: WrapAlignment.center,
                  children: [
                    suggestion("كيف كان أدائي هذا الشهر؟"),
                    suggestion("وش يعني سجل ائتماني"),
                    suggestion("كيف أزيد دخلي"),
                    suggestion("كم صرفت الشهر هذا؟"),
                  ],
                ),
              ],
            ),
          ),
        ],
      );
    }

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(actions: [UpgradeButton(), gap(width: 8)]),
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage("assets/images/background.png"),
            fit: BoxFit.cover,
          ),
        ),
        child: Stack(
          alignment: Alignment.bottomCenter,
          children: [
            ValueListenableBuilder<List<MessageClass>>(
              valueListenable: controller.messages,
              builder: (_, chat, __) {
                return chat.isEmpty
                    ? chatAbdurrahman()
                    : ListView.separated(
                        padding: Dimensions.bodyPadding.copyWith(
                          top: 126,
                          bottom: 150,
                        ),
                        itemCount: chat.length,
                        separatorBuilder: (context, index) => gap(height: 5),
                        itemBuilder: (_, i) {
                          final msg = chat[i];
                          return Message(messageClass: msg);
                        },
                      );
              },
            ),
            buildMessageField(),
          ],
        ),
      ),
    );
  }
}
