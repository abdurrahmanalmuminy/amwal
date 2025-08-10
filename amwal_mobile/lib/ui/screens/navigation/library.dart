import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/blog_post.dart';
import 'package:amwal_mobile/ui/widgets/chat_abdurrahman.dart';
import 'package:amwal_mobile/ui/widgets/section.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/material.dart';

class Library extends StatefulWidget {
  const Library({super.key});

  @override
  State<Library> createState() => _LibraryState();
}

class _LibraryState extends State<Library> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(title: Text("المكتبة"), automaticallyImplyLeading: false),
      body: Container(
        padding: Dimensions.bodyPadding,
        width: double.infinity,
        height: double.infinity,
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage("assets/images/background.png"),
            fit: BoxFit.cover,
          ),
        ),
        child: ListView(
          children: [
            Section(
              title: "بودكاست أموال",
              child: AspectRatio(
                aspectRatio: 16 / 9,
                child: Container(
                  decoration: BoxDecoration(
                    image: DecorationImage(
                      image: AssetImage("assets/images/podcast_cover.jpg"),
                    ),
                    border: Border.all(
                      width: 1,
                      color: Theme.of(
                        context,
                      ).colorScheme.onSurface.withValues(alpha: 0.05),
                    ),
                    borderRadius: BorderRadius.circular(20),
                  ),
                ),
              ),
            ),
            gap(height: 20),
            Section(
              title: "اسأل عبدالرحمن",
              hideMore: true,
              child: ChatAbdurrahman(hideAsk: true),
            ),
            gap(height: 20),
            Section(
              title: "المدونة",
              child: ListView(
                shrinkWrap: true,
                physics: NeverScrollableScrollPhysics(),
                children: [BlogPost(), gap(height: 10), BlogPost()],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
