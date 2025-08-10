import 'package:amwal_mobile/ui/widgets/card.dart';
import 'package:flutter/material.dart';
import 'package:uicons/uicons.dart';

class BlogPost extends StatelessWidget {
  const BlogPost({super.key});

  @override
  Widget build(BuildContext context) {
    return CustomCard(
      noShadow: true,
      child: ListTile(
        title: Text(
          "أول راتب؟ لا تضيّعه مثلنا",
          style: Theme.of(context).textTheme.titleSmall,
        ),
        subtitle: Text(
          "مبروك، وصلك أول راتب؟  أو يمكن ثاني، ثالث… بس دايم يختفي بسرعة؟",
          style: Theme.of(context).textTheme.bodySmall,
        ),
        trailing: Icon(UIcons.regularRounded.angle_small_left, size: 18),
      ),
    );
  }
}
