import 'package:amwal_mobile/models/mock_data.dart';
import 'package:amwal_mobile/models/setting.dart';
import 'package:amwal_mobile/ui/theme/colors.dart';
import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/card.dart';
import 'package:amwal_mobile/ui/widgets/upgrade_button.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/material.dart';
import 'package:uicons/uicons.dart';

class Settings extends StatefulWidget {
  const Settings({super.key});

  @override
  State<Settings> createState() => _SettingsState();
}

class _SettingsState extends State<Settings> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage("assets/images/background.png"),
            fit: BoxFit.cover,
          ),
        ),
        child: SingleChildScrollView(
          child: Column(
            children: [
              AppBar(
                backgroundColor: AppColors.primaryColor,
                iconTheme: IconThemeData(color: Colors.white),
                title: Text(
                  "حياك الله! 👋",
                  style: TextStyle(color: Colors.white),
                ),
                actions: [UpgradeButton(), gap(width: 8)],
                automaticallyImplyLeading: false,
              ),
              Container(
                padding: EdgeInsets.only(
                  right: 15,
                  left: 15,
                  bottom: 15,
                  top: 5,
                ),
                width: double.infinity,
                decoration: BoxDecoration(
                  color: AppColors.primaryColor,
                  borderRadius: BorderRadius.only(
                    bottomLeft: Radius.circular(25),
                    bottomRight: Radius.circular(25),
                  ),
                ),
                child: CustomCard(
                  child: ListTile(
                    leading: Icon(
                      UIcons.regularRounded.user,
                      size: 18,
                      color: AppColors.primaryColor,
                    ),
                    title: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Opacity(
                          opacity: 0.5,
                          child: Text(
                            "الملف الشخصي",
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ),
                        Text(mockData.name, style: TextStyle(height: 1)),
                      ],
                    ),
                    trailing: Icon(
                      UIcons.regularRounded.angle_small_left,
                      size: 18,
                    ),
                  ),
                ),
              ),
              gap(height: 20),
              Padding(
                padding: Dimensions.bodyPadding,
                child: ListView.separated(
                  padding: EdgeInsets.zero,
                  physics: NeverScrollableScrollPhysics(),
                  shrinkWrap: true,
                  itemCount: settingsList.length,
                  separatorBuilder: (_, __) => gap(height: 20),
                  itemBuilder: (context, groupIndex) {
                    final settingGroup = settingsList[groupIndex];

                    return CustomCard(
                      child: ListView.separated(
                        padding: EdgeInsets.zero,
                        physics: NeverScrollableScrollPhysics(),
                        shrinkWrap: true,
                        itemCount: settingGroup.length,
                        separatorBuilder: (_, __) => Divider(
                          height: 1,
                          color: Theme.of(
                            context,
                          ).inputDecorationTheme.fillColor,
                        ),
                        itemBuilder: (context, itemIndex) {
                          final setting = settingGroup[itemIndex];
                          bool signOut = setting.label == "تسجيل الخروج";
                          return ListTile(
                            leading: Icon(setting.icon, size: 18),
                            iconColor: signOut
                                ? Colors.red
                                : AppColors.primaryColor,
                            title: Text(
                              setting.label,
                              style: Theme.of(context).textTheme.titleSmall!
                                  .copyWith(color: signOut ? Colors.red : null),
                            ),
                            trailing: signOut
                                ? null
                                : Icon(
                                    UIcons.regularRounded.angle_small_left,
                                    size: 18,
                                  ),
                          );
                        },
                      ),
                    );
                  },
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(top: 10, bottom: 150),
                child: Image.asset(
                  Theme.of(context).colorScheme.onSurface == Colors.white
                      ? "assets/branding/branding_dark.png"
                      : "assets/branding/branding.png",
                  width: 175,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
