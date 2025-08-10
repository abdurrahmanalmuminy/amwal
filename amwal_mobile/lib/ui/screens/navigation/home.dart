import 'package:amwal_mobile/models/cashflow.dart';
import 'package:amwal_mobile/models/insight.dart';
import 'package:amwal_mobile/models/mock_data.dart';
import 'package:amwal_mobile/ui/screens/sheets/meet_abdurrahman.dart';
import 'package:amwal_mobile/ui/theme/colors.dart';
import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/cashflow.dart';
import 'package:amwal_mobile/ui/widgets/chat_abdurrahman.dart';
import 'package:amwal_mobile/ui/widgets/goal.dart';
import 'package:amwal_mobile/ui/widgets/insight.dart';
import 'package:amwal_mobile/ui/widgets/section.dart';
import 'package:amwal_mobile/ui/widgets/transaction.dart';
import 'package:amwal_mobile/ui/widgets/upgrade_button.dart';
import 'package:amwal_mobile/ui/widgets/weekly_spending.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:dots_indicator/dots_indicator.dart';
import 'package:flutter/material.dart';
import 'package:uicons/uicons.dart';

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      meetAbdurrahman(context);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBody: true,
      extendBodyBehindAppBar: true,
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
                leadingWidth: 200,
                iconTheme: IconThemeData(color: Colors.white),
                leading: Row(
                  children: [
                    gap(width: 8),
                    IconButton(
                      onPressed: () {},
                      icon: Icon(UIcons.regularRounded.bell),
                    ),
                    IconButton(
                      onPressed: () {},
                      icon: Icon(UIcons.regularRounded.search),
                    ),
                  ],
                ),
                actions: [UpgradeButton(), gap(width: 8)],
              ),
              Stack(
                alignment: AlignmentGeometry.directional(-1, 2),
                children: [
                  Container(
                    height: 150,
                    padding: EdgeInsets.only(bottom: 50),
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: AppColors.primaryColor,
                      borderRadius: BorderRadius.only(
                        bottomLeft: Radius.circular(25),
                        bottomRight: Radius.circular(25),
                      ),
                    ),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          "حياك الله ${mockData.name}! 👋",
                          style: Theme.of(context).textTheme.titleSmall!
                              .copyWith(
                                color: Colors.white.withValues(alpha: 0.5),
                              ),
                        ),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          crossAxisAlignment: CrossAxisAlignment.center,
                          children: [
                            Text(
                              '762.05',
                              style: Theme.of(context).textTheme.titleLarge!
                                  .copyWith(fontSize: 38, color: Colors.white),
                            ),
                            gap(width: 5),
                            Image.asset(
                              'assets/images/riyal_symbol.png', // Change to your image path
                              width: 26,
                              color: Colors.white,
                            ),
                          ],
                        ),
                        Text(
                          "17% مقارنة بالأسبوع السابق",
                          style: Theme.of(
                            context,
                          ).textTheme.bodySmall!.copyWith(color: Colors.white),
                        ),
                      ],
                    ),
                  ),
                  SizedBox(
                    height: 75,
                    child: ListView.separated(
                      itemCount: insights.length,
                      scrollDirection: Axis.horizontal,
                      padding: Dimensions.bodyPadding.copyWith(
                        top: 0,
                        bottom: 15,
                      ),
                      separatorBuilder: (_, __) => gap(width: 10),
                      itemBuilder: (context, index) {
                        final insight = insights[index];
                        return Insight(insight: insight);
                      },
                    ),
                  ),
                ],
              ),
              SafeArea(
                top: false,
                child: Padding(
                  padding: Dimensions.bodyPadding,
                  child: Column(
                    children: [
                      gap(height: 40),
                      ChatAbdurrahman(),
                      gap(height: 20),
                      Section(
                        title: "التدفق النقدي",
                        hideMore: true,
                        child: Row(
                          children: [
                            Expanded(
                              child: Cashflow(
                                cashflow: CashflowClass(
                                  title: "الدخل",
                                  amount: "9.850",
                                ),
                              ),
                            ),
                            gap(width: 10),
                            Expanded(
                              child: Cashflow(
                                cashflow: CashflowClass(
                                  title: "الإنفاق",
                                  amount: "5,212",
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      gap(height: 20),
                      Section(
                        title: "نفقاتك الأسبوعية",
                        hideMore: true,
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          crossAxisAlignment: CrossAxisAlignment.center,
                          children: [
                            Container(
                              padding: EdgeInsets.symmetric(
                                horizontal: 15,
                                vertical: 10,
                              ),
                              decoration: BoxDecoration(
                                color: Theme.of(
                                  context,
                                ).inputDecorationTheme.fillColor,
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Column(
                                children: [
                                  Text("الأحد 20 يوليو - السبت 26 يوليو"),
                                  Row(
                                    mainAxisSize: MainAxisSize.min,
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Text(
                                        "الإنفاق",
                                        style: TextStyle(color: Colors.red),
                                      ),
                                      Icon(
                                        UIcons.regularRounded.arrow_small_down,
                                        size: 18,
                                        color: Colors.red,
                                      ),
                                    ],
                                  ),
                                ],
                              ),
                            ),
                            SizedBox(height: 250, child: WeeklySpending()),
                          ],
                        ),
                      ),
                      gap(height: 20),
                      Section(title: "أهدافي", hideMore: true, child: Goal()),
                      gap(height: 20),
                      Section(
                        title: "عروض التمويل",
                        hideMore: true,
                        child: Column(
                          children: [
                            AspectRatio(
                              aspectRatio: 16 / 9,
                              child: Container(
                                decoration: BoxDecoration(
                                  image: DecorationImage(
                                    image: AssetImage(
                                      "assets/images/offer.jpeg",
                                    ),
                                  ),
                                  border: Border.all(
                                    width: 1,
                                    color: Theme.of(context)
                                        .colorScheme
                                        .onSurface
                                        .withValues(alpha: 0.05),
                                  ),
                                  borderRadius: BorderRadius.circular(20),
                                ),
                              ),
                            ),
                            gap(height: 10),
                            DotsIndicator(
                              dotsCount: 3,
                              position: 0,
                              decorator: DotsDecorator(
                                color: Theme.of(context).dividerTheme.color!,
                                activeColor: AppColors.primaryColor,
                              ),
                            ),
                          ],
                        ),
                      ),
                      gap(height: 20),
                      Section(
                        title: "المعاملات",
                        child: ListView(
                          padding: EdgeInsets.zero,
                          shrinkWrap: true,
                          physics: NeverScrollableScrollPhysics(),
                          children: [Transaction(), Transaction()],
                        ),
                      ),
                      gap(height: 20),
                      Section(
                        title: "الاستثمار",
                        child: Center(
                          child: Image.asset(
                            height: 300,
                            width: double.infinity,
                            Theme.of(context).colorScheme.onSurface ==
                                    Colors.white
                                ? "assets/images/companies_dark.png"
                                : "assets/images/companies.png",
                            fit: BoxFit.fitHeight,
                          ),
                        ),
                      ),
                      gap(height: 20),
                      Section(
                        title: "بودكاست أموال",
                        child: AspectRatio(
                          aspectRatio: 16 / 9,
                          child: Container(
                            decoration: BoxDecoration(
                              image: DecorationImage(
                                image: AssetImage(
                                  "assets/images/podcast_cover.jpg",
                                ),
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
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
