import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/material.dart';

class Investment extends StatefulWidget {
  const Investment({super.key});

  @override
  State<Investment> createState() => _InvestmentState();
}

class _InvestmentState extends State<Investment> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(toolbarHeight: 0, automaticallyImplyLeading: false),
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
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Expanded(flex: 3, child: SizedBox()),
                Image.asset(
                  height: 450,
                  width: double.infinity,
                  Theme.of(context).colorScheme.onSurface == Colors.white
                      ? "assets/images/companies_dark.png"
                      : "assets/images/companies.png",
                  fit: BoxFit.fitHeight,
                ),
                Expanded(child: SizedBox()),
                Padding(
                  padding: Dimensions.bodyPadding,
                  child: Column(
                    children: [
                      Text(
                        "الاستثمار للجميع!",
                        style: Theme.of(context).textTheme.titleLarge,
                        textAlign: TextAlign.center,
                      ),
                      gap(height: 5),
                      Text(
                        "طوّرنا تقنيةً لتنمية أموالك واستثمارها تلقائيًا، ابدأ رحلة استثمارك الآن.",
                        style: Theme.of(context).textTheme.bodyMedium,
                        textAlign: TextAlign.center,
                      ),
                      gap(height: 40),
                      SizedBox(
                        width: 220,
                        height: 60,
                        child: ElevatedButton(
                          onPressed: () {},
                          child: Text("خلينا ندبل فلوسك"),
                        ),
                      ),
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
