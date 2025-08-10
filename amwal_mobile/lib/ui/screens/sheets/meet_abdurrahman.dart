import 'package:amwal_mobile/ui/theme/dimentions.dart';
import 'package:amwal_mobile/ui/widgets/widgets.dart';
import 'package:flutter/material.dart';

void meetAbdurrahman(BuildContext context) {
  showModalBottomSheet<void>(
    context: context,
    backgroundColor: Theme.of(context).scaffoldBackgroundColor,
    isScrollControlled: true,
    builder: (BuildContext context) {
      return DraggableScrollableSheet(
        initialChildSize: 0.8,
        expand: false,
        builder: (context, scrollController) {
          return Scaffold(
            extendBody: true,
            extendBodyBehindAppBar: true,
            appBar: AppBar(title: Text("قابل عبدالرحمن")),
            body: Container(
              width: double.infinity,
              height: double.infinity,
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
                        ShaderMask(
                          blendMode: BlendMode
                              .srcIn, // ✅ ensures gradient replaces foreground color
                          shaderCallback: (Rect bounds) {
                            bool isDark =
                                Theme.of(context).colorScheme.onSecondary ==
                                Colors.black;
                            return LinearGradient(
                              colors: [
                                isDark ? Color(0xFFEFE4FF) : Color(0xFF7D3AEC),
                                isDark ? Color(0xFF8799FF) : Color(0xFF0026FF),
                              ],
                              begin: Alignment.topCenter,
                              end: Alignment.bottomLeft,
                            ).createShader(bounds);
                          },
                          child: Container(
                            padding: EdgeInsets.symmetric(
                              horizontal: 15,
                              vertical: 20,
                            ),
                            decoration: BoxDecoration(
                              border: Border.all(
                                width: 1,
                                color: Theme.of(
                                  context,
                                ).colorScheme.onSurface.withValues(alpha: 0.05),
                              ),
                              borderRadius: BorderRadius.circular(20),
                            ),
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
                                  "أنا عبدالرحمن، مدربك المالي الشخصي. سجّلت أن دخلك الشهري 7,000 ريال وهدفك شراء سيارة، وعندك التزامات شهرية مثل الإيجار واشتراك جوال. كل هذا سجلته عندي.",
                                  style: Theme.of(context).textTheme.titleMedium
                                      ?.copyWith(
                                        color: Colors
                                            .white, // ✅ white base for gradient to apply
                                      ),
                                  textAlign: TextAlign.center,
                                ),
                              ],
                            ),
                          ),
                        ),
                        gap(height: 20),
                        Text("من الآن، راح أساعدك:\n\n✅ تتابع مصروفاتك\n✅ تحقق أهدافك\n✅ وتتعلم كيف تبني استقرار مالي حقيقي\n\n راح أذكّرك، أوجّهك، وأعطيك خطط شخصية على مقاسك."),
                        Expanded(child: SizedBox()),

                      SizedBox(
                        width: 220,
                        height: 60,
                        child: ElevatedButton(
                          onPressed: () {
                            Navigator.pop(context);
                          },
                          child: Text("ابدأ الأن"),
                        ),
                      ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      );
    },
  );
}
