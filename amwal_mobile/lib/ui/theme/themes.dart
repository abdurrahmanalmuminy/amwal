import 'package:flutter/material.dart';
import 'package:amwal_mobile/ui/theme/colors.dart';

class AppTheme {
  static ThemeData light = _baseTheme(Brightness.light);
  static ThemeData dark = _baseTheme(Brightness.dark);

  static ThemeData _baseTheme(Brightness brightness) {
    final isDark = brightness == Brightness.dark;

    final baseColor = isDark ? Colors.white : Colors.black;
    final bgColor = isDark
        ? AppColors.backgroundDark
        : AppColors.backgroundLight;

    return ThemeData(
      brightness: brightness,
      fontFamily: "Readex_Pro",
      splashFactory: NoSplash.splashFactory,
      highlightColor: Colors.transparent,
      splashColor: Colors.transparent,
      hoverColor: Colors.transparent,
      scaffoldBackgroundColor: bgColor,
      canvasColor: Colors.transparent,
      colorScheme: ColorScheme(
        brightness: brightness,
        primary: AppColors.primaryColor,
        onPrimary: isDark ? Colors.black : Colors.white,
        secondary: AppColors.primaryColor,
        onSecondary: isDark ? Colors.black : Colors.white,
        error: Colors.red,
        onError: Colors.white,
        surface: baseColor.withValues(alpha: 0.1),
        onSurface: baseColor,
      ),
      cardColor: isDark ? Color(0xFF08090D) : Colors.white,
      textTheme: const TextTheme(
        titleLarge: TextStyle(fontWeight: FontWeight.bold),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: baseColor.withValues(alpha: 0.05),
        hintStyle: TextStyle(color: baseColor.withValues(alpha: 0.5)),
        border: OutlineInputBorder(borderSide: BorderSide.none, borderRadius: BorderRadius.circular(20)),
      ),
      appBarTheme: AppBarTheme(
        iconTheme: IconThemeData(color: AppColors.primaryColor),
        backgroundColor: bgColor.withValues(alpha: 0),
        centerTitle: false,
      ),
      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        type: BottomNavigationBarType.fixed,
        backgroundColor: isDark ? Color(0xFF08090D).withValues(alpha: 0.85) : Colors.white.withValues(alpha: 0.85),
        elevation: 0,
      ),
      progressIndicatorTheme: ProgressIndicatorThemeData(
        linearTrackColor: baseColor.withValues(alpha: 0.2),
      ),
      dividerTheme: DividerThemeData(color: baseColor.withValues(alpha: 0.1)),
      listTileTheme: ListTileThemeData(
        selectedTileColor: AppColors.primaryColor.withValues(alpha: 0.1),
      ),
      iconButtonTheme: IconButtonThemeData(
        style: ButtonStyle(iconSize: WidgetStatePropertyAll(20)),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: ButtonStyle(foregroundColor: WidgetStatePropertyAll(baseColor)),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStatePropertyAll(AppColors.primaryColor),
          foregroundColor: WidgetStatePropertyAll(Colors.white),
          shadowColor: WidgetStatePropertyAll(
            AppColors.primaryColor.withValues(alpha: 0.20),
          ),
          elevation: WidgetStatePropertyAll(5),
          textStyle: const WidgetStatePropertyAll(
            TextStyle(
              fontFamily: "Readex_Pro",
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
        ),
      ),
      chipTheme: ChipThemeData(
        showCheckmark: false,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(100)),
        backgroundColor: baseColor.withValues(alpha: 0.05),
        iconTheme: IconThemeData(color: Colors.white),
        labelStyle: TextStyle(fontFamily: "Readex_Pro"),
        side: BorderSide.none,
      ),
      searchBarTheme: SearchBarThemeData(
        hintStyle: WidgetStatePropertyAll(
          TextStyle(color: baseColor.withValues(alpha: 0.5)),
        ),
        elevation: const WidgetStatePropertyAll(0),
        padding: const WidgetStatePropertyAll(
          EdgeInsets.symmetric(horizontal: 10),
        ),
        shape: WidgetStatePropertyAll(
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(100)),
        ),
        side: WidgetStatePropertyAll(
          BorderSide(color: baseColor.withValues(alpha: 0.25)),
        ),
      ),
      tabBarTheme: TabBarThemeData(
        dividerColor: baseColor.withValues(alpha: 0.15),
        overlayColor: const WidgetStatePropertyAll(Colors.transparent),
        tabAlignment: TabAlignment.center,
      ),
      disabledColor: isDark
          ? AppColors.primaryColor.withValues(alpha: 0.25)
          : null,
    );
  }
}
