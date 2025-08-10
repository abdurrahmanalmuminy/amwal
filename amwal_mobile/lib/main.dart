import 'package:amwal_mobile/l10/l10n.dart';
import 'package:amwal_mobile/providers/locale_provider.dart';
import 'package:amwal_mobile/providers/theme_provider.dart';
import 'package:amwal_mobile/ui/screens/onboarding/welcome.dart';
import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:provider/provider.dart';
import 'package:amwal_mobile/ui/theme/themes.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    // await Firebase.initializeApp(
    //   options: DefaultFirebaseOptions.currentPlatform,
    // );
    // final userProvider = UserProvider();
    // await userProvider.loadUserFromPreferences();
    runApp(
      MultiProvider(
        providers: [
          // ChangeNotifierProvider<UserProvider>.value(value: userProvider),
          ChangeNotifierProvider(create: (_) => LocaleProvider()),
          ChangeNotifierProvider(create: (_) => ThemeProvider()),
        ],
        //child: MyApp(userProvider: userProvider),
        child: MyApp(),
      ),
    );
  } catch (e) {
    // final ErrorHandler errorHandler = ErrorHandler();
    // errorHandler.recordError(e, stackTrace);
    // debugPrint("Initialization error: $e");
    // debugPrint("Stacktrace: $stackTrace");

    // Wrap ErrorFallbackApp with necessary providers
    runApp(
      MultiProvider(
        providers: [
          ChangeNotifierProvider(create: (_) => LocaleProvider()),
          ChangeNotifierProvider(create: (_) => ThemeProvider()),
        ],
        child: const ErrorFallbackApp(),
      ),
    );
  }
}

class MyApp extends StatelessWidget {
  //final UserProvider userProvider;
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    // final userProvider = Provider.of<UserProvider>(context);
    // final UserDTO? currentUser = userProvider.currentUser;

    final themeProvider = Provider.of<ThemeProvider>(context);

    return MaterialApp(
      title: "أموال | amwal",
      theme: AppTheme.light,
      darkTheme: AppTheme.dark,
      themeMode: themeProvider.themeMode,
      localizationsDelegates: const [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      home: Welcome(),
      // home: FutureBuilder(
      //   future: Future.value(AuthController().auth.currentUser),
      //   builder: (context, snapshot) {
      //     if (snapshot.connectionState == ConnectionState.waiting) {
      //       return const Center(child: CircularProgressIndicator.adaptive());
      //     }

      //     final firebaseUser = snapshot.data;

      //     if (firebaseUser != null && currentUser != null) {
      //       // User is logged in and saved in shared preferences
      //       return Navigation();
      //     } else if (firebaseUser != null && currentUser == null) {
      //       // User is logged in to Firebase but not saved in shared preferences
      //       return PersonalInfo(
      //         userId: firebaseUser.uid,
      //         phoneNumber: firebaseUser.phoneNumber ?? "",
      //       );
      //     } else {
      //       // Default to the Welcome page
      //       return Welcome();
      //     }
      //   },
      // ),
      supportedLocales: L10n.all,
      locale: const Locale('ar'),
    );
  }
}

class ErrorFallbackApp extends StatelessWidget {
  const ErrorFallbackApp({super.key});

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);

    return MaterialApp(
      title: "Error fall back",
      theme: AppTheme.light,
      darkTheme: AppTheme.dark,
      themeMode: themeProvider.themeMode,
      supportedLocales: L10n.all,
      locale: Provider.of<LocaleProvider>(context).currentLocale,
      localizationsDelegates: const [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
    );
  }
}
