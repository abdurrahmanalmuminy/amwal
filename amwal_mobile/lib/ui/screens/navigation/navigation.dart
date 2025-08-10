import 'dart:ui';

import 'package:amwal_mobile/models/mock_data.dart';
import 'package:amwal_mobile/ui/screens/navigation/expenses.dart';
import 'package:amwal_mobile/ui/screens/navigation/home.dart';
import 'package:amwal_mobile/ui/screens/navigation/investment.dart';
import 'package:amwal_mobile/ui/screens/navigation/library.dart';
import 'package:amwal_mobile/ui/screens/navigation/settings.dart';
import 'package:flutter/material.dart';
import 'package:uicons/uicons.dart';

class Navigation extends StatefulWidget {
  const Navigation({super.key});

  @override
  State<Navigation> createState() => _NavigationState();
}

class _NavigationState extends State<Navigation> {
  int currentIndex = 0;
  List<Widget> page = [Home(), Investment(), Expenses(), Library(), Settings()];

  @override
  Widget build(BuildContext context) {
    print(mockData.toJson());
    return Scaffold(
      extendBody: true,
      body: IndexedStack(index: currentIndex, children: page),
      bottomNavigationBar: ClipRect(
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 2, sigmaY: 2),
          child: Container(
            decoration: BoxDecoration(
              border: Border(
                top: BorderSide(color: Theme.of(context).colorScheme.surface),
              ),
            ),
            child: BottomNavigationBar(
              currentIndex: currentIndex,
              iconSize: 20,
              unselectedFontSize: 12,
              selectedFontSize: 12,
              onTap: (index) {
                setState(() {
                  currentIndex = index;
                });
              },
              items: [
                BottomNavigationBarItem(
                  icon: Icon(UIcons.solidRounded.home),
                  label: "الرئيسية",
                ),
                BottomNavigationBarItem(
                  icon: Icon(UIcons.solidRounded.coins),
                  label: "الاستثمار",
                ),
                BottomNavigationBarItem(
                  icon: Icon(UIcons.solidRounded.stats),
                  label: "النفقات",
                ),
                BottomNavigationBarItem(
                  icon: Icon(UIcons.solidRounded.apps),
                  label: "المكتبة",
                ),
                BottomNavigationBarItem(
                  icon: Icon(UIcons.solidRounded.settings),
                  label: "الاعدادات",
                ),
              ],
            ),
          ),
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.endFloat,
      floatingActionButton: currentIndex != 0
          ? null
          : FloatingActionButton(
              onPressed: () {},
              shape: CircleBorder(),
              child: Icon(UIcons.regularRounded.plus, color: Colors.white),
            ),
    );
  }
}
