import 'package:flutter/material.dart';
import 'package:uicons/uicons.dart';

class InsightModel {
  final String title;
  final String value;
  final IconData icon;
  final Color color;

  InsightModel({
    required this.title,
    required this.value,
    required this.icon,
    required this.color,
  });
}

List<InsightModel> insights = [
  InsightModel(
    title: "الصحة المالية",
    value: "86%",
    icon: UIcons.solidRounded.user,
    color: Colors.green,
  ),
  InsightModel(
    title: "مؤشر المخاطرة",
    value: "22%",
    icon: UIcons.solidRounded.info,
    color: Colors.red,
  ),
  InsightModel(
    title: "الإستثمار",
    value: "3,681",
    icon: UIcons.solidRounded.coins,
    color: Colors.deepPurpleAccent,
  ),
  InsightModel(
    title: "النفقات",
    value: "1,400",
    icon: UIcons.solidRounded.stats,
    color: Colors.orange,
  ),
];
