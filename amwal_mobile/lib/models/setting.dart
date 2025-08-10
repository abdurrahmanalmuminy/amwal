import 'package:flutter/widgets.dart';
import 'package:uicons/uicons.dart';

class SettingClass {
  final String label;
  final IconData icon;

  SettingClass(this.label, this.icon);
}

List<SettingClass> shourtcuts = [
  SettingClass("العملة", UIcons.regularRounded.credit_card),
  SettingClass("اللغة", UIcons.regularRounded.globe),
];

List<SettingClass> info = [
  SettingClass("اعزم اخوياك", UIcons.regularRounded.gift),
  SettingClass("قيمنا على متجر التطبيقات", UIcons.regularRounded.star),
  SettingClass("تواصل معنا", UIcons.regularRounded.headset),
  SettingClass("مجتمع أموال", UIcons.regularRounded.users),
];

List<SettingClass> policies = [
  SettingClass("سياسة الخصوصية", UIcons.regularRounded.shield_check),
  SettingClass("الشروط والأحكام", UIcons.regularRounded.document_signed),
];

List<SettingClass> account = [
  SettingClass("تسجيل الخروج", UIcons.regularRounded.sign_out_alt),
];

List<List<SettingClass>> settingsList = [
  shourtcuts,
  info,
  policies,
  account,
];
