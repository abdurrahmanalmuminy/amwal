import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

class WeeklySpending extends StatefulWidget {
  const WeeklySpending({super.key});

  @override
  State<WeeklySpending> createState() => _WeeklySpendingState();
}

class _WeeklySpendingState extends State<WeeklySpending> with SingleTickerProviderStateMixin {
  final List<double> targetSpending = [20, 35, 10, 45, 30, 50, 25];
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3), // مدة الأنميشن
    );

    _animation = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOut),
    )..addListener(() {
        setState(() {}); // نعيد بناء الرسم البياني مع كل تغير
      });

    _controller.forward(); // بدء الأنميشن
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  List<String> get weekDays => const [
        'الأحد',
        'الإثنين',
        'الثلاثاء',
        'الأربعاء',
        'الخميس',
        'الجمعة',
        'السبت',
      ];

  @override
  Widget build(BuildContext context) {
    // حساب القيم المتحركة حسب progress
    final animatedSpending = List.generate(
      targetSpending.length,
      (i) => targetSpending[i] * _animation.value,
    );

    return LineChart(
      LineChartData(
        minY: 0,
        maxY: 60,
        lineTouchData: LineTouchData(enabled: false),
        gridData: FlGridData(show: false),
        borderData: FlBorderData(show: false),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              interval: 1,
              getTitlesWidget: (value, meta) {
                int index = value.toInt();
                if (index >= 0 && index < weekDays.length) {
                  return Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: Text(
                      weekDays[index],
                      style: TextStyle(fontSize: 10, color: Colors.grey[700]),
                    ),
                  );
                }
                return const SizedBox.shrink();
              },
            ),
          ),
        ),
        lineBarsData: [
          LineChartBarData(
            isCurved: true,
            spots: List.generate(
              animatedSpending.length,
              (index) => FlSpot(index.toDouble(), animatedSpending[index]),
            ),
            color: Colors.blueAccent,
            barWidth: 3,
            isStrokeCapRound: true,
            dotData: FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              color: Colors.blueAccent.withOpacity(0.1),
            ),
          ),
        ],
      ),
    );
  }
}
