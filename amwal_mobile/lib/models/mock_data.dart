class TransactionClass {
  final String date;
  final String description;
  final int amount;

  TransactionClass({
    required this.date,
    required this.description,
    required this.amount,
  });

  factory TransactionClass.fromJson(Map<String, dynamic> json) {
    return TransactionClass(
      date: json['date'],
      description: json['description'],
      amount: json['amount'],
    );
  }

  Map<String, dynamic> toJson() {
    return {'date': date, 'description': description, 'amount': amount};
  }
}

class MockData {
  String name;
  int monthlyIncome;
  List<String> financialGoals;
  final int currentSavings;
  List<String> monthlyCommitments;
  final String financialDisciplineScore;
  final String creditScore;
  final List<TransactionClass> transactions;

  MockData({
    required this.name,
    required this.monthlyIncome,
    required this.financialGoals,
    required this.currentSavings,
    required this.monthlyCommitments,
    required this.financialDisciplineScore,
    required this.creditScore,
    required this.transactions,
  });

  factory MockData.fromJson(Map<String, dynamic> json) {
    return MockData(
      name: json['name'],
      monthlyIncome: json['monthly_income'],
      financialGoals: json['financial_goal'],
      currentSavings: json['current_savings'],
      monthlyCommitments: json['monthly_commitments'],
      financialDisciplineScore: json['financial_discipline_score'],
      creditScore: json['credit_score'],
      transactions: (json['transactions'] as List)
          .map((transaction) => TransactionClass.fromJson(transaction))
          .toList(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'monthly_income': monthlyIncome,
      'financial_goal': financialGoals,
      'current_savings': currentSavings,
      'monthly_commitments': monthlyCommitments,
      'financial_discipline_score': financialDisciplineScore,
      'credit_score': creditScore,
      'transactions': transactions
          .map((transaction) => transaction.toJson())
          .toList(),
    };
  }
}

MockData mockData = MockData(
  name: "عبدالله",
  monthlyIncome: 9000,
  financialGoals: ["شراء سيارة جديدة خلال 6 أشهر"],
  currentSavings: 5000,
  monthlyCommitments: ["ايجار شقة"],
  financialDisciplineScore: "86%",
  creditScore: "23%",
  transactions: [
    TransactionClass(
      date: "2025-07-01",
      description: "راتب شهري",
      amount: 9000,
    ),
    TransactionClass(
      date: "2025-07-02",
      description: "إيجار شقة",
      amount: -2500,
    ),
    TransactionClass(
      date: "2025-07-03",
      description: "فواتير كهرباء وماء",
      amount: -400,
    ),
    TransactionClass(
      date: "2025-07-04",
      description: "قهوة يومية",
      amount: -45,
    ),
    TransactionClass(date: "2025-07-06", description: "مطعم", amount: -120),
    TransactionClass(
      date: "2025-07-08",
      description: "ادخار تلقائي",
      amount: -800,
    ),
  ],
);
