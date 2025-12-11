import sys
sys.path.append('.')
from Mycroft_Framework.rl_portfolio.evaluation.statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer('Mycroft_Framework/rl_portfolio/models/run_20251210_165354/results.json')
df = analyzer.rl_vs_baselines_analysis()

print("\nTop RL Victories:")
top_wins = df[df['verdict'].str.contains('Agent')].nlargest(5, 'mean_diff')
print(top_wins[['agent1', 'agent2', 'mean_diff', 'p_value', 'significant']])

analyzer.generate_statistical_report('Mycroft_Framework/rl_portfolio/models/run_20251210_165354/STATISTICAL_ANALYSIS.md')
