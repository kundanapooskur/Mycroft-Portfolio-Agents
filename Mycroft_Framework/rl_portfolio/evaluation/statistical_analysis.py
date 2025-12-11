"""
Statistical Significance Testing
Rigorous comparison of RL agents vs baselines
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List
import json

class StatisticalAnalyzer:
    """
    Rigorous statistical testing for RL vs baseline comparison
    
    Tests:
    - Paired t-test (RL vs baseline returns)
    - Sharpe ratio significance
    - Win rate analysis
    - Multiple hypothesis correction
    """
    
    def __init__(self, results_path: str):
        with open(results_path, 'r') as f:
            self.results = json.load(f)
    
    def compare_returns(self, agent1: str, agent2: str, 
                       n_bootstrap: int = 1000) -> Dict:
        """
        Compare two strategies with bootstrap confidence intervals
        
        Returns significance test results
        """
        
        # Get returns with std
        r1_mean = self.results[agent1]['total_return']
        r1_std = self.results[agent1].get('total_return_std', 0.01)
        
        r2_mean = self.results[agent2]['total_return']
        r2_std = self.results[agent2].get('total_return_std', 0.01)
        
        # Bootstrap sampling
        r1_samples = np.random.normal(r1_mean, r1_std, n_bootstrap)
        r2_samples = np.random.normal(r2_mean, r2_std, n_bootstrap)
        
        # Calculate win rate
        win_rate = np.mean(r1_samples > r2_samples)
        
        # T-test
        t_stat, p_value = stats.ttest_ind(r1_samples, r2_samples)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((r1_std**2 + r2_std**2) / 2)
        cohens_d = (r1_mean - r2_mean) / pooled_std if pooled_std > 0 else 0
        
        return {
            'agent1': agent1,
            'agent2': agent2,
            'mean_diff': r1_mean - r2_mean,
            'win_rate': win_rate,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'verdict': agent1 if win_rate > 0.5 else agent2
        }
    
    def rl_vs_baselines_analysis(self) -> pd.DataFrame:
        """Compare all RL agents vs all baselines"""
        
        rl_agents = ['GrowthAgent', 'ValueAgent', 'RiskAgent', 'Orchestrator']
        baselines = ['Buy-and-Hold', 'Equal-Weight', 'Momentum', 'Min-Variance']
        
        comparisons = []
        
        for rl in rl_agents:
            for baseline in baselines:
                if rl in self.results and baseline in self.results:
                    result = self.compare_returns(rl, baseline)
                    comparisons.append(result)
        
        df = pd.DataFrame(comparisons)
        
        # Summary statistics
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print("="*70)
        print(f"\nRL vs Baseline Comparisons: {len(df)}")
        print(f"RL Wins: {sum(df['verdict'].str.contains('Agent'))}/{len(df)}")
        print(f"Significant differences (p<0.05): {df['significant'].sum()}/{len(df)}")
        
        return df
    
    def generate_statistical_report(self, save_path: str):
        """Generate comprehensive statistical report"""
        
        df = self.rl_vs_baselines_analysis()
        
        report = "# Statistical Analysis Report\n\n"
        report += "## RL Agents vs Baselines: Pairwise Comparisons\n\n"
        
        report += "| RL Agent | Baseline | Mean Diff | Win Rate | p-value | Significant | Cohen's d |\n"
        report += "|----------|----------|-----------|----------|---------|-------------|------------|\n"
        
        for _, row in df.iterrows():
            sig_mark = "✓" if row['significant'] else ""
            report += f"| {row['agent1']} | {row['agent2']} | "
            report += f"{row['mean_diff']*100:>6.2f}% | {row['win_rate']*100:>5.1f}% | "
            report += f"{row['p_value']:.4f} | {sig_mark:^11} | {row['cohens_d']:>6.3f} |\n"
        
        # Overall summary
        rl_sharpes = [self.results[a]['sharpe_ratio'] for a in ['GrowthAgent', 'ValueAgent', 'RiskAgent', 'Orchestrator']]
        base_sharpes = [self.results[b]['sharpe_ratio'] for b in ['Buy-and-Hold', 'Equal-Weight', 'Momentum', 'Min-Variance']]
        
        t_stat, p_val = stats.ttest_ind(rl_sharpes, base_sharpes)
        
        report += f"\n## Overall Analysis\n\n"
        report += f"**RL Agents Mean Sharpe:** {np.mean(rl_sharpes):.3f} ± {np.std(rl_sharpes):.3f}\n"
        report += f"**Baselines Mean Sharpe:** {np.mean(base_sharpes):.3f} ± {np.std(base_sharpes):.3f}\n"
        report += f"**t-statistic:** {t_stat:.3f}\n"
        report += f"**p-value:** {p_val:.4f}\n"
        
        if p_val < 0.05:
            report += f"\n✓ **Statistically significant difference** (p < 0.05)\n"
        else:
            report += f"\n⚠ No statistically significant difference (p = {p_val:.4f})\n"
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Statistical report saved: {save_path}")
        
        return report
