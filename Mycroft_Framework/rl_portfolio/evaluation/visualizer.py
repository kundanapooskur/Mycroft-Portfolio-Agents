"""
Visualization Tools for Mycroft RL Portfolio
Creates charts and plots for presentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json

class PortfolioVisualizer:
    """Create professional visualizations for results"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_performance_comparison(self, results: Dict, 
                                   save_path: str = None):
        """Bar chart comparing all agents and baselines"""
        
        if save_path is None:
            save_path = f"{self.save_dir}/performance_comparison.png"
        
        agents = list(results.keys())
        returns = [results[a]['total_return'] * 100 for a in agents]
        sharpes = [results[a]['sharpe_ratio'] for a in agents]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Returns
        colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in returns]
        ax1.barh(agents, returns, color=colors, alpha=0.8)
        ax1.set_xlabel('Total Return (%)', fontsize=12)
        ax1.set_title('Portfolio Returns Comparison', fontsize=14, fontweight='bold')
        ax1.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax1.grid(axis='x', alpha=0.3)
        
        # Sharpe Ratios
        colors2 = ['#3498db' if s > 1 else '#95a5a6' for s in sharpes]
        ax2.barh(agents, sharpes, color=colors2, alpha=0.8)
        ax2.set_xlabel('Sharpe Ratio', fontsize=12)
        ax2.set_title('Risk-Adjusted Returns', fontsize=14, fontweight='bold')
        ax2.axvline(1.0, color='red', linestyle='--', linewidth=0.8, label='Benchmark')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_risk_return_scatter(self, results: Dict,
                                save_path: str = None):
        """Risk-return scatter plot"""
        
        if save_path is None:
            save_path = f"{self.save_dir}/risk_return_scatter.png"
        
        agents = list(results.keys())
        returns = [results[a]['total_return'] * 100 for a in agents]
        vols = [results[a]['volatility'] * 100 for a in agents]
        sharpes = [results[a]['sharpe_ratio'] for a in agents]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(vols, returns, s=300, c=sharpes, 
                           cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=2)
        
        for i, agent in enumerate(agents):
            ax.annotate(agent, (vols[i], returns[i]), 
                       fontsize=10, ha='center', va='bottom')
        
        ax.set_xlabel('Volatility (%)', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_drawdown_comparison(self, results: Dict,
                                save_path: str = None):
        """Compare maximum drawdowns"""
        
        if save_path is None:
            save_path = f"{self.save_dir}/drawdown_comparison.png"
        
        agents = list(results.keys())
        drawdowns = [results[a]['max_drawdown'] * 100 for a in agents]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Reds([abs(d)/max(abs(min(drawdowns)), 1) for d in drawdowns])
        ax.barh(agents, drawdowns, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel('Maximum Drawdown (%)', fontsize=12)
        ax.set_title('Downside Risk Comparison', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def create_summary_table(self, results: Dict) -> pd.DataFrame:
        """Create formatted summary table"""
        
        data = []
        for agent, metrics in results.items():
            data.append({
                'Agent': agent,
                'Return (%)': f"{metrics['total_return']*100:.2f}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                'Max Drawdown (%)': f"{metrics['max_drawdown']*100:.2f}",
                'Volatility (%)': f"{metrics['volatility']*100:.2f}",
                'Final Value ($)': f"{metrics['final_value']:,.2f}"
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_path = f"{self.save_dir}/results_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")
        
        return df
