"""
Comprehensive Evaluation System
Compares RL agents against baselines with full analysis
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import json
from typing import Dict
import os

from Mycroft_Framework.rl_portfolio.environments import PortfolioEnv
from Mycroft_Framework.rl_portfolio.agents import (
    GrowthAgent, ValueAgent, RiskAgent, MycroftOrchestrator
)
from Mycroft_Framework.rl_portfolio.evaluation.baselines import (
    BuyAndHold, EqualWeight, MomentumStrategy, MinimumVariance, evaluate_baseline
)
from Mycroft_Framework.rl_portfolio.evaluation.visualizer import PortfolioVisualizer
from Mycroft_Framework.rl_portfolio.tools import RegimeDetector, RiskCalculator

class ComprehensiveEvaluator:
    """
    Run complete evaluation with:
    - All RL agents
    - All baseline strategies  
    - Statistical analysis
    - Visualizations
    - Risk metrics
    """
    
    def __init__(self, model_dir: str, test_env: PortfolioEnv, config: Dict):
        self.model_dir = model_dir
        self.test_env = test_env
        self.config = config
        self.results = {}
        
        # Initialize tools
        self.regime_detector = RegimeDetector()
        self.risk_calculator = RiskCalculator()
        self.visualizer = PortfolioVisualizer(model_dir)
        
        print("="*70)
        print("COMPREHENSIVE EVALUATION SYSTEM")
        print("="*70)
    
    def load_trained_agents(self) -> Dict:
        """Load all trained RL agents"""
        
        print("\n[Loading] Trained RL agents...")
        
        # Create dummy env for loading
        dummy_env = self.test_env
        
        # Load individual agents
        growth = GrowthAgent(self.config['tickers'], dummy_env)
        growth.load(self.model_dir)
        
        value = ValueAgent(self.config['tickers'], dummy_env)
        value.load(self.model_dir)
        
        risk = RiskAgent(self.config['tickers'], dummy_env)
        risk.load(self.model_dir)
        
        # Load orchestrator
        orchestrator = MycroftOrchestrator([growth, value, risk], dummy_env)
        orchestrator.load(self.model_dir)
        
        print("✓ All agents loaded")
        
        return {
            'GrowthAgent': growth,
            'ValueAgent': value,
            'RiskAgent': risk,
            'Orchestrator': orchestrator
        }
    
    def evaluate_all_strategies(self, n_episodes: int = 10) -> Dict:
        """Evaluate RL agents + baselines"""
        
        print("\n[Evaluation] Running all strategies...")
        
        # Load RL agents
        rl_agents = self.load_trained_agents()
        
        # Evaluate RL agents
        for name, agent in rl_agents.items():
            print(f"\n  Evaluating {name}...")
            if name == 'Orchestrator':
                metrics = self._evaluate_orchestrator(agent, n_episodes)
            else:
                metrics = self._evaluate_agent(agent, n_episodes)
            self.results[name] = metrics
        
        # Create baselines
        n_stocks = len(self.config['tickers'])
        baselines = {
            'Buy-and-Hold': BuyAndHold(n_stocks),
            'Equal-Weight': EqualWeight(n_stocks),
            'Momentum': MomentumStrategy(n_stocks),
            'Min-Variance': MinimumVariance(n_stocks)
        }
        
        # Evaluate baselines
        for name, baseline in baselines.items():
            print(f"\n  Evaluating {name} baseline...")
            metrics = evaluate_baseline(baseline, self.test_env, n_episodes)
            self.results[name] = metrics
        
        print("\n✓ All evaluations complete!")
        
        return self.results
    
    def _evaluate_agent(self, agent, n_episodes: int) -> Dict:
        """Evaluate single RL agent"""
        all_metrics = []
        
        for ep in range(n_episodes):
            obs, _ = self.test_env.reset()
            done = False
            
            while not done:
                action = agent.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.test_env.step(action)
                done = terminated or truncated
            
            metrics = self.test_env.get_portfolio_metrics()
            if metrics:
                all_metrics.append(metrics)
        
        if not all_metrics:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 
                    'max_drawdown': 0.0, 'volatility': 0.0,
                    'final_value': self.test_env.initial_balance}
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
        
        return avg_metrics
    
    def _evaluate_orchestrator(self, orchestrator, n_episodes: int) -> Dict:
        """Evaluate orchestrator"""
        from Mycroft_Framework.rl_portfolio.agents.orchestrator import OrchestrationEnv
        
        all_metrics = []
        test_orch_env = OrchestrationEnv(
            orchestrator.agents, self.test_env, orchestrator.WEIGHT_STRATEGIES
        )
        
        for ep in range(n_episodes):
            obs, _ = test_orch_env.reset()
            done = False
            
            while not done:
                action, _ = orchestrator.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_orch_env.step(action)
                done = terminated or truncated
            
            metrics = self.test_env.get_portfolio_metrics()
            if metrics:
                all_metrics.append(metrics)
        
        if not all_metrics:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 
                    'max_drawdown': 0.0, 'volatility': 0.0,
                    'final_value': self.test_env.initial_balance}
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
        
        return avg_metrics
    
    def generate_all_visualizations(self):
        """Create all charts"""
        
        print("\n[Visualizations] Creating charts...")
        
        self.visualizer.plot_performance_comparison(self.results)
        self.visualizer.plot_risk_return_scatter(self.results)
        self.visualizer.plot_drawdown_comparison(self.results)
        
        print("✓ All visualizations created!")
    
    def generate_analysis_report(self) -> str:
        """Generate comprehensive markdown report"""
        
        report = f"""# Mycroft RL Portfolio System - Evaluation Report

## Performance Summary

### RL Agents vs Baselines

"""
        
        # Create comparison table
        report += "| Strategy | Return | Sharpe | Max DD | Volatility | Type |\n"
        report += "|----------|--------|--------|--------|------------|------|\n"
        
        for name, metrics in sorted(self.results.items(), 
                                   key=lambda x: x[1]['sharpe_ratio'], 
                                   reverse=True):
            agent_type = "RL Agent" if name in ['GrowthAgent', 'ValueAgent', 'RiskAgent', 'Orchestrator'] else "Baseline"
            
            report += f"| {name:<20} | {metrics['total_return']*100:>6.2f}% | {metrics['sharpe_ratio']:>6.3f} | "
            report += f"{metrics['max_drawdown']*100:>6.2f}% | {metrics['volatility']*100:>6.2f}% | {agent_type} |\n"
        
        # Analysis
        report += "\n## Key Findings\n\n"
        
        # Best performer
        best = max(self.results.items(), key=lambda x: x[1]['sharpe_ratio'])
        report += f"**Best Risk-Adjusted Performance:** {best[0]} (Sharpe: {best[1]['sharpe_ratio']:.3f})\n\n"
        
        # RL vs Baseline comparison
        rl_sharpes = [self.results[n]['sharpe_ratio'] for n in ['GrowthAgent', 'ValueAgent', 'RiskAgent', 'Orchestrator']]
        baseline_sharpes = [self.results[n]['sharpe_ratio'] for n in ['Buy-and-Hold', 'Equal-Weight', 'Momentum', 'Min-Variance']]
        
        report += f"**RL Agents Average Sharpe:** {np.mean(rl_sharpes):.3f} ± {np.std(rl_sharpes):.3f}\n"
        report += f"**Baselines Average Sharpe:** {np.mean(baseline_sharpes):.3f} ± {np.std(baseline_sharpes):.3f}\n\n"
        
        if np.mean(rl_sharpes) > np.mean(baseline_sharpes):
            report += "✓ **RL agents outperform baseline strategies on average**\n\n"
        
        # Save report
        report_path = f"{self.model_dir}/EVALUATION_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Report saved: {report_path}")
        
        return report


def main(model_dir: str):
    """Run comprehensive evaluation"""
    
    import yaml
    
    # Load config
    with open('Mycroft_Framework/rl_portfolio/configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create test environment
    from datetime import datetime, timedelta
    start = datetime.strptime(config['data']['start_date'], "%Y-%m-%d")
    end = datetime.strptime(config['data']['end_date'], "%Y-%m-%d")
    total_days = (end - start).days
    train_days = int(total_days * (1 - config['data']['test_split']))
    split_date = start + timedelta(days=train_days)
    
    test_env = PortfolioEnv(
        tickers=config['tickers'],
        start_date=split_date.strftime("%Y-%m-%d"),
        end_date=config['data']['end_date'],
        initial_balance=config['portfolio']['initial_balance'],
        lookback_window=config['environment']['lookback_window'],
        transaction_cost=config['portfolio']['transaction_cost'],
        max_position_size=config['portfolio']['max_position_size'],
        risk_free_rate=config['environment']['risk_free_rate']
    )
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(model_dir, test_env, config)
    results = evaluator.evaluate_all_strategies(n_episodes=10)
    evaluator.generate_all_visualizations()
    report = evaluator.generate_analysis_report()
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"Results directory: {model_dir}")
    print("Generated files:")
    print("  - results.json")
    print("  - EVALUATION_REPORT.md")
    print("  - performance_comparison.png")
    print("  - risk_return_scatter.png")
    print("  - drawdown_comparison.png")
    print("="*70)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        # Use most recent model
        models_dir = "Mycroft_Framework/rl_portfolio/models"
        runs = [d for d in os.listdir(models_dir) if d.startswith('run_')]
        model_dir = os.path.join(models_dir, sorted(runs)[-1])
    
    main(model_dir)
