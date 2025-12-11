"""
Enhanced Training Pipeline for V2
"""

import os
import yaml
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import sys
sys.path.append('.')

from Mycroft_Framework.rl_portfolio.environments import PortfolioEnv
from Mycroft_Framework.rl_portfolio.agents import (
    GrowthAgent, ValueAgent, RiskAgent, MycroftOrchestratorV2
)

class MycroftTrainerV2:
    def __init__(self, config_path="Mycroft_Framework/rl_portfolio/configs/config.yaml"):
        print("="*70)
        print("MYCROFT V2 TRAINING - ENHANCED ORCHESTRATOR")
        print("="*70)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"\n[Config] Tickers: {self.config['tickers']}")
        
        self.train_env = self._create_environment(split='train')
        self.test_env = self._create_environment(split='test')
        
        self.growth_agent = GrowthAgent(self.config['tickers'], self.train_env)
        self.value_agent = ValueAgent(self.config['tickers'], self.train_env)
        self.risk_agent = RiskAgent(self.config['tickers'], self.train_env)
        
        self.agents = [self.growth_agent, self.value_agent, self.risk_agent]
        
        # Use V2 orchestrator
        self.orchestrator = MycroftOrchestratorV2(self.agents, self.train_env)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"Mycroft_Framework/rl_portfolio/models/run_v2_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"\n[Setup] Models: {self.save_dir}")
        
    def _create_environment(self, split='train'):
        start = datetime.strptime(self.config['data']['start_date'], "%Y-%m-%d")
        end = datetime.strptime(self.config['data']['end_date'], "%Y-%m-%d")
        
        total_days = (end - start).days
        train_days = int(total_days * (1 - self.config['data']['test_split']))
        split_date = start + timedelta(days=train_days)
        
        if split == 'train':
            env_start = self.config['data']['start_date']
            env_end = split_date.strftime("%Y-%m-%d")
            print(f"\n[Env] TRAIN: {env_start} to {env_end}")
        else:
            env_start = split_date.strftime("%Y-%m-%d")
            env_end = self.config['data']['end_date']
            print(f"\n[Env] TEST: {env_start} to {env_end}")
        
        return PortfolioEnv(
            tickers=self.config['tickers'],
            start_date=env_start, end_date=env_end,
            initial_balance=self.config['portfolio']['initial_balance'],
            lookback_window=self.config['environment']['lookback_window'],
            transaction_cost=self.config['portfolio']['transaction_cost'],
            max_position_size=self.config['portfolio']['max_position_size'],
            risk_free_rate=self.config['environment']['risk_free_rate']
        )
    
    def train_all(self, agent_steps=30000, orch_steps=40000):
        print("\n" + "="*70)
        print("PHASE 1: TRAINING AGENTS (SAC)")
        print("="*70)
        
        for agent in self.agents:
            print(f"\n[{agent.name}] Training {agent_steps} steps...")
            agent.train(self.train_env, agent_steps)
            agent.save(self.save_dir)
            
            metrics = self._evaluate_agent(agent)
            print(f"[{agent.name}] Test: Return={metrics['total_return']*100:.2f}%, Sharpe={metrics['sharpe_ratio']:.3f}")
        
        print("\n" + "="*70)
        print("PHASE 2: TRAINING V2 ORCHESTRATOR (DQN + Tools)")
        print("="*70)
        
        self.orchestrator.train(orch_steps)
        self.orchestrator.save(self.save_dir)
        
        metrics = self._evaluate_orchestrator_v2()
        print(f"\n[OrchestratorV2] Test: Return={metrics['total_return']*100:.2f}%, Sharpe={metrics['sharpe_ratio']:.3f}")
        
        self._final_report()
    
    def _evaluate_agent(self, agent, n_ep=10):
        all_metrics = []
        for _ in range(n_ep):
            obs, _ = self.test_env.reset()
            done = False
            while not done:
                action = agent.get_action(obs, deterministic=True)
                obs, r, term, trunc, info = self.test_env.step(action)
                done = term or trunc
            m = self.test_env.get_portfolio_metrics()
            if m: all_metrics.append(m)
        
        if not all_metrics:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'volatility': 0, 'final_value': 100000}
        
        return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    
    def _evaluate_orchestrator_v2(self, n_ep=10):
        """Evaluate V2 orchestrator with its own environment"""
        from Mycroft_Framework.rl_portfolio.agents.orchestrator_v2 import OrchestrationEnvV2
        from Mycroft_Framework.rl_portfolio.tools import RegimeDetector, RiskCalculator
        
        all_metrics = []
        test_orch_env = OrchestrationEnvV2(
            self.agents, self.test_env, self.orchestrator.WEIGHT_STRATEGIES,
            RegimeDetector(), RiskCalculator()
        )
        
        for _ in range(n_ep):
            obs, _ = test_orch_env.reset()
            done = False
            while not done:
                action, _ = self.orchestrator.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = test_orch_env.step(action)
                done = term or trunc
            m = self.test_env.get_portfolio_metrics()
            if m: all_metrics.append(m)
        
        if not all_metrics:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'volatility': 0, 'final_value': 100000}
        
        return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    
    def _final_report(self):
        print("\n" + "="*70)
        print("FINAL V2 RESULTS")
        print("="*70)
        
        results = {}
        for agent in self.agents:
            results[agent.name] = self._evaluate_agent(agent)
        
        results['OrchestratorV2'] = self._evaluate_orchestrator_v2()
        
        print(f"\n{'Agent':<20} {'Return':<12} {'Sharpe':<10}")
        print("-"*42)
        for name, m in results.items():
            print(f"{name:<20} {m['total_return']*100:>10.2f}%  {m['sharpe_ratio']:>8.3f}")
        print("-"*42)
        
        import json
        with open(f"{self.save_dir}/results_v2.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved: {self.save_dir}/results_v2.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-steps', type=int, default=30000)
    parser.add_argument('--orch-steps', type=int, default=40000)
    args = parser.parse_args()
    
    trainer = MycroftTrainerV2()
    trainer.train_all(args.agent_steps, args.orch_steps)
