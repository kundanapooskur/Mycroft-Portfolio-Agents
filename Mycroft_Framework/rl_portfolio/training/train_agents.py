"""
Mycroft Training Pipeline
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
    GrowthAgent, ValueAgent, RiskAgent, MycroftOrchestrator
)

class MycroftTrainer:
    def __init__(self, config_path: str = "Mycroft_Framework/rl_portfolio/configs/config.yaml"):
        print("="*70)
        print("MYCROFT RL PORTFOLIO TRAINING SYSTEM")
        print("Multi-Agent Reinforcement Learning for AI Stock Portfolio")
        print("="*70)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"\n[Config] Loaded from {config_path}")
        print(f"[Config] Tickers: {self.config['tickers']}")
        
        self.train_env = self._create_environment(split='train')
        self.test_env = self._create_environment(split='test')
        
        self.growth_agent = GrowthAgent(self.config['tickers'], self.train_env)
        self.value_agent = ValueAgent(self.config['tickers'], self.train_env)
        self.risk_agent = RiskAgent(self.config['tickers'], self.train_env)
        
        self.agents = [self.growth_agent, self.value_agent, self.risk_agent]
        self.orchestrator = MycroftOrchestrator(self.agents, self.train_env)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"Mycroft_Framework/rl_portfolio/models/run_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"\n[Setup] Models will be saved to: {self.save_dir}")
        
    def _create_environment(self, split: str = 'train') -> PortfolioEnv:
        start = datetime.strptime(self.config['data']['start_date'], "%Y-%m-%d")
        end = datetime.strptime(self.config['data']['end_date'], "%Y-%m-%d")
        
        total_days = (end - start).days
        train_days = int(total_days * (1 - self.config['data']['test_split']))
        split_date = start + timedelta(days=train_days)
        
        if split == 'train':
            env_start = self.config['data']['start_date']
            env_end = split_date.strftime("%Y-%m-%d")
            print(f"\n[Environment] Creating TRAIN env: {env_start} to {env_end}")
        else:
            env_start = split_date.strftime("%Y-%m-%d")
            env_end = self.config['data']['end_date']
            print(f"\n[Environment] Creating TEST env: {env_start} to {env_end}")
        
        return PortfolioEnv(
            tickers=self.config['tickers'],
            start_date=env_start,
            end_date=env_end,
            initial_balance=self.config['portfolio']['initial_balance'],
            lookback_window=self.config['environment']['lookback_window'],
            transaction_cost=self.config['portfolio']['transaction_cost'],
            max_position_size=self.config['portfolio']['max_position_size'],
            risk_free_rate=self.config['environment']['risk_free_rate']
        )
    
    def train_individual_agents(self, timesteps_per_agent: int = 15000):
        print("\n" + "="*70)
        print("PHASE 1: TRAINING INDIVIDUAL AGENTS (SAC)")
        print("="*70)
        
        for agent in self.agents:
            print(f"\n{'='*70}")
            agent.train(self.train_env, timesteps=timesteps_per_agent)
            agent.save(self.save_dir)
            
            metrics = self._evaluate_agent(agent, self.test_env)
            print(f"\n[{agent.name}] Test Performance:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        
        print("\n[Phase 1] Individual agent training complete!")
    
    def train_orchestrator(self, timesteps: int = 20000):
        print("\n" + "="*70)
        print("PHASE 2: TRAINING ORCHESTRATOR (DQN Meta-Learning)")
        print("="*70)
        
        self.orchestrator.train(timesteps=timesteps)
        self.orchestrator.save(self.save_dir)
        
        metrics = self._evaluate_orchestrator(self.test_env)
        print(f"\n[Orchestrator] Test Performance:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("\n[Phase 2] Orchestrator training complete!")
    
    def _evaluate_agent(self, agent, env, n_episodes: int = 5) -> Dict:
        all_metrics = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action = agent.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            metrics = env.get_portfolio_metrics()
            if metrics:
                all_metrics.append(metrics)
        
        if not all_metrics:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 
                    'volatility': 0.0, 'final_value': env.initial_balance}
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def _evaluate_orchestrator(self, env, n_episodes: int = 5) -> Dict:
        all_metrics = []
        
        from Mycroft_Framework.rl_portfolio.agents.orchestrator import OrchestrationEnv
        test_orch_env = OrchestrationEnv(self.agents, env, self.orchestrator.WEIGHT_STRATEGIES)
        
        for episode in range(n_episodes):
            obs, _ = test_orch_env.reset()
            done = False
            
            while not done:
                action, _ = self.orchestrator.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_orch_env.step(action)
                done = terminated or truncated
            
            metrics = env.get_portfolio_metrics()
            if metrics:
                all_metrics.append(metrics)
        
        if not all_metrics:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
                    'volatility': 0.0, 'final_value': env.initial_balance}
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def _generate_comparison_report(self):
        print("\n" + "="*70)
        print("FINAL PERFORMANCE COMPARISON")
        print("="*70)
        
        results = {}
        
        for agent in self.agents:
            metrics = self._evaluate_agent(agent, self.test_env, n_episodes=10)
            results[agent.name] = metrics
        
        orch_metrics = self._evaluate_orchestrator(self.test_env, n_episodes=10)
        results['Orchestrator'] = orch_metrics
        
        print("\nAgent Performance Summary:")
        print("-" * 70)
        print(f"{'Agent':<20} {'Return':<12} {'Sharpe':<12} {'Max DD':<12} {'Volatility':<12}")
        print("-" * 70)
        
        for agent_name, metrics in results.items():
            print(f"{agent_name:<20} "
                  f"{metrics['total_return']*100:>10.2f}%  "
                  f"{metrics['sharpe_ratio']:>10.3f}  "
                  f"{metrics['max_drawdown']*100:>10.2f}%  "
                  f"{metrics['volatility']*100:>10.2f}%")
        
        print("-" * 70)
        
        import json
        with open(f"{self.save_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {self.save_dir}/results.json")
