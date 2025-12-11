"""
Baseline Strategies for Comparison
Implements buy-and-hold, equal-weight, and momentum strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List

class BaselineStrategy:
    """Base class for baseline strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

class BuyAndHold(BaselineStrategy):
    """Buy and hold equal-weighted portfolio"""
    
    def __init__(self, n_stocks: int):
        super().__init__("Buy-and-Hold")
        self.n_stocks = n_stocks
        self.weights = np.ones(n_stocks) / n_stocks
    
    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        return self.weights

class EqualWeight(BaselineStrategy):
    """Equal-weight rebalanced portfolio"""
    
    def __init__(self, n_stocks: int):
        super().__init__("Equal-Weight")
        self.n_stocks = n_stocks
        self.weights = np.ones(n_stocks) / n_stocks
    
    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        return self.weights

class MomentumStrategy(BaselineStrategy):
    """Momentum-based portfolio (overweight recent winners)"""
    
    def __init__(self, n_stocks: int):
        super().__init__("Momentum")
        self.n_stocks = n_stocks
    
    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        # Extract returns from observation
        # State: [prices(n), holdings(n), cash(1), returns(n), volatility(n)]
        returns_start = self.n_stocks * 2 + 1
        returns_end = returns_start + self.n_stocks
        
        recent_returns = observation[returns_start:returns_end]
        
        # Rank by returns
        ranks = np.argsort(recent_returns)[::-1]
        
        # Weights: exponential decay by rank
        weights = np.zeros(self.n_stocks)
        for i, rank in enumerate(ranks):
            weights[rank] = np.exp(-i * 0.5)
        
        # Normalize
        return weights / weights.sum()

class MinimumVariance(BaselineStrategy):
    """Minimum variance portfolio"""
    
    def __init__(self, n_stocks: int):
        super().__init__("Min-Variance")
        self.n_stocks = n_stocks
    
    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        # Extract volatility from observation
        vol_start = self.n_stocks * 3 + 1
        volatility = observation[vol_start:]
        
        # Inverse volatility weighting
        inv_vol = 1.0 / (volatility + 1e-8)
        weights = inv_vol / inv_vol.sum()
        
        return weights


def evaluate_baseline(strategy: BaselineStrategy, env, n_episodes: int = 10) -> Dict:
    """Evaluate a baseline strategy"""
    
    all_metrics = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action = strategy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        metrics = env.get_portfolio_metrics()
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        return {'total_return': 0.0, 'sharpe_ratio': 0.0, 
                'max_drawdown': 0.0, 'volatility': 0.0}
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics
