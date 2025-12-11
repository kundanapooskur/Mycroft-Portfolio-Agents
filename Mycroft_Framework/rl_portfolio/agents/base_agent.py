"""
Base Agent Class for Mycroft Portfolio Agents
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BasePortfolioAgent(ABC):
    """
    Abstract base class for portfolio agents
    Each agent specializes in different investment strategies
    """
    
    def __init__(self, name: str, tickers: list):
        self.name = name
        self.tickers = tickers
        self.n_stocks = len(tickers)
        self.performance_history = []
        
    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Get portfolio allocation action based on observation
        Returns: weights [0-1] for each stock
        """
        pass
    
    @abstractmethod
    def train(self, env, timesteps: int):
        """Train the agent"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save agent model"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load agent model"""
        pass
    
    def update_performance(self, reward: float):
        """Track agent performance"""
        self.performance_history.append(reward)
    
    def get_recent_performance(self, window: int = 100) -> float:
        """Get average performance over recent window"""
        if len(self.performance_history) < window:
            window = len(self.performance_history)
        
        if window == 0:
            return 0.0
        
        return np.mean(self.performance_history[-window:])
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'n_stocks': self.n_stocks,
            'avg_performance': self.get_recent_performance()
        }
