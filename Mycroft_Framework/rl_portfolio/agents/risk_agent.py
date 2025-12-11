"""
Risk Agent - Focuses on portfolio protection and volatility management
Uses SAC to minimize drawdown and manage tail risk
"""

import numpy as np
from stable_baselines3 import SAC
from .base_agent import BasePortfolioAgent
from .growth_agent import AgentCallback

class RiskAgent(BasePortfolioAgent):
    """
    Risk management focused agent
    Objective: Portfolio protection and volatility management
    Strategy: Lower volatility stocks, diversification, drawdown minimization
    """
    
    def __init__(self, tickers: list, env):
        super().__init__("RiskAgent", tickers)
        self.env = env
        
        # Risk-adjusted preference (favor diversification)
        self.risk_preference = self._get_risk_preference()
        
        # Initialize SAC with conservative parameters
        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=1e-4,  # Lower learning rate
            buffer_size=100000,
            learning_starts=2000,  # More data before training
            batch_size=256,
            tau=0.02,  # Very slow updates
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef=0.2,  # Lower entropy for stability
            verbose=1,
            tensorboard_log="./Mycroft_Framework/rl_portfolio/logs/risk/"
        )
        
    def _get_risk_preference(self) -> np.ndarray:
        """
        Define risk-adjusted preferences
        Equal weights for maximum diversification
        """
        # Start with equal diversification
        weights = np.ones(self.n_stocks) / self.n_stocks
        
        # Slightly favor stable large-caps
        stable_tickers = ['MSFT', 'AMZN', 'GOOGL', 'NOW']
        
        for i, ticker in enumerate(self.tickers):
            if ticker in stable_tickers:
                weights[i] *= 1.2
        
        weights /= weights.sum()
        return weights
    
    def get_action(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get portfolio allocation action with risk management bias"""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        
        # Apply risk preference
        action = action * self.risk_preference
        action = np.clip(action, 0, 1)
        
        # Enforce diversification - no position > 20%
        max_position = 0.20
        action = np.clip(action, 0, max_position)
        
        if action.sum() > 0:
            action = action / action.sum()
        else:
            action = self.risk_preference.copy()
        
        return action
    
    def train(self, env, timesteps: int):
        """Train the risk agent"""
        print(f"\n[{self.name}] Starting training for {timesteps} timesteps...")
        
        callback = AgentCallback(self)
        self.model.learn(
            total_timesteps=timesteps,
            callback=callback,
            progress_bar=True
        )
        
        print(f"[{self.name}] Training complete!")
    
    def save(self, path: str):
        """Save agent model"""
        self.model.save(f"{path}/{self.name}_model")
        print(f"[{self.name}] Model saved to {path}")
    
    def load(self, path: str):
        """Load agent model"""
        self.model = SAC.load(f"{path}/{self.name}_model", env=self.env)
        print(f"[{self.name}] Model loaded from {path}")
