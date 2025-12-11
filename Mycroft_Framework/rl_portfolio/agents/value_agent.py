"""
Value Agent - Focuses on undervalued AI companies
Uses SAC with emphasis on fundamental value stocks
"""

import numpy as np
from stable_baselines3 import SAC
from .base_agent import BasePortfolioAgent
from .growth_agent import AgentCallback

class ValueAgent(BasePortfolioAgent):
    """
    Value-focused portfolio agent
    Objective: Undervalued AI companies with strong fundamentals
    Emphasizes: Established companies (MSFT, GOOGL, CRM, NOW)
    """
    
    def __init__(self, tickers: list, env):
        super().__init__("ValueAgent", tickers)
        self.env = env
        
        # Value stocks preference
        self.value_preference = self._get_value_preference()
        
        # Initialize SAC model with different hyperparameters
        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.01,  # Slower updates for stability
            gamma=0.995,  # Higher discount for long-term focus
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            verbose=1,
            tensorboard_log="./Mycroft_Framework/rl_portfolio/logs/value/"
        )
        
    def _get_value_preference(self) -> np.ndarray:
        """
        Define value stock preferences
        Higher weights for established companies with strong fundamentals
        """
        weights = np.ones(self.n_stocks) / self.n_stocks
        
        # Boost established value stocks
        value_tickers = ['MSFT', 'GOOGL', 'AMZN', 'CRM', 'NOW']
        
        for i, ticker in enumerate(self.tickers):
            if ticker in value_tickers:
                weights[i] *= 1.5
        
        weights /= weights.sum()
        return weights
    
    def get_action(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get portfolio allocation action with value bias"""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        
        # Apply value bias
        action = action * self.value_preference
        action = np.clip(action, 0, 1)
        
        if action.sum() > 0:
            action = action / action.sum()
        else:
            action = self.value_preference.copy()
        
        return action
    
    def train(self, env, timesteps: int):
        """Train the value agent"""
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
