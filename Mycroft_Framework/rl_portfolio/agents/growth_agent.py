"""
Growth Agent - Focuses on high-growth AI companies
Uses SAC to maximize returns with momentum signals
"""

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from .base_agent import BasePortfolioAgent

class GrowthAgent(BasePortfolioAgent):
    """
    Growth-focused portfolio agent
    Objective: Maximize returns through high-growth AI stocks
    Emphasizes: Semiconductors (NVDA, AMD, AVGO) and Cloud (AMZN, MSFT)
    """
    
    def __init__(self, tickers: list, env):
        super().__init__("GrowthAgent", tickers)
        self.env = env
        
        # Growth stocks preference (higher weights for semiconductors/cloud)
        self.growth_preference = self._get_growth_preference()
        
        # Initialize SAC model
        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            verbose=1,
            tensorboard_log="./Mycroft_Framework/rl_portfolio/logs/growth/"
        )
        
    def _get_growth_preference(self) -> np.ndarray:
        """
        Define growth stock preferences
        Higher weights for semiconductors and cloud infrastructure
        """
        # Default equal weights
        weights = np.ones(self.n_stocks) / self.n_stocks
        
        # Boost semiconductor and cloud stocks
        growth_tickers = ['NVDA', 'AMD', 'AVGO', 'AMZN', 'MSFT', 'GOOGL', 'MU', 'MRVL']
        
        for i, ticker in enumerate(self.tickers):
            if ticker in growth_tickers:
                weights[i] *= 1.5
        
        # Normalize
        weights /= weights.sum()
        return weights
    
    def get_action(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get portfolio allocation action"""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        
        # Apply growth bias
        action = action * self.growth_preference
        action = np.clip(action, 0, 1)
        
        # Normalize to sum to 1
        if action.sum() > 0:
            action = action / action.sum()
        else:
            action = self.growth_preference.copy()
        
        return action
    
    def train(self, env, timesteps: int):
        """Train the growth agent"""
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


class AgentCallback(BaseCallback):
    """Callback to track agent performance during training"""
    
    def __init__(self, agent: BasePortfolioAgent):
        super().__init__()
        self.agent = agent
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Track rewards
        if len(self.locals.get('rewards', [])) > 0:
            reward = self.locals['rewards'][0]
            self.agent.update_performance(reward)
        
        return True
