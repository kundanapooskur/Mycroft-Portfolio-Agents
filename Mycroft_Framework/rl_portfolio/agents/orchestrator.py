"""
Mycroft Orchestrator - Meta-Learning Agent
Uses DQN to learn optimal coordination between specialized agents
"""

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Tuple
import gymnasium as gym

class MycroftOrchestrator:
    """Meta-Agent that coordinates Growth, Value, and Risk agents"""
    
    WEIGHT_STRATEGIES = [
        [0.7, 0.2, 0.1],  # Growth-focused
        [0.5, 0.4, 0.1],  # Growth-value balanced
        [0.2, 0.7, 0.1],  # Value-focused
        [0.4, 0.4, 0.2],  # Balanced
        [0.3, 0.3, 0.4],  # Risk-focused
        [0.2, 0.2, 0.6],  # High risk management
        [0.6, 0.3, 0.1],  # Aggressive growth
        [0.1, 0.4, 0.5],  # Conservative
    ]
    
    def __init__(self, agents: List, env):
        self.agents = agents
        self.env = env
        self.n_agents = len(agents)
        self.orch_env = OrchestrationEnv(agents, env, self.WEIGHT_STRATEGIES)
        
        self.model = DQN(
            "MlpPolicy", self.orch_env,
            learning_rate=1e-4, buffer_size=50000, learning_starts=1000,
            batch_size=128, tau=0.005, gamma=0.99, train_freq=4,
            exploration_fraction=0.3, exploration_initial_eps=1.0,
            exploration_final_eps=0.05, verbose=1,
            tensorboard_log="./Mycroft_Framework/rl_portfolio/logs/orchestrator/"
        )
        
        print(f"[Orchestrator] Initialized with {self.n_agents} agents")
        print(f"[Orchestrator] Action space: {len(self.WEIGHT_STRATEGIES)} weight strategies")
    
    def get_action(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        strategy_idx, _ = self.model.predict(observation, deterministic=deterministic)
        weights = np.array(self.WEIGHT_STRATEGIES[strategy_idx])
        
        base_obs = observation[:self.env.observation_space.shape[0]] if len(observation) > self.env.observation_space.shape[0] else observation
        
        agent_actions = [agent.get_action(base_obs, deterministic=True) for agent in self.agents]
        combined_action = sum(w * a for w, a in zip(weights, agent_actions))
        
        if combined_action.sum() > 0:
            combined_action = combined_action / combined_action.sum()
        
        return combined_action, weights
    
    def train(self, timesteps: int):
        print(f"\n[Orchestrator] Starting meta-learning for {timesteps} timesteps...")
        callback = OrchestratorCallback(self)
        self.model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
        print("[Orchestrator] Meta-learning complete!")
    
    def save(self, path: str):
        self.model.save(f"{path}/orchestrator_model")
        print(f"[Orchestrator] Model saved to {path}")
    
    def load(self, path: str):
        self.model = DQN.load(f"{path}/orchestrator_model", env=self.orch_env)
        print(f"[Orchestrator] Model loaded from {path}")


class OrchestrationEnv(gym.Env):
    """Environment for training the orchestrator"""
    
    def __init__(self, agents: List, base_env, weight_strategies: List):
        super().__init__()
        self.agents = agents
        self.base_env = base_env
        self.weight_strategies = weight_strategies
        self.action_space = gym.spaces.Discrete(len(weight_strategies))
        
        base_obs_dim = base_env.observation_space.shape[0]
        n_agents = len(agents)
        regime_dim = 4  # Simplified regime features
        
        total_obs_dim = base_obs_dim + n_agents + regime_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )
        
        self.weights_history = []
        self.current_weights = None
    
    def reset(self, seed=None, options=None):
        base_obs, info = self.base_env.reset(seed=seed)
        for agent in self.agents:
            agent.performance_history = []
        self.weights_history = []
        return self._get_observation(base_obs), info
    
    def _get_observation(self, base_obs: np.ndarray) -> np.ndarray:
        # Agent performances
        agent_perfs = np.array([agent.get_recent_performance(window=50) for agent in self.agents])
        
        # Simple regime features from base observation
        n_stocks = (len(base_obs) - 1) // 4
        returns_start = n_stocks * 2 + 1
        volatility_start = returns_start + n_stocks
        
        recent_returns = base_obs[returns_start:volatility_start]
        recent_volatility = base_obs[volatility_start:]
        
        regime_features = np.array([
            np.mean(recent_volatility) if len(recent_volatility) > 0 else 0.0,
            np.mean(recent_returns) if len(recent_returns) > 0 else 0.0,
            np.std(recent_returns) if len(recent_returns) > 0 else 0.0,
            (np.max(recent_returns) - np.min(recent_returns)) if len(recent_returns) > 0 else 0.0
        ])
        
        full_obs = np.concatenate([base_obs, agent_perfs, regime_features])
        return full_obs.astype(np.float32)
    
    def step(self, action: int):
        weights = np.array(self.weight_strategies[action])
        self.current_weights = weights
        self.weights_history.append(weights)
        
        base_obs = self.base_env._get_observation()
        agent_actions = [agent.get_action(base_obs, deterministic=True) for agent in self.agents]
        
        combined_action = sum(w * a for w, a in zip(weights, agent_actions))
        if combined_action.sum() > 0:
            combined_action = combined_action / combined_action.sum()
        
        base_obs, reward, terminated, truncated, info = self.base_env.step(combined_action)
        
        for agent in self.agents:
            agent.update_performance(reward)
        
        full_obs = self._get_observation(base_obs)
        return full_obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        self.base_env.render(mode)
        if self.current_weights is not None:
            print(f"  Weights: G={self.current_weights[0]:.2f}, V={self.current_weights[1]:.2f}, R={self.current_weights[2]:.2f}")


class OrchestratorCallback(BaseCallback):
    def __init__(self, orchestrator):
        super().__init__()
        self.orchestrator = orchestrator
    def _on_step(self) -> bool:
        return True
