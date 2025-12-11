"""
Mycroft Orchestrator V2 - IMPROVED
Integrates custom tools for sophisticated decision-making
"""

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Tuple
import gymnasium as gym
import sys
sys.path.append('.')

from Mycroft_Framework.rl_portfolio.tools import RegimeDetector, RiskCalculator

class MycroftOrchestratorV2:
    """
    IMPROVED Orchestrator with Tool Integration
    
    Key Improvements:
    - Uses RegimeDetector for market classification
    - Uses RiskCalculator for portfolio risk metrics
    - Expanded state space with tool outputs
    - Regime-aware weight strategies
    """
    
    # EXPANDED weight strategies based on regime
    WEIGHT_STRATEGIES = [
        # Bull market strategies
        [0.7, 0.2, 0.1],  # 0: Aggressive growth (bull)
        [0.6, 0.3, 0.1],  # 1: Growth-value (bull)
        
        # Bear market strategies  
        [0.1, 0.3, 0.6],  # 2: Defensive (bear)
        [0.2, 0.2, 0.6],  # 3: High risk mgmt (bear)
        
        # Volatile market strategies
        [0.2, 0.5, 0.3],  # 4: Value-focused (volatile)
        [0.3, 0.3, 0.4],  # 5: Balanced defensive (volatile)
        
        # Stable market strategies
        [0.5, 0.4, 0.1],  # 6: Growth-value balanced (stable)
        [0.4, 0.4, 0.2],  # 7: Fully balanced (stable)
        
        # Adaptive strategies
        [0.33, 0.33, 0.34],  # 8: Equal (uncertain)
        [0.25, 0.5, 0.25],   # 9: Value-heavy (any)
    ]
    
    def __init__(self, agents: List, env):
        self.agents = agents
        self.env = env
        self.n_agents = len(agents)
        
        # Initialize custom tools
        self.regime_detector = RegimeDetector(lookback=60)
        self.risk_calculator = RiskCalculator()
        
        # Create enhanced orchestration environment
        self.orch_env = OrchestrationEnvV2(
            agents, env, self.WEIGHT_STRATEGIES,
            self.regime_detector, self.risk_calculator
        )
        
        # DQN with improved hyperparameters
        self.model = DQN(
            "MlpPolicy",
            self.orch_env,
            learning_rate=5e-5,  # Lower for stability
            buffer_size=100000,  # Larger memory
            learning_starts=2000,  # More experience before training
            batch_size=256,  # Larger batches
            tau=0.01,  # Slower target updates
            gamma=0.995,  # Higher discount (long-term focus)
            train_freq=4,
            gradient_steps=2,  # More gradient steps per update
            exploration_fraction=0.4,  # Longer exploration
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,  # Lower final epsilon
            target_update_interval=500,  # Less frequent target updates
            verbose=1,
            tensorboard_log="./Mycroft_Framework/rl_portfolio/logs/orchestrator_v2/"
        )
        
        print(f"[OrchestratorV2] Initialized with {self.n_agents} agents")
        print(f"[OrchestratorV2] Action space: {len(self.WEIGHT_STRATEGIES)} regime-aware strategies")
        print(f"[OrchestratorV2] Enhanced with: Regime Detection + Risk Calculation")
    
    def get_action(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Returns: (combined_action, weights, metadata)"""
        strategy_idx, _ = self.model.predict(observation, deterministic=deterministic)
        weights = np.array(self.WEIGHT_STRATEGIES[strategy_idx])
        
        base_obs = observation[:self.env.observation_space.shape[0]] if len(observation) > self.env.observation_space.shape[0] else observation
        
        agent_actions = [agent.get_action(base_obs, deterministic=True) for agent in self.agents]
        combined_action = sum(w * a for w, a in zip(weights, agent_actions))
        
        if combined_action.sum() > 0:
            combined_action = combined_action / combined_action.sum()
        
        # Get regime info for metadata
        regime_info = self.orch_env.last_regime_info
        
        metadata = {
            'strategy_idx': int(strategy_idx),
            'weights': weights.tolist(),
            'regime': regime_info.get('regime', 'unknown') if regime_info else 'unknown'
        }
        
        return combined_action, weights, metadata
    
    def train(self, timesteps: int):
        print(f"\n[OrchestratorV2] Starting enhanced meta-learning for {timesteps} timesteps...")
        print("[OrchestratorV2] Learning regime-aware agent coordination...")
        
        callback = OrchestratorCallbackV2(self)
        self.model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
        
        print("[OrchestratorV2] Enhanced meta-learning complete!")
    
    def save(self, path: str):
        self.model.save(f"{path}/orchestrator_v2_model")
        print(f"[OrchestratorV2] Model saved to {path}")
    
    def load(self, path: str):
        self.model = DQN.load(f"{path}/orchestrator_v2_model", env=self.orch_env)
        print(f"[OrchestratorV2] Model loaded from {path}")


class OrchestrationEnvV2(gym.Env):
    """ENHANCED Orchestration Environment with Tool Integration"""
    
    def __init__(self, agents: List, base_env, weight_strategies: List,
                 regime_detector: RegimeDetector, risk_calculator: RiskCalculator):
        super().__init__()
        
        self.agents = agents
        self.base_env = base_env
        self.weight_strategies = weight_strategies
        self.regime_detector = regime_detector
        self.risk_calculator = risk_calculator
        
        self.action_space = gym.spaces.Discrete(len(weight_strategies))
        
        # EXPANDED observation space with tool outputs
        base_obs_dim = base_env.observation_space.shape[0]
        n_agents = len(agents)
        regime_dim = 5  # regime features from detector
        risk_dim = 3    # VaR, volatility, drawdown from calculator
        
        total_obs_dim = base_obs_dim + n_agents + regime_dim + risk_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )
        
        self.weights_history = []
        self.regime_history = []
        self.last_regime_info = None
        
        print(f"[OrchEnvV2] Obs space expanded to {total_obs_dim} (base:{base_obs_dim} + agents:{n_agents} + regime:{regime_dim} + risk:{risk_dim})")
    
    def reset(self, seed=None, options=None):
        base_obs, info = self.base_env.reset(seed=seed)
        
        for agent in self.agents:
            agent.performance_history = []
        
        self.weights_history = []
        self.regime_history = []
        self.last_regime_info = None
        
        return self._get_observation(base_obs), info
    
    def _get_observation(self, base_obs: np.ndarray) -> np.ndarray:
        """Enhanced observation with tool outputs"""
        
        # 1. Agent performances
        agent_perfs = np.array([agent.get_recent_performance(50) for agent in self.agents])
        
        # 2. Regime features from RegimeDetector
        if self.base_env.current_step >= self.base_env.lookback_window:
            recent_prices = self.base_env.data.iloc[
                max(0, self.base_env.current_step-60):self.base_env.current_step
            ]
            recent_returns = self.base_env.returns.iloc[
                max(0, self.base_env.current_step-60):self.base_env.current_step
            ]
            
            regime_info = self.regime_detector.detect_regime(recent_prices, recent_returns)
            self.last_regime_info = regime_info
            
            regime_features = np.array([
                regime_info['features']['trend'],
                regime_info['features']['volatility'],
                regime_info['features']['vol_clustering'],
                regime_info['features']['skewness'],
                regime_info['features']['max_drawdown']
            ])
        else:
            regime_features = np.zeros(5)
            self.last_regime_info = {'regime': 'unknown', 'confidence': 0.0}
        
        # 3. Risk metrics from RiskCalculator
        if len(self.base_env.returns_history) > 10:
            risk_metrics = self.risk_calculator.calculate_risk_metrics(
                np.array(self.base_env.returns_history),
                self.base_env.portfolio_value
            )
            risk_features = np.array([
                risk_metrics['volatility'],
                risk_metrics['max_drawdown'],
                risk_metrics['sortino_ratio']
            ])
        else:
            risk_features = np.zeros(3)
        
        # Combine all features
        full_obs = np.concatenate([
            base_obs,
            agent_perfs,
            regime_features,
            risk_features
        ]).astype(np.float32)
        
        return full_obs
    
    def step(self, action: int):
        weights = np.array(self.weight_strategies[action])
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
        
        # ENHANCED REWARD: bonus for good coordination
        if len(self.base_env.returns_history) > 1:
            portfolio_return = self.base_env.returns_history[-1]
            # Check if coordination beat best individual agent's recent performance
            best_agent_perf = max([a.get_recent_performance(5) for a in self.agents])
            if reward > best_agent_perf:
                reward += 2.0  # Coordination bonus
        
        return full_obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        self.base_env.render(mode)


class OrchestratorCallbackV2(BaseCallback):
    def __init__(self, orchestrator):
        super().__init__()
        self.orchestrator = orchestrator
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
        return True
