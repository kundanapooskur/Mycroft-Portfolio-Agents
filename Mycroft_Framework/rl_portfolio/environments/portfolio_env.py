"""
Mycroft Portfolio Environment - IMPROVED VERSION
Enhanced reward function for better RL agent performance
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple, Optional, List
from datetime import datetime

class PortfolioEnv(gym.Env):
    """
    IMPROVED Portfolio Management Environment
    
    Key Improvements:
    - Multi-component reward function (return + risk + consistency)
    - Sharpe ratio optimization built-in
    - Drawdown penalties
    - Outperformance bonuses
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 tickers: List[str],
                 start_date: str,
                 end_date: str,
                 initial_balance: float = 100000,
                 lookback_window: int = 20,
                 transaction_cost: float = 0.0005,
                 max_position_size: float = 0.20,
                 risk_free_rate: float = 0.045):
        
        super().__init__()
        
        self.tickers = tickers
        self.n_stocks = len(tickers)
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate / 252
        
        print(f"[PortfolioEnv] Initializing with {self.n_stocks} tickers: {tickers}")
        print(f"[PortfolioEnv] Downloading data from {start_date} to {end_date}...")
        
        self.data = self._download_data(tickers, start_date, end_date)
        self.dates = self.data.index
        self.n_steps = len(self.dates)
        
        print(f"[PortfolioEnv] Downloaded {self.n_steps} days of data")
        
        self._calculate_features()
        
        state_dim = self.n_stocks * 4 + 1
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(self.n_stocks,), dtype=np.float32
        )
        
        print(f"[PortfolioEnv] State dim: {state_dim}, Action dim: {self.n_stocks}")
        
    def _download_data(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        try:
            raw_data = yf.download(tickers, start=start, end=end, progress=False)
            
            if len(tickers) == 1:
                data = raw_data[['Adj Close']].copy()
                data.columns = tickers
            else:
                if 'Adj Close' in raw_data.columns:
                    data = raw_data['Adj Close'].copy()
                else:
                    data = raw_data['Close'].copy()
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
            
            data = data.ffill().bfill()
            
            for ticker in tickers:
                if ticker not in data.columns:
                    raise ValueError(f"Ticker {ticker} not found")
            
            return data[tickers]
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise
    
    def _calculate_features(self):
        self.returns = self.data.pct_change().fillna(0)
        self.volatility = self.returns.rolling(window=20).std().fillna(0)
        self.normalized_prices = (self.data - self.data.mean()) / (self.data.std() + 1e-8)
        
        # Equal-weight benchmark for comparison
        self.equal_weight_returns = self.returns.mean(axis=1)
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.cash = self.initial_balance
        self.holdings = np.zeros(self.n_stocks)
        self.portfolio_value = self.initial_balance
        
        self.portfolio_values = [self.initial_balance]
        self.actions_taken = []
        self.returns_history = []
        self.peak_value = self.initial_balance
        
        return self._get_observation(), self._get_info()
    
    def _get_observation(self) -> np.ndarray:
        current_prices = self.normalized_prices.iloc[self.current_step].values
        
        current_stock_prices = self.data.iloc[self.current_step].values
        holdings_value = self.holdings * current_stock_prices
        total_value = self.cash + holdings_value.sum()
        
        if total_value > 0:
            holdings_pct = holdings_value / total_value
            cash_pct = self.cash / total_value
        else:
            holdings_pct = np.zeros(self.n_stocks)
            cash_pct = 1.0
        
        recent_returns = self.returns.iloc[
            max(0, self.current_step - self.lookback_window):self.current_step
        ].mean().values
        
        recent_volatility = self.volatility.iloc[self.current_step].values
        
        obs = np.concatenate([
            current_prices,
            holdings_pct,
            [cash_pct],
            recent_returns,
            recent_volatility
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Normalize action
        action = np.clip(action, 0, 1)
        action_sum = action.sum()
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones(self.n_stocks) / self.n_stocks
        
        action = np.clip(action, 0, self.max_position_size)
        action = action / action.sum()
        
        # Execute trade
        current_prices = self.data.iloc[self.current_step].values
        current_value = self.cash + (self.holdings * current_prices).sum()
        
        target_values = action * current_value
        target_holdings = target_values / (current_prices + 1e-8)
        
        trades = target_holdings - self.holdings
        trade_costs = np.abs(trades * current_prices).sum() * self.transaction_cost
        
        self.holdings = target_holdings
        self.cash = current_value - target_values.sum() - trade_costs
        
        self.current_step += 1
        
        # Calculate new value
        if self.current_step < self.n_steps:
            new_prices = self.data.iloc[self.current_step].values
            new_value = self.cash + (self.holdings * new_prices).sum()
        else:
            new_value = current_value
        
        # ====== IMPROVED REWARD FUNCTION ======
        daily_return = (new_value - current_value) / current_value if current_value > 0 else 0
        
        # Component 1: Base return (amplified)
        reward = (daily_return - self.risk_free_rate) * 20
        
        # Component 2: Sharpe ratio bonus
        if len(self.returns_history) > 10:
            recent_returns = np.array(self.returns_history[-10:])
            if np.std(recent_returns) > 0:
                sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8)
                reward += sharpe * 2
        
        # Component 3: Outperformance bonus
        equal_weight_return = self.equal_weight_returns.iloc[self.current_step]
        if daily_return > equal_weight_return:
            reward += (daily_return - equal_weight_return) * 30
        
        # Component 4: Drawdown penalty
        self.peak_value = max(self.peak_value, current_value)
        drawdown = (new_value - self.peak_value) / self.peak_value
        if drawdown < -0.03:  # More than 3% from peak
            reward -= abs(drawdown) * 50
        
        # Component 5: Volatility penalty
        if len(self.returns_history) > 5:
            recent_vol = np.std(self.returns_history[-5:])
            reward -= recent_vol * 10
        
        # Component 6: Diversification bonus
        # Penalize concentrated positions
        concentration = np.sum(action ** 2)  # Herfindahl index
        if concentration < 0.15:  # Well diversified
            reward += 1.0
        elif concentration > 0.25:  # Too concentrated
            reward -= 2.0
        
        # ======================================
        
        self.portfolio_values.append(new_value)
        self.actions_taken.append(action.copy())
        self.returns_history.append(daily_return)
        self.portfolio_value = new_value
        
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        obs = self._get_observation() if not terminated else np.zeros_like(self._get_observation())
        
        return obs, reward, terminated, truncated, self._get_info()
    
    def _get_info(self) -> dict:
        return {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'step': self.current_step,
            'date': str(self.dates[min(self.current_step, len(self.dates)-1)])
        }
    
    def render(self, mode='human'):
        if len(self.portfolio_values) > 1:
            total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance * 100
            print(f"Step: {self.current_step}/{self.n_steps} | "
                  f"Portfolio Value: ${self.portfolio_value:,.2f} | "
                  f"Return: {total_return:.2f}%")
    
    def get_portfolio_metrics(self) -> dict:
        if len(self.portfolio_values) < 2:
            return {}
        
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        total_return = (values[-1] - values[0]) / values[0]
        
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.sqrt(252) * (np.mean(returns) - self.risk_free_rate) / np.std(returns)
        else:
            sharpe = 0
        
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        volatility = np.std(returns) * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'final_value': values[-1]
        }
