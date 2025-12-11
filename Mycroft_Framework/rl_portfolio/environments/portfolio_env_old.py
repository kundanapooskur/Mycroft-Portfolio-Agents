"""
Mycroft Portfolio Environment
Gymnasium environment for training RL agents on AI stock portfolio management
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple, Optional, List
from datetime import datetime

class PortfolioEnv(gym.Env):
    """
    Portfolio Management Environment for AI Stocks
    
    State: [normalized_prices, holdings, cash_ratio, returns, volatility]
    Action: Portfolio allocation weights [0-1] for each stock
    Reward: Risk-adjusted returns (Sharpe ratio based)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 tickers: List[str],
                 start_date: str,
                 end_date: str,
                 initial_balance: float = 100000,
                 lookback_window: int = 20,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 0.25,
                 risk_free_rate: float = 0.045):
        
        super().__init__()
        
        self.tickers = tickers
        self.n_stocks = len(tickers)
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        
        print(f"[PortfolioEnv] Initializing with {self.n_stocks} tickers: {tickers}")
        print(f"[PortfolioEnv] Downloading data from {start_date} to {end_date}...")
        
        # Download historical data
        self.data = self._download_data(tickers, start_date, end_date)
        self.dates = self.data.index
        self.n_steps = len(self.dates)
        
        print(f"[PortfolioEnv] Downloaded {self.n_steps} days of data")
        
        # Calculate technical indicators
        self._calculate_features()
        
        # State: [prices(n), holdings(n), cash(1), returns(n), volatility(n)]
        state_dim = self.n_stocks * 4 + 1
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Action: target portfolio weights [0-1] for each stock
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(self.n_stocks,), dtype=np.float32
        )
        
        print(f"[PortfolioEnv] State dim: {state_dim}, Action dim: {self.n_stocks}")
        
    def _download_data(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """Download historical price data"""
        try:
            # Download data
            raw_data = yf.download(tickers, start=start, end=end, progress=False)
            
            # Handle different data structures based on number of tickers
            if len(tickers) == 1:
                # Single ticker returns simple DataFrame
                data = raw_data[['Adj Close']].copy()
                data.columns = tickers
            else:
                # Multiple tickers returns MultiIndex DataFrame
                if 'Adj Close' in raw_data.columns:
                    data = raw_data['Adj Close'].copy()
                else:
                    # Fallback to Close if Adj Close not available
                    data = raw_data['Close'].copy()
                
                # Ensure column names are correct
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
            
            # Forward fill then backward fill any NaNs
            data = data.ffill().bfill()
            
            # Ensure we have all tickers
            for ticker in tickers:
                if ticker not in data.columns:
                    raise ValueError(f"Ticker {ticker} not found in downloaded data")
            
            return data[tickers]  # Return in correct order
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise
    
    def _calculate_features(self):
        """Calculate technical indicators"""
        # Returns
        self.returns = self.data.pct_change().fillna(0)
        
        # Rolling volatility (20-day)
        self.volatility = self.returns.rolling(window=20).std().fillna(0)
        
        # Normalize prices (for state representation)
        self.normalized_prices = (self.data - self.data.mean()) / (self.data.std() + 1e-8)
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Start after lookback window to have enough history
        self.current_step = self.lookback_window
        
        # Reset portfolio state
        self.balance = self.initial_balance
        self.cash = self.initial_balance
        self.holdings = np.zeros(self.n_stocks)  # Number of shares
        self.portfolio_value = self.initial_balance
        
        # Track history for metrics
        self.portfolio_values = [self.initial_balance]
        self.actions_taken = []
        self.returns_history = []
        
        return self._get_observation(), self._get_info()
    
    def _get_observation(self) -> np.ndarray:
        """Construct state vector"""
        # Current prices (normalized)
        current_prices = self.normalized_prices.iloc[self.current_step].values
        
        # Holdings as percentage of portfolio
        current_stock_prices = self.data.iloc[self.current_step].values
        holdings_value = self.holdings * current_stock_prices
        total_value = self.cash + holdings_value.sum()
        
        if total_value > 0:
            holdings_pct = holdings_value / total_value
            cash_pct = self.cash / total_value
        else:
            holdings_pct = np.zeros(self.n_stocks)
            cash_pct = 1.0
        
        # Recent returns (20-day average)
        recent_returns = self.returns.iloc[
            max(0, self.current_step - self.lookback_window):self.current_step
        ].mean().values
        
        # Recent volatility
        recent_volatility = self.volatility.iloc[self.current_step].values
        
        # Concatenate all features
        obs = np.concatenate([
            current_prices,
            holdings_pct,
            [cash_pct],
            recent_returns,
            recent_volatility
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment"""
        
        # Normalize action to sum to 1 (portfolio weights)
        action = np.clip(action, 0, 1)
        action_sum = action.sum()
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones(self.n_stocks) / self.n_stocks
        
        # Clip individual positions to max size
        action = np.clip(action, 0, self.max_position_size)
        action = action / action.sum()  # Renormalize
        
        # Current state
        current_prices = self.data.iloc[self.current_step].values
        current_value = self.cash + (self.holdings * current_prices).sum()
        
        # Calculate target holdings
        target_values = action * current_value
        target_holdings = target_values / (current_prices + 1e-8)
        
        # Calculate trades
        trades = target_holdings - self.holdings
        trade_costs = np.abs(trades * current_prices).sum() * self.transaction_cost
        
        # Execute trades
        self.holdings = target_holdings
        self.cash = current_value - target_values.sum() - trade_costs
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new portfolio value
        if self.current_step < self.n_steps:
            new_prices = self.data.iloc[self.current_step].values
            new_value = self.cash + (self.holdings * new_prices).sum()
        else:
            new_value = current_value
        
        # Calculate reward (daily return adjusted for risk)
        daily_return = (new_value - current_value) / current_value if current_value > 0 else 0
        
        # Risk-adjusted reward (Sharpe-like)
        reward = daily_return - self.risk_free_rate
        
        # Penalty for extreme actions (encourage stability)
        action_penalty = 0.001 * np.std(action)
        reward -= action_penalty
        
        # Update tracking
        self.portfolio_values.append(new_value)
        self.actions_taken.append(action.copy())
        self.returns_history.append(daily_return)
        self.portfolio_value = new_value
        
        # Check if episode is done
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        # Get next observation
        obs = self._get_observation() if not terminated else np.zeros_like(self._get_observation())
        
        return obs, reward, terminated, truncated, self._get_info()
    
    def _get_info(self) -> dict:
        """Return additional information"""
        return {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'step': self.current_step,
            'date': str(self.dates[min(self.current_step, len(self.dates)-1)])
        }
    
    def render(self, mode='human'):
        """Render environment state"""
        if len(self.portfolio_values) > 1:
            total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance * 100
            print(f"Step: {self.current_step}/{self.n_steps} | "
                  f"Portfolio Value: ${self.portfolio_value:,.2f} | "
                  f"Return: {total_return:.2f}%")
    
    def get_portfolio_metrics(self) -> dict:
        """Calculate portfolio performance metrics"""
        if len(self.portfolio_values) < 2:
            return {}
        
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        total_return = (values[-1] - values[0]) / values[0]
        
        # Sharpe Ratio (annualized)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.sqrt(252) * (np.mean(returns) - self.risk_free_rate) / np.std(returns)
        else:
            sharpe = 0
        
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'final_value': values[-1]
        }
