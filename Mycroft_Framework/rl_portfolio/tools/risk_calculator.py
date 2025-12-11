"""
Risk Calculator Tool
Calculates VaR, CVaR, stress tests, and risk metrics
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List

class RiskCalculator:
    """
    Custom Tool: Advanced Risk Calculation
    
    Provides:
    - Value at Risk (VaR) - 95% and 99%
    - Conditional VaR (CVaR/Expected Shortfall)
    - Stress testing scenarios
    - Risk decomposition by asset
    - Monte Carlo simulation
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
        
    def calculate_var(self, returns: np.ndarray, 
                     portfolio_value: float,
                     method: str = 'historical') -> Dict:
        """
        Calculate Value at Risk
        
        Args:
            returns: Array of portfolio returns
            portfolio_value: Current portfolio value
            method: 'historical', 'parametric', or 'monte_carlo'
        """
        
        results = {}
        
        for confidence in self.confidence_levels:
            if method == 'historical':
                var = self._historical_var(returns, confidence)
            elif method == 'parametric':
                var = self._parametric_var(returns, confidence)
            else:
                var = self._monte_carlo_var(returns, confidence)
            
            # Calculate CVaR (expected loss beyond VaR)
            cvar = self._calculate_cvar(returns, var)
            
            # Convert to dollar terms
            var_dollar = var * portfolio_value
            cvar_dollar = cvar * portfolio_value
            
            results[f'VaR_{int(confidence*100)}'] = {
                'percentage': float(var),
                'dollar': float(var_dollar),
                'CVaR_percentage': float(cvar),
                'CVaR_dollar': float(cvar_dollar)
            }
        
        return results
    
    def _historical_var(self, returns: np.ndarray, confidence: float) -> float:
        """Historical VaR - empirical quantile"""
        return -np.percentile(returns, (1 - confidence) * 100)
    
    def _parametric_var(self, returns: np.ndarray, confidence: float) -> float:
        """Parametric VaR - assumes normal distribution"""
        mu = np.mean(returns)
        sigma = np.std(returns)
        z_score = stats.norm.ppf(1 - confidence)
        return -(mu + z_score * sigma)
    
    def _monte_carlo_var(self, returns: np.ndarray, 
                        confidence: float, n_sims: int = 10000) -> float:
        """Monte Carlo VaR - simulate future returns"""
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Simulate returns
        simulated = np.random.normal(mu, sigma, n_sims)
        
        return -np.percentile(simulated, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: np.ndarray, var: float) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        # Returns worse than VaR
        tail_returns = returns[returns < -var]
        
        if len(tail_returns) == 0:
            return var
        
        return -np.mean(tail_returns)
    
    def stress_test(self, returns: pd.DataFrame, 
                   portfolio_weights: np.ndarray,
                   scenarios: Dict[str, Dict] = None) -> Dict:
        """
        Run stress test scenarios
        
        Scenarios:
        - Market crash (-20% across board)
        - Tech sector crash (-30% tech stocks)
        - Volatility spike (2x volatility)
        - Correlation breakdown (decorrelation)
        """
        
        if scenarios is None:
            scenarios = self._default_scenarios()
        
        results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            shock = scenario_params.get('shock', 0)
            vol_multiplier = scenario_params.get('vol_multiplier', 1.0)
            
            # Apply scenario
            shocked_returns = returns * (1 + shock) * vol_multiplier
            
            # Calculate portfolio impact
            portfolio_return = (shocked_returns * portfolio_weights).sum(axis=1).mean()
            
            results[scenario_name] = {
                'portfolio_impact': float(portfolio_return),
                'description': scenario_params.get('description', '')
            }
        
        return results
    
    def _default_scenarios(self) -> Dict:
        """Default stress test scenarios"""
        return {
            'market_crash': {
                'shock': -0.20,
                'vol_multiplier': 1.5,
                'description': 'Market-wide 20% decline with 50% vol increase'
            },
            'volatility_spike': {
                'shock': 0.0,
                'vol_multiplier': 2.5,
                'description': 'Volatility increases 2.5x'
            },
            'severe_drawdown': {
                'shock': -0.35,
                'vol_multiplier': 2.0,
                'description': 'Severe market stress (35% decline, 2x vol)'
            }
        }
    
    def risk_decomposition(self, returns: pd.DataFrame,
                          portfolio_weights: np.ndarray,
                          asset_names: List[str]) -> pd.DataFrame:
        """
        Decompose portfolio risk by asset
        
        Returns marginal and component VaR for each asset
        """
        
        # Portfolio variance
        cov_matrix = returns.cov()
        portfolio_var = portfolio_weights @ cov_matrix @ portfolio_weights
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Marginal VaR (derivative of portfolio vol w.r.t. weight)
        marginal_var = (cov_matrix @ portfolio_weights) / portfolio_vol
        
        # Component VaR (marginal * weight)
        component_var = marginal_var * portfolio_weights
        
        # Percentage contribution
        pct_contribution = component_var / portfolio_vol
        
        decomp = pd.DataFrame({
            'asset': asset_names,
            'weight': portfolio_weights,
            'marginal_var': marginal_var,
            'component_var': component_var,
            'pct_contribution': pct_contribution * 100
        })
        
        return decomp.sort_values('pct_contribution', ascending=False)
    
    def calculate_risk_metrics(self, returns: np.ndarray,
                              portfolio_value: float) -> Dict:
        """Calculate comprehensive risk metrics"""
        
        return {
            'volatility': float(np.std(returns) * np.sqrt(252)),
            'downside_deviation': float(self._downside_deviation(returns)),
            'max_drawdown': float(self._max_drawdown(returns)),
            'sortino_ratio': float(self._sortino_ratio(returns)),
            'calmar_ratio': float(self._calmar_ratio(returns)),
            'var_95': self._historical_var(returns, 0.95) * portfolio_value,
            'var_99': self._historical_var(returns, 0.99) * portfolio_value
        }
    
    def _downside_deviation(self, returns: np.ndarray, 
                           target: float = 0.0) -> float:
        """Calculate downside deviation (semi-deviation)"""
        downside = returns[returns < target]
        return np.std(downside) * np.sqrt(252) if len(downside) > 0 else 0.0
    
    def _max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))
    
    def _sortino_ratio(self, returns: np.ndarray, 
                      risk_free: float = 0.045/252) -> float:
        """Sortino ratio (return / downside deviation)"""
        excess_return = np.mean(returns) - risk_free
        downside_dev = self._downside_deviation(returns)
        
        if downside_dev == 0:
            return 0.0
        
        return (excess_return / downside_dev) * np.sqrt(252)
    
    def _calmar_ratio(self, returns: np.ndarray) -> float:
        """Calmar ratio (annualized return / max drawdown)"""
        annual_return = np.mean(returns) * 252
        max_dd = abs(self._max_drawdown(returns))
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / max_dd
