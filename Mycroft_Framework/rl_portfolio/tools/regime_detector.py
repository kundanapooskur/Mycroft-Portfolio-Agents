"""
Market Regime Detection Tool
Classifies market conditions as Bull/Bear/High Volatility/Stable
Uses statistical clustering and volatility analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.cluster import KMeans

class RegimeDetector:
    """
    Custom Tool: Market Regime Detection
    
    Uses:
    - Volatility clustering (GARCH-like)
    - Trend analysis (moving averages)
    - Volume analysis
    
    Output: Bull/Bear/Volatile/Stable classification
    """
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.regime_history = []
        
    def detect_regime(self, prices: pd.DataFrame, 
                     returns: pd.DataFrame = None) -> Dict[str, any]:
        """
        Detect current market regime
        
        Returns:
            regime: str - One of ['bull', 'bear', 'volatile', 'stable']
            confidence: float - Confidence score [0-1]
            features: dict - Raw features used
        """
        
        if returns is None:
            returns = prices.pct_change()
        
        # Calculate regime features
        features = self._calculate_features(prices, returns)
        
        # Classify regime
        regime, confidence = self._classify_regime(features)
        
        self.regime_history.append({
            'regime': regime,
            'confidence': confidence,
            'timestamp': prices.index[-1] if hasattr(prices.index[-1], 'strftime') else str(prices.index[-1])
        })
        
        return {
            'regime': regime,
            'confidence': confidence,
            'features': features
        }
    
    def _calculate_features(self, prices: pd.DataFrame, 
                           returns: pd.DataFrame) -> Dict:
        """Calculate statistical features for regime detection"""
        
        # Recent window
        recent_prices = prices.iloc[-self.lookback:]
        recent_returns = returns.iloc[-self.lookback:]
        
        # Feature 1: Trend (SMA slope)
        sma_20 = recent_prices.mean(axis=1).rolling(20).mean()
        sma_50 = recent_prices.mean(axis=1).rolling(50).mean()
        trend = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] if len(sma_50) > 0 else 0
        
        # Feature 2: Volatility (realized vol)
        volatility = recent_returns.std().mean() * np.sqrt(252)
        
        # Feature 3: Volatility clustering (autocorr of squared returns)
        squared_returns = recent_returns ** 2
        vol_clustering = squared_returns.mean(axis=1).autocorr(lag=5)
        if np.isnan(vol_clustering):
            vol_clustering = 0
        
        # Feature 4: Return skewness
        skewness = recent_returns.mean(axis=1).skew()
        if np.isnan(skewness):
            skewness = 0
        
        # Feature 5: Max drawdown
        cumulative = (1 + recent_returns.mean(axis=1)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'trend': float(trend),
            'volatility': float(volatility),
            'vol_clustering': float(vol_clustering),
            'skewness': float(skewness),
            'max_drawdown': float(max_dd)
        }
    
    def _classify_regime(self, features: Dict) -> Tuple[str, float]:
        """
        Classify regime based on features
        
        Rules:
        - Bull: positive trend + low volatility
        - Bear: negative trend + high volatility
        - Volatile: high volatility + high vol_clustering
        - Stable: low volatility + low vol_clustering
        """
        
        trend = features['trend']
        vol = features['volatility']
        vol_clust = features['vol_clustering']
        
        # Thresholds (calibrated for typical market conditions)
        high_vol_threshold = 0.25  # 25% annualized
        high_trend_threshold = 0.02  # 2% relative
        high_clust_threshold = 0.3
        
        # Classification logic
        if vol > high_vol_threshold:
            if vol_clust > high_clust_threshold:
                regime = 'volatile'
                confidence = min(vol / high_vol_threshold, 1.0)
            elif trend < -high_trend_threshold:
                regime = 'bear'
                confidence = min(-trend / high_trend_threshold, 1.0)
            else:
                regime = 'volatile'
                confidence = 0.6
        else:
            if trend > high_trend_threshold:
                regime = 'bull'
                confidence = min(trend / high_trend_threshold, 1.0)
            else:
                regime = 'stable'
                confidence = 0.7
        
        return regime, min(confidence, 1.0)
    
    def get_regime_distribution(self) -> Dict[str, float]:
        """Get distribution of regimes over history"""
        if not self.regime_history:
            return {}
        
        regimes = [r['regime'] for r in self.regime_history]
        unique, counts = np.unique(regimes, return_counts=True)
        
        return {regime: count / len(regimes) 
                for regime, count in zip(unique, counts)}
    
    def get_regime_transitions(self) -> pd.DataFrame:
        """Analyze regime transition matrix"""
        if len(self.regime_history) < 2:
            return pd.DataFrame()
        
        transitions = []
        for i in range(len(self.regime_history) - 1):
            transitions.append({
                'from': self.regime_history[i]['regime'],
                'to': self.regime_history[i+1]['regime']
            })
        
        return pd.DataFrame(transitions)
