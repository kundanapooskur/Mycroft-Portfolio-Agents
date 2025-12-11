import sys
sys.path.append('.')
import numpy as np
import pandas as pd
from Mycroft_Framework.rl_portfolio.tools import RegimeDetector, RiskCalculator

print("Testing Custom Tools...")
print("="*50)

# Test data
dates = pd.date_range('2024-01-01', periods=100)
prices = pd.DataFrame({
    'NVDA': np.random.randn(100).cumsum() + 100,
    'MSFT': np.random.randn(100).cumsum() + 300
}, index=dates)

# Test Regime Detector
print("\n1. Regime Detector:")
detector = RegimeDetector()
regime_info = detector.detect_regime(prices)
print(f"   Regime: {regime_info['regime']}")
print(f"   Confidence: {regime_info['confidence']:.2f}")
print(f"   Features: {regime_info['features']}")

# Test Risk Calculator
print("\n2. Risk Calculator:")
returns = prices.pct_change().dropna()
calc = RiskCalculator()
var_results = calc.calculate_var(returns.mean(axis=1).values, 100000)
print(f"   VaR 95%: ${var_results['VaR_95']['dollar']:,.2f}")
print(f"   CVaR 95%: ${var_results['VaR_95']['CVaR_dollar']:,.2f}")

print("\n✓ All tools working!")
