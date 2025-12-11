import sys
sys.path.append('.')
from Mycroft_Framework.rl_portfolio.environments import PortfolioEnv
import numpy as np

env = PortfolioEnv(
    tickers=['NVDA', 'MSFT', 'GOOGL'],
    start_date='2024-07-21',
    end_date='2024-12-10',
    initial_balance=100000
)

print("RANDOM AGENT (BEFORE LEARNING):")
print("="*50)

obs, _ = env.reset()
done = False
step = 0

while not done and step < 10:
    action = env.action_space.sample()  # Random!
    obs, reward, term, trunc, info = env.step(action)
    done = term or trunc
    
    print(f"Day {step+1}: Portfolio = ${info['portfolio_value']:,.2f}, "
          f"Action = {action[:3]} (showing first 3 stocks)")
    step += 1

metrics = env.get_portfolio_metrics()
print(f"\nFinal Return: {metrics['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
