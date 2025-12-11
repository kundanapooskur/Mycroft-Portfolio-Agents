import sys
sys.path.append('.')
import numpy as np
from Mycroft_Framework.rl_portfolio.environments import PortfolioEnv
from Mycroft_Framework.rl_portfolio.agents import ValueAgent

print("\n" + "="*70)
print("LIVE DEMONSTRATION: TRAINED AGENT IN ACTION")
print("="*70)

# Create environment
env = PortfolioEnv(
    tickers=['NVDA', 'AVGO', 'AMD', 'MU', 'MRVL', 'AMZN', 'MSFT', 'GOOGL', 'NOW', 'CRM'],
    start_date='2024-12-01',
    end_date='2024-12-10',
    initial_balance=100000
)

# Load trained agent
agent = ValueAgent(env.tickers, env)
agent.load('Mycroft_Framework/rl_portfolio/models/run_20251210_165354')

print("\nRunning trained ValueAgent on last 5 trading days...\n")

obs, _ = env.reset()
for day in range(5):
    action = agent.get_action(obs, deterministic=True)
    obs, reward, term, trunc, info = env.step(action)
    
    print(f"Day {day+1} ({info['date']}):")
    print(f"  Portfolio Value: ${info['portfolio_value']:,.2f}")
    print(f"  Top 3 Holdings:")
    top_3_idx = np.argsort(action)[-3:][::-1]
    for idx in top_3_idx:
        print(f"    {env.tickers[idx]}: {action[idx]:.1%}")
    print()

metrics = env.get_portfolio_metrics()
print(f"Final Performance:")
print(f"  Return: {metrics['total_return']*100:.2f}%")
print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
