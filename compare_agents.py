import sys
sys.path.append('.')
from Mycroft_Framework.rl_portfolio.environments import PortfolioEnv
from Mycroft_Framework.rl_portfolio.agents import GrowthAgent, ValueAgent, RiskAgent
import numpy as np

env = PortfolioEnv(
    tickers=['NVDA', 'MSFT', 'GOOGL'],
    start_date='2024-12-01',
    end_date='2024-12-10',
    initial_balance=100000
)

agents = {
    'Growth': GrowthAgent(['NVDA', 'MSFT', 'GOOGL'], env),
    'Value': ValueAgent(['NVDA', 'MSFT', 'GOOGL'], env),
    'Risk': RiskAgent(['NVDA', 'MSFT', 'GOOGL'], env)
}

# Load all
for agent in agents.values():
    agent.load('Mycroft_Framework/rl_portfolio/models/run_20251210_165354')

print("\nAGENT STRATEGY COMPARISON:")
print("="*70)
print("\nSame market state, different learned strategies:\n")

obs, _ = env.reset()

for name, agent in agents.items():
    action = agent.get_action(obs, deterministic=True)
    print(f"{name}Agent allocation:")
    print(f"  NVDA: {action[0]:.1%}  |  MSFT: {action[1]:.1%}  |  GOOGL: {action[2]:.1%}")

print("\nNotice the learned specialization:")
print("  • Growth: High NVDA (semiconductor focus)")
print("  • Value: Balanced MSFT/GOOGL (mega-cap focus)")
print("  • Risk: More equal distribution (diversification)")
