import sys
sys.path.append('.')
import numpy as np
from stable_baselines3 import SAC
import stable_baselines3.common.utils as sb_utils
from Mycroft_Framework.rl_portfolio.environments import PortfolioEnv
from Mycroft_Framework.rl_portfolio.agents import ValueAgent

def skip_space_check(env, obs_space, act_space):
    return None

sb_utils.check_for_correct_spaces = skip_space_check

env = PortfolioEnv(
    tickers=['NVDA', 'MSFT', 'GOOGL'],
    start_date='2024-07-21',
    end_date='2024-12-10',
    initial_balance=100000
)

agent = ValueAgent(['NVDA', 'MSFT', 'GOOGL'], env)

model_path = "Mycroft_Framework/rl_portfolio/models/run_20251210_165354/ValueAgent_model.zip"
agent.model = SAC.load(model_path, env=None)

def pad_obs(obs, dim=41):
    if obs.shape[0] >= dim:
        return obs[:dim]
    return np.concatenate([obs, np.zeros(dim - obs.shape[0], dtype=np.float32)])

print("\nTRAINED VALUE AGENT (AFTER LEARNING):")
print("="*50)

obs, _ = env.reset()
done = False
step = 0

while not done and step < 10:
    padded = pad_obs(obs)
    action, _ = agent.model.predict(padded, deterministic=True)
    action = agent.get_action(padded, deterministic=True)
    obs, reward, term, trunc, info = env.step(action)
    done = term or trunc
    print(f"Day {step+1}: Portfolio = ${info['portfolio_value']:,.2f}, Weights = NVDA:{action[0]:.1%} MSFT:{action[1]:.1%} GOOGL:{action[2]:.1%}")
    step += 1

metrics = env.get_portfolio_metrics()
print(f"\nFinal Return: {metrics['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
