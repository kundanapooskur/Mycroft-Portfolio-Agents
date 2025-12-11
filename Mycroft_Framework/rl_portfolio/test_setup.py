#!/usr/bin/env python3
"""
Quick test to verify Mycroft RL setup works
Tests: environment, data download, agent initialization
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import yaml
import numpy as np

print("="*70)
print("MYCROFT RL SETUP TEST")
print("="*70)

# Test 1: Load config
print("\n[Test 1] Loading configuration...")
try:
    with open('Mycroft_Framework/rl_portfolio/configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded successfully")
    print(f"  Tickers: {config['tickers']}")
except Exception as e:
    print(f"✗ Config load failed: {e}")
    sys.exit(1)

# Test 2: Create environment
print("\n[Test 2] Creating portfolio environment...")
try:
    from Mycroft_Framework.rl_portfolio.environments import PortfolioEnv
    
    env = PortfolioEnv(
        tickers=config['tickers'][:3],  # Test with 3 stocks
        start_date="2024-01-01",
        end_date="2024-06-30",
        initial_balance=10000,
        lookback_window=20
    )
    print(f"✓ Environment created successfully")
    print(f"  State dim: {env.observation_space.shape}")
    print(f"  Action dim: {env.action_space.shape}")
    print(f"  Data points: {len(env.dates)}")
except Exception as e:
    print(f"✗ Environment creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Environment reset and step
print("\n[Test 3] Testing environment mechanics...")
try:
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Initial observation shape: {obs.shape}")
    
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Environment step successful")
    print(f"  Reward: {reward:.4f}")
    print(f"  Portfolio value: ${info['portfolio_value']:,.2f}")
except Exception as e:
    print(f"✗ Environment mechanics failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Initialize agents
print("\n[Test 4] Initializing agents...")
try:
    from Mycroft_Framework.rl_portfolio.agents import (
        GrowthAgent, ValueAgent, RiskAgent
    )
    
    growth = GrowthAgent(config['tickers'][:3], env)
    value = ValueAgent(config['tickers'][:3], env)
    risk = RiskAgent(config['tickers'][:3], env)
    
    print(f"✓ Growth Agent initialized")
    print(f"✓ Value Agent initialized")
    print(f"✓ Risk Agent initialized")
    
    # Test action
    action = growth.get_action(obs)
    print(f"  Sample action sum: {action.sum():.4f} (should be ~1.0)")
    
except Exception as e:
    print(f"✗ Agent initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Initialize orchestrator
print("\n[Test 5] Initializing orchestrator...")
try:
    from Mycroft_Framework.rl_portfolio.agents import MycroftOrchestrator
    
    orchestrator = MycroftOrchestrator([growth, value, risk], env)
    print(f"✓ Orchestrator initialized")
    print(f"  Weight strategies: {len(orchestrator.WEIGHT_STRATEGIES)}")
    
    # Reset orchestrator environment to get proper observation
    orch_obs, _ = orchestrator.orch_env.reset()
    print(f"  Orchestrator obs shape: {orch_obs.shape}")
    
    # Test coordinated action with orchestrator observation
    action, weights = orchestrator.get_action(orch_obs)
    print(f"✓ Coordinated action generated")
    print(f"  Action sum: {action.sum():.4f}")
    print(f"  Agent weights: {weights}")
    
except Exception as e:
    print(f"✗ Orchestrator test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Quick training test (10 steps)
print("\n[Test 6] Quick training test (10 steps)...")
try:
    print("  Training Growth Agent for 10 steps...")
    growth.model.learn(total_timesteps=10, progress_bar=False)
    print("✓ Agent training works")
except Exception as e:
    print(f"✗ Training test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nYour Mycroft RL system is ready!")
print("\nTo train the full system, run:")
print("  python3 Mycroft_Framework/rl_portfolio/train.py")
print("\nFor a quick demo (faster), run:")
print("  python3 Mycroft_Framework/rl_portfolio/train.py --agent-steps 1000 --orch-steps 1000")
print("="*70 + "\n")
