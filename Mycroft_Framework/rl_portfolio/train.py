#!/usr/bin/env python3
"""
Mycroft RL Portfolio Training - Main Entry Point

Usage:
    python3 Mycroft_Framework/rl_portfolio/train.py
    
Or with custom config:
    python3 Mycroft_Framework/rl_portfolio/train.py --config path/to/config.yaml
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from Mycroft_Framework.rl_portfolio.training import MycroftTrainer

def main():
    parser = argparse.ArgumentParser(
        description='Train Mycroft RL Portfolio Agents'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='Mycroft_Framework/rl_portfolio/configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--agent-steps',
        type=int,
        default=15000,
        help='Training timesteps per agent (default: 15000)'
    )
    parser.add_argument(
        '--orch-steps',
        type=int,
        default=20000,
        help='Training timesteps for orchestrator (default: 20000)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MYCROFT: USING AI TO INVEST IN AI")
    print("Multi-Agent Reinforcement Learning Portfolio System")
    print("="*70)
    print(f"\nConfiguration: {args.config}")
    print(f"Agent Training Steps: {args.agent_steps}")
    print(f"Orchestrator Training Steps: {args.orch_steps}")
    
    # Initialize trainer
    trainer = MycroftTrainer(config_path=args.config)
    
    # Run training
    trainer.train_individual_agents(timesteps_per_agent=args.agent_steps)
    trainer.train_orchestrator(timesteps=args.orch_steps)
    trainer._generate_comparison_report()
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print(f"✓ Models saved to: {trainer.save_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
