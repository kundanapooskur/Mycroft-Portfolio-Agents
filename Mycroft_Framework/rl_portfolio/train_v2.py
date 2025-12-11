#!/usr/bin/env python3
"""
Mycroft V2 Training - With Improved Orchestrator
"""

import argparse
import sys
import os
sys.path.append('.')

from Mycroft_Framework.rl_portfolio.training.train_agents import MycroftTrainer
from Mycroft_Framework.rl_portfolio.agents import MycroftOrchestratorV2

# Monkey-patch to use V2 orchestrator
original_init = MycroftTrainer.__init__

def enhanced_init(self, config_path="Mycroft_Framework/rl_portfolio/configs/config.yaml"):
    original_init(self, config_path)
    
    # Replace orchestrator with V2
    print("\n[UPGRADE] Using Enhanced Orchestrator V2 with Custom Tools")
    self.orchestrator = MycroftOrchestratorV2(self.agents, self.train_env)

MycroftTrainer.__init__ = enhanced_init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-steps', type=int, default=30000)
    parser.add_argument('--orch-steps', type=int, default=40000)
    args = parser.parse_args()
    
    print("="*70)
    print("MYCROFT V2: ENHANCED ORCHESTRATOR WITH CUSTOM TOOLS")
    print("="*70)
    print(f"Agent Steps: {args.agent_steps}")
    print(f"Orchestrator Steps: {args.orch_steps} (EXTENDED)")
    
    trainer = MycroftTrainer()
    trainer.train_individual_agents(timesteps_per_agent=args.agent_steps)
    trainer.train_orchestrator(timesteps=args.orch_steps)
    trainer._generate_comparison_report()
    
    print("\n✓ V2 TRAINING COMPLETE!")
    print(f"✓ Models: {trainer.save_dir}")

if __name__ == "__main__":
    main()
