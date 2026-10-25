import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rl_inference_scheduler.trainer import train_dqn as train_dqn_v1
from rl_inference_scheduler.llm_trainer import train_dqn_v2
from rl_inference_scheduler.dqn import DQNAgent
from experiments.workloads import get_scenario, SCENARIOS
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--arrival-rate', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/dqn_model.pt')
    parser.add_argument('--v2', action='store_true', help='Train v2 hardware-aware DQN')
    parser.add_argument('--scenario', type=str, default='normal_load',
                        choices=list(SCENARIOS.keys()),
                        help='Workload scenario (v2 only)')
    args = parser.parse_args()

    os.makedirs('results/plots', exist_ok=True)

    if args.v2:
        config = get_scenario(args.scenario, seed=args.seed)
        config['max_steps'] = args.max_steps
        print(f"Training v2 DQN on scenario '{args.scenario}' for {args.episodes} episodes...")
        agent, rewards, losses = train_dqn_v2(
            config=config,
            episodes=args.episodes,
            max_steps=args.max_steps,
        )
        if args.output == 'results/dqn_model.pt':
            args.output = f'results/dqn_v2_{args.scenario}.pt'
    else:
        print(f"Training v1 DQN for {args.episodes} episodes...")
        agent, rewards, losses = train_dqn_v1(
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            arrival_rate=args.arrival_rate,
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    agent.save(args.output)
    print(f"Model saved to {args.output}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6, label='Episode Reward')
    window = min(20, len(rewards) // 2)
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(rewards)), smoothed, label=f'Moving Avg ({window})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(losses, alpha=0.6, label='Loss')
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(losses)), smoothed, label=f'Moving Avg ({window})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/plots/training_curves.png', dpi=150)
    plt.close()
    print("Training curves saved to results/plots/training_curves.png")


if __name__ == '__main__':
    main()
