import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rl_inference_scheduler.vectorized_env import TensorizedBatchEnv
from rl_inference_scheduler.rollout_buffer import RolloutBuffer
from rl_inference_scheduler.ppo import PPOAgent


CURRICULUM_PHASES = [
    (0, 'normal_load'),
    (10, 'burst_load'),
    (30, 'priority_pressure'),
    (50, 'memory_pressure'),
    (70, 'adversarial_long_then_short'),
]


def train_ppo_vectorized(num_envs=256, scenario='normal_load', rollout_steps=200,
                         epochs=50, lr=3e-4, device='cpu'):
    init_scenario = 'normal_load' if scenario == 'mixed_curriculum' else scenario
    env = TensorizedBatchEnv(num_envs=num_envs, scenario=init_scenario,
                             max_steps=rollout_steps + 1, device=device)
    agent = PPOAgent(obs_dim=21, action_dim=12, lr=lr, device=device)
    buffer = RolloutBuffer(rollout_steps, num_envs, obs_dim=21, device=device)

    episode_rewards = []
    scenario_labels = []
    infos = {
        'loss': [], 'policy_loss': [], 'value_loss': [], 'entropy': [],
        'completions': [], 'drops': [], 'oom': [],
        'ttft_misses': [], 'deadline_misses': [],
    }

    is_curriculum = (scenario == 'mixed_curriculum')

    def get_phase_scenario(epoch):
        for start, s in reversed(CURRICULUM_PHASES):
            if epoch >= start:
                return s
        return 'normal_load'

    current_scenario = None
    seed = 42
    obs = env.reset(seed=seed)

    for epoch in range(epochs):
        if is_curriculum:
            new_scenario = get_phase_scenario(epoch)
            if new_scenario != current_scenario:
                current_scenario = new_scenario
                env.set_scenario(current_scenario)
                obs = env.reset(seed=seed + epoch)
                if epoch >= 100:
                    import random as _random
                    current_scenario = _random.choice([
                        'normal_load', 'burst_load', 'priority_pressure',
                        'memory_pressure', 'adversarial_long_then_short'])

        buffer.clear()
        ep_reward = 0.0

        for step in range(rollout_steps):
            action, logprob, value = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            buffer.store(obs, action, reward, done, logprob, value)
            obs = next_obs
            ep_reward += reward.sum().item()

            if done:
                obs = env.reset(seed=seed + epoch + 1)
                break

        _, _, last_value = agent.act(obs)

        stats = agent.update(buffer, last_value)

        avg_reward = ep_reward / num_envs
        episode_rewards.append(avg_reward)
        scenario_labels.append(current_scenario if is_curriculum else scenario)
        for k in stats:
            infos[k].append(stats[k])

        label = current_scenario if is_curriculum else scenario
        sys.stdout.write(
            f"Epoch {epoch+1:3d}/{epochs} [{label:<15}] | "
            f"Reward: {avg_reward:>7.2f} | "
            f"Loss: {stats['loss']:.4f} | "
            f"Entropy: {stats['entropy']:.3f} | "
            f"C: {info['completions'].sum().item():.0f} "
            f"D: {info['drops'].sum().item():.0f} "
            f"O: {info['oom'].sum().item():.0f}"
        )
        if info['ttft_slo_misses'].sum().item() > 0:
            sys.stdout.write(f" TTFT:{info['ttft_slo_misses'].sum().item()}")
        sys.stdout.write("\n")
        sys.stdout.flush()

    os.makedirs('saved_models', exist_ok=True)
    save_name = 'ppo_v3_mixed.pt' if is_curriculum else 'ppo_v3.pt'
    agent.save(f'saved_models/{save_name}')

    os.makedirs('plots', exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if is_curriculum:
        colors = {'normal_load': 'blue', 'burst_load': 'orange',
                  'priority_pressure': 'green', 'memory_pressure': 'red',
                  'adversarial_long_then_short': 'purple'}
        for i in range(len(episode_rewards)):
            c = colors.get(scenario_labels[i], 'gray')
            axes[0].scatter(i, episode_rewards[i], color=c, s=8)
        axes[0].plot(episode_rewards, color='gray', alpha=0.3)
    else:
        axes[0].plot(episode_rewards)
    axes[0].set_title('Average Episode Reward')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Reward')
    axes[1].plot(infos['loss'], label='Total Loss')
    axes[1].plot(infos['policy_loss'], label='Policy Loss')
    axes[1].plot(infos['value_loss'], label='Value Loss')
    axes[1].set_title('Training Losses')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[2].plot(infos['entropy'])
    axes[2].set_title('Policy Entropy')
    axes[2].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('plots/ppo_training_v3.png', dpi=150)
    plt.close()

    print(f"\nTraining complete. Model saved to saved_models/{save_name}")
    return agent, episode_rewards, infos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--scenario', type=str, default='normal_load')
    parser.add_argument('--rollout-steps', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    train_ppo_vectorized(
        num_envs=args.num_envs,
        scenario=args.scenario,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
