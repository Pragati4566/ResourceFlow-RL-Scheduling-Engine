import numpy as np
from typing import List, Tuple
from .environment import SchedulingEnvironment
from .dqn import DQNAgent


def train_dqn(
    episodes: int = 500,
    max_steps: int = 200,
    seed: int = 42,
    arrival_rate: float = 0.2,
    render_every: int = 50,
) -> Tuple[DQNAgent, List[float], List[float]]:
    env = SchedulingEnvironment(max_steps=max_steps, seed=seed, arrival_rate=arrival_rate)
    agent = DQNAgent(
        state_dim=env.observation_dim,
        action_dim=env.action_space,
    )

    episode_rewards: List[float] = []
    episode_losses: List[float] = []

    for episode in range(episodes):
        state = env.reset(seed=seed + episode)
        total_reward = 0.0
        losses: List[float] = []

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.update()
            if loss > 0:
                losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)
        avg_loss = float(np.mean(losses)) if losses else 0.0
        episode_losses.append(avg_loss)

        agent.decay_epsilon()

        if (episode + 1) % render_every == 0:
            print(
                f"Episode {episode+1}/{episodes} | "
                f"Reward: {total_reward:>8.1f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f}"
            )

    return agent, episode_rewards, episode_losses
