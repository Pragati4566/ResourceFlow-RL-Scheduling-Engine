import torch
import torch.nn as nn
import torch.optim as optim


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.features(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value

    def act(self, obs):
        logits, value = self.forward(obs)
        probs = torch.distributions.Categorical(logits=logits)
        action = probs.sample()
        logprob = probs.log_prob(action)
        return action, logprob, value

    def evaluate(self, obs, action):
        logits, value = self.forward(obs)
        probs = torch.distributions.Categorical(logits=logits)
        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        return logprob, entropy, value


class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_epsilon=0.2, epochs=10, batch_size=64, device='cpu'):
        self.device = device
        self.actor_critic = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size

    def act(self, obs):
        with torch.no_grad():
            return self.actor_critic.act(obs)

    def update(self, buffer, last_value):
        buffer.compute_gae(last_value, self.gamma, self.lam)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        updates = 0

        for _ in range(self.epochs):
            for batch in buffer.get_minibatch(self.batch_size):
                obs_b, act_b, adv_b, ret_b, old_logp_b = [x.to(self.device) for x in batch]

                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                logprob, entropy, value = self.actor_critic.evaluate(obs_b, act_b)

                ratio = (logprob - old_logp_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(value, ret_b)

                entropy_loss = -entropy.mean() * 0.01

                loss = policy_loss + 0.5 * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                updates += 1

        n = updates if updates > 0 else 1
        return {
            'loss': total_loss / n,
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n,
            'entropy': total_entropy / n,
        }

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
