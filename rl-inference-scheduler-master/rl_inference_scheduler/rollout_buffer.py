import torch


class RolloutBuffer:
    def __init__(self, buffer_size, num_envs, obs_dim, device='cpu'):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.pos = 0
        self.full = False

        self.observations = torch.zeros(buffer_size, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(buffer_size, num_envs, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, num_envs, device=device)
        self.dones = torch.zeros(buffer_size, num_envs, dtype=torch.bool, device=device)
        self.logprobs = torch.zeros(buffer_size, num_envs, device=device)
        self.values = torch.zeros(buffer_size, num_envs, device=device)
        self.advantages = torch.zeros(buffer_size, num_envs, device=device)
        self.returns = torch.zeros(buffer_size, num_envs, device=device)

    def store(self, obs, action, reward, done, logprob, value):
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.logprobs[self.pos] = logprob
        self.values[self.pos] = value
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.pos = 0
            self.full = True

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        gae = 0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * (~self.dones[t]).float() - self.values[t]
            gae = delta + gamma * lam * (~self.dones[t]).float() * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]

    def get_minibatch(self, batch_size):
        n = self.buffer_size if self.full else self.pos
        total = n * self.num_envs

        obs_flat = self.observations[:n].reshape(n * self.num_envs, -1)
        act_flat = self.actions[:n].reshape(n * self.num_envs)
        adv_flat = self.advantages[:n].reshape(n * self.num_envs)
        ret_flat = self.returns[:n].reshape(n * self.num_envs)
        logp_flat = self.logprobs[:n].reshape(n * self.num_envs)

        indices = torch.randperm(total, device=self.device)
        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            yield obs_flat[idx], act_flat[idx], adv_flat[idx], ret_flat[idx], logp_flat[idx]

    def clear(self):
        self.pos = 0
        self.full = False
