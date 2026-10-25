import numpy as np
from typing import Tuple, Dict, Optional
from .simulator import InferenceServer


def make_observation(server: InferenceServer) -> np.ndarray:
    queue = server.queue

    prio_counts = [0, 0, 0, 0, 0]
    total_queued_compute = 0.0
    total_queued_memory = 0.0
    total_wait = 0
    for j in queue:
        prio_counts[j.priority - 1] += 1
        total_queued_compute += j.compute_units
        total_queued_memory += j.memory_mb
        total_wait += server.time - j.arrival_time

    avg_wait = total_wait / len(queue) if queue else 0.0

    obs = np.array([
        min(float(len(queue)), 30.0) / 30.0,
        float(len(server.running)) / float(server.max_concurrent),
        server.vram_used / max(server.vram_total, 1),
        server.vram_free / max(server.vram_total, 1),
        float(server.gpu_idle),
        min(float(server.gpu_idle_ticks), 50.0) / 50.0,
        min(float(prio_counts[0]), 10.0) / 10.0,
        min(float(prio_counts[1]), 10.0) / 10.0,
        min(float(prio_counts[3] + prio_counts[4]), 10.0) / 10.0,
        min(total_queued_compute, 1000.0) / 1000.0,
        min(total_queued_memory, server.vram_total) / max(server.vram_total, 1),
        min(avg_wait, 200.0) / 200.0,
    ], dtype=np.float32)

    return obs


class SchedulingEnvironment:
    def __init__(self, max_steps: int = 200, seed: int = 42, arrival_rate: float = 0.2):
        self.server = InferenceServer(seed=seed, arrival_rate=arrival_rate)
        self.max_steps = max_steps
        self.current_step = 0
        self.action_space = 6
        self.observation_dim = 12

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self.server.reset(seed=seed)
        self.current_step = 0
        return make_observation(self.server)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        state = self.server.step(action)
        self.current_step += 1
        obs = make_observation(self.server)
        reward = self._compute_reward(state)
        done = self.current_step >= self.max_steps
        return obs, reward, done, state

    def _compute_reward(self, state: Dict) -> float:
        reward = 0.0

        for job in state['just_completed']:
            prio_weight = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.5, 5: 0.25}.get(job.priority, 1.0)
            reward += 10.0 * prio_weight

        if state['vram_used'] > state['vram_total'] * 0.95:
            reward -= 50.0

        if state['gpu_idle'] and len(state['queue']) > 0:
            reward -= 5.0

        reward -= 10.0 * len(state.get('dropped_this_tick', []))

        reward -= 0.1 * len(state['queue'])

        return np.clip(reward, -10.0, 10.0)
