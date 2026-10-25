import numpy as np
from typing import Tuple, Dict, Optional
from .llm_simulator import HardwareAwareServer


def make_observation(server: HardwareAwareServer) -> np.ndarray:
    queue = server.queue
    running = server.running

    q_len = len(queue)
    high_prio = sum(1 for j in queue if j.priority <= 2)
    avg_prompt = float(np.mean([j.prompt_len for j in queue])) if queue else 0.0
    max_prompt = float(max([j.prompt_len for j in queue])) if queue else 0.0
    avg_output = float(np.mean([j.output_len for j in queue])) if queue else 0.0
    avg_prio = float(np.mean([j.priority for j in queue])) if queue else 3.0

    oldest_wait = max((server.time - j.arrival_time for j in queue), default=0.0)
    avg_wait = float(np.mean([server.time - j.arrival_time for j in queue])) if queue else 0.0

    near_ttft = sum(1 for j in queue
                    if (server.time - j.arrival_time) / max(j.ttft_slo_ticks, 1) > 0.8)
    near_deadline = sum(1 for j in queue
                        if (server.time - j.arrival_time) / max(j.deadline_ticks, 1) > 0.8)

    short_jobs = sum(1 for j in queue if j.total_tokens < 300)
    long_context = sum(1 for j in queue if j.prompt_len > 500)

    vram_ratio = server.vram_used_ratio
    vram_headroom = 1.0 - vram_ratio

    obs = np.array([
        min(q_len, 30.0) / 30.0,
        len(running) / server.max_concurrent,
        (server.max_concurrent - len(running)) / server.max_concurrent,
        vram_ratio,
        vram_headroom,
        min(avg_prompt, 1000.0) / 1000.0,
        min(max_prompt, 2000.0) / 2000.0,
        min(avg_output, 2000.0) / 2000.0,
        avg_prio / 5.0,
        min(high_prio, 10.0) / 10.0,
        min(oldest_wait, 300.0) / 300.0,
        min(avg_wait, 300.0) / 300.0,
        min(near_ttft, 10.0) / 10.0,
        min(near_deadline, 10.0) / 10.0,
        min(short_jobs, 10.0) / 10.0,
        min(long_context, 10.0) / 10.0,
        float(server.gpu_active),
        min(server.tokens_generated_this_tick / 20.0, 1.0),
        min(server.ttft_slo_miss_count > 0, 1.0),
        min(server.deadline_miss_count > 0, 1.0),
        min(server.memory_pressure_events > 5, 1.0),
    ], dtype=np.float32)

    return obs


class V2Environment:
    def __init__(self, config: dict, max_steps: int = 500):
        self.config = config.copy()
        self.max_steps = max_steps
        self.current_step = 0
        self.action_space = 12
        self.observation_dim = 21

        self._ema_gpu_util = 0.0
        self._ema_completion_rate = 0.0
        self._ema_ttft_slo_miss_rate = 0.0
        self._ema_deadline_miss_rate = 0.0
        self._ema_memory_pressure_rate = 0.0
        self._ema_gamma = 0.95

        self.server = HardwareAwareServer(self.config)
        self.recent_completions: list = []

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.config['seed'] = seed
        self.server = HardwareAwareServer(self.config)
        self.current_step = 0
        self._ema_gpu_util = 0.0
        self._ema_completion_rate = 0.0
        self._ema_ttft_slo_miss_rate = 0.0
        self._ema_deadline_miss_rate = 0.0
        self._ema_memory_pressure_rate = 0.0
        self.recent_completions.clear()
        return make_observation(self.server)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, float, Dict]:
        state = self.server.step(action)
        self.current_step += 1

        gamma = self._ema_gamma
        self._ema_gpu_util = gamma * self._ema_gpu_util + (1 - gamma) * float(state['gpu_active'])
        self._ema_completion_rate = gamma * self._ema_completion_rate + (1 - gamma) * float(len(state['just_completed']) > 0)
        self._ema_ttft_slo_miss_rate = gamma * self._ema_ttft_slo_miss_rate + (1 - gamma) * float(len(state['ttft_slo_misses_this_tick']) > 0)
        self._ema_deadline_miss_rate = gamma * self._ema_deadline_miss_rate + (1 - gamma) * float(len(state['deadline_misses_this_tick']) > 0)
        self._ema_memory_pressure_rate = gamma * self._ema_memory_pressure_rate + (1 - gamma) * float(state['vram_used_ratio'] > 0.90)

        obs = make_observation(self.server)
        clipped, unclipped = self._compute_reward(state)
        done = self.current_step >= self.max_steps

        state_for_info = dict(state)
        state_for_info['clipped_reward'] = clipped
        state_for_info['unclipped_reward'] = unclipped
        state_for_info['ema_gpu_util'] = self._ema_gpu_util
        state_for_info['ema_completion_rate'] = self._ema_completion_rate

        return obs, clipped, done, state_for_info

    def _compute_reward(self, state: Dict) -> Tuple[float, float]:
        reward = 0.0

        for job in state['just_completed']:
            reward += 2.0
            if job.priority <= 2:
                reward += 3.0

        reward += 0.5 * state.get('tokens_generated_this_tick', 0) / 10.0

        reward -= 4.0 * len(state.get('ttft_slo_misses_this_tick', []))
        reward -= 6.0 * len(state.get('deadline_misses_this_tick', []))
        reward -= 10.0 * state.get('oom_events_this_tick', 0)

        for job in state.get('dropped_this_tick', []):
            reward -= 2.0
            if job.priority <= 2:
                reward -= 1.0

        active = state.get('running_count', 0)
        idle = state['max_concurrent'] - active
        if idle > 0 and state.get('queue_length', 0) > 0:
            reward -= 0.5 * idle

        vram_ratio = state.get('vram_used_ratio', 0)
        if vram_ratio > 0.90:
            reward -= 2.0 * (vram_ratio - 0.90) * 10

        if state.get('oldest_job_wait', 0) > state.get('starvation_threshold', 100):
            reward -= 3.0

        unclipped = reward
        clipped = float(np.clip(reward, -10.0, 10.0))
        return clipped, unclipped
