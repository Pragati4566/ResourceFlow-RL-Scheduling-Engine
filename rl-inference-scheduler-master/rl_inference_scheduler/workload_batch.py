import random
import torch
from experiments.workloads import get_scenario


class BatchWorkload:
    def __init__(self, scenario_name: str, num_envs: int, base_seed: int = 42):
        config = get_scenario(scenario_name, seed=base_seed)
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.config = config

        self.env_rngs = [random.Random(base_seed + i) for i in range(num_envs)]

        self.arrival_rate_func = config['arrival_rate_func']
        self.prompt_len_func = config['prompt_len_func']
        self.output_len_func = config['output_len_func']
        self.priority_func = config['priority_func']
        self.ttft_slo_mult = config['ttft_slo_mult']
        self.deadline_mult = config['deadline_mult']
        self.kv_mb_per_token = config['kv_mb_per_token']
        self.prefill_speed = config['prefill_speed']
        self.decode_speed = config['decode_speed']
        self.max_steps = config.get('max_steps', 500)

    def set_scenario(self, scenario_name: str):
        config = get_scenario(scenario_name, seed=self.base_seed)
        self.config = config
        self.arrival_rate_func = config['arrival_rate_func']
        self.prompt_len_func = config['prompt_len_func']
        self.output_len_func = config['output_len_func']
        self.priority_func = config['priority_func']
        self.ttft_slo_mult = config['ttft_slo_mult']
        self.deadline_mult = config['deadline_mult']
        self.kv_mb_per_token = config['kv_mb_per_token']
        self.prefill_speed = config['prefill_speed']
        self.decode_speed = config['decode_speed']
        self.max_steps = config.get('max_steps', 500)

    def reset(self, base_seed: int = None):
        if base_seed is not None:
            self.base_seed = base_seed
        self.env_rngs = [random.Random(self.base_seed + i) for i in range(self.num_envs)]

    def generate_arrivals(self, env_times: torch.Tensor):
        prompt_len = torch.full((self.num_envs,), -1, dtype=torch.int32)
        output_len = torch.zeros(self.num_envs, dtype=torch.int32)
        priority = torch.zeros(self.num_envs, dtype=torch.int32)
        arrival_time = torch.zeros(self.num_envs, dtype=torch.int32)
        ttft_slo = torch.zeros(self.num_envs, dtype=torch.int32)
        deadline = torch.zeros(self.num_envs, dtype=torch.int32)

        has_arrival = False
        for i in range(self.num_envs):
            t = env_times[i].item()
            rng = self.env_rngs[i]
            if rng.random() < self.arrival_rate_func(t):
                has_arrival = True
                p = self.prompt_len_func(t, rng)
                o = self.output_len_func(t, rng)
                pr = self.priority_func(t, rng)
                total_work = p / self.prefill_speed + o / self.decode_speed

                prompt_len[i] = p
                output_len[i] = o
                priority[i] = pr
                arrival_time[i] = t
                ttft_slo[i] = max(5, int(p * self.ttft_slo_mult))
                deadline[i] = max(10, int(total_work * self.deadline_mult))

        if not has_arrival:
            return None

        return {
            'prompt_len': prompt_len,
            'output_len': output_len,
            'priority': priority,
            'arrival_time': arrival_time,
            'ttft_slo_ticks': ttft_slo,
            'deadline_ticks': deadline,
        }
