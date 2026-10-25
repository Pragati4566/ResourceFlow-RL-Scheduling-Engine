import csv
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


@dataclass
class TraceRequest:
    arrival_time: int
    prompt_len: int
    output_len: int
    priority: int
    deadline_ticks: int
    ttft_slo_ticks: int
    user_type: str = "normal"


def load_trace_csv(path: str) -> List[TraceRequest]:
    requests = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for lineno, row in enumerate(reader, start=2):
            try:
                arrival_time = int(row['arrival_time'])
                prompt_len = int(row['prompt_len'])
                output_len = int(row['output_len'])
                priority = int(row['priority'])
                deadline_ticks = int(row['deadline_ticks'])
                ttft_slo_ticks = int(row['ttft_slo_ticks'])
                user_type = row.get('user_type', 'normal').strip()
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"Error parsing trace line {lineno}: {e}. "
                    f"Expected columns: arrival_time,prompt_len,output_len,"
                    f"priority,deadline_ticks,ttft_slo_ticks[,user_type]"
                )
            if prompt_len < 1 or output_len < 1:
                raise ValueError(f"Line {lineno}: prompt_len and output_len must be >= 1")
            if not (1 <= priority <= 5):
                raise ValueError(f"Line {lineno}: priority must be 1-5")
            requests.append(TraceRequest(
                arrival_time=arrival_time,
                prompt_len=prompt_len,
                output_len=output_len,
                priority=priority,
                deadline_ticks=deadline_ticks,
                ttft_slo_ticks=ttft_slo_ticks,
                user_type=user_type,
            ))
    return requests


class TraceReplayWorkload:
    def __init__(self, requests: List[TraceRequest], num_envs: int):
        requests = sorted(requests, key=lambda r: r.arrival_time)
        self.requests = requests
        self.num_envs = num_envs
        self.pos = 0
        self._sentinel = -1

    def reset(self, seed: Optional[int] = None):
        self.pos = 0

    def generate_arrivals(self, env_times: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        device = env_times.device
        n = self.num_envs
        prompt_len = torch.full((n,), self._sentinel, dtype=torch.int32, device=device)
        output_len = torch.zeros(n, dtype=torch.int32, device=device)
        priority = torch.zeros(n, dtype=torch.int32, device=device)
        arrival_time = torch.zeros(n, dtype=torch.int32, device=device)
        ttft_slo_ticks = torch.zeros(n, dtype=torch.int32, device=device)
        deadline_ticks = torch.zeros(n, dtype=torch.int32, device=device)

        current_tick = int(env_times[0].item())
        any_arrival = False

        while self.pos < len(self.requests):
            req = self.requests[self.pos]
            if req.arrival_time > current_tick:
                break

            env_id = self.pos % n

            prompt_len[env_id] = req.prompt_len
            output_len[env_id] = req.output_len
            priority[env_id] = req.priority
            arrival_time[env_id] = req.arrival_time
            ttft_slo_ticks[env_id] = req.ttft_slo_ticks
            deadline_ticks[env_id] = req.deadline_ticks
            any_arrival = True

            self.pos += 1

        if not any_arrival:
            return None

        return {
            'prompt_len': prompt_len,
            'output_len': output_len,
            'priority': priority,
            'arrival_time': arrival_time,
            'ttft_slo_ticks': ttft_slo_ticks,
            'deadline_ticks': deadline_ticks,
        }