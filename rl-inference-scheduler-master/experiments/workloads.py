import random
from typing import Dict, Callable


def _default_prompt_len(t: int, rng: random.Random) -> int:
    return rng.randint(50, 500)


def _default_output_len(t: int, rng: random.Random) -> int:
    return rng.randint(50, 1000)


def _default_priority(t: int, rng: random.Random) -> int:
    return rng.choices([1, 2, 3, 4, 5], weights=[1, 2, 3, 2, 1])[0]


def normal_load(seed: int = 42) -> Dict:
    return {
        'name': 'normal_load',
        'seed': seed,
        'arrival_rate_func': lambda t: 0.2,
        'prompt_len_func': _default_prompt_len,
        'output_len_func': _default_output_len,
        'priority_func': _default_priority,
        'ttft_slo_mult': 2.0,
        'deadline_mult': 5.0,
        'vram_total_mb': 16384,
        'model_base_memory_mb': 8000,
        'kv_mb_per_token': 0.25,
        'max_concurrent': 4,
        'prefill_speed': 5.0,
        'decode_speed': 10.0,
        'max_steps': 500,
        'description': 'Balanced traffic with mixed prompt lengths and priorities',
    }


def burst_load(seed: int = 42) -> Dict:
    def arrival(t: int) -> float:
        if t < 80:
            return 0.05
        elif t < 200:
            return 0.5
        else:
            return 0.2

    return {
        'name': 'burst_load',
        'seed': seed,
        'arrival_rate_func': arrival,
        'prompt_len_func': _default_prompt_len,
        'output_len_func': _default_output_len,
        'priority_func': _default_priority,
        'ttft_slo_mult': 2.0,
        'deadline_mult': 4.0,
        'vram_total_mb': 16384,
        'model_base_memory_mb': 8000,
        'kv_mb_per_token': 0.25,
        'max_concurrent': 4,
        'prefill_speed': 5.0,
        'decode_speed': 10.0,
        'max_steps': 500,
        'description': 'Traffic spikes from low to burst to medium, testing queue control and batching',
    }


def memory_pressure(seed: int = 42) -> Dict:
    def prompt_len(t: int, rng: random.Random) -> int:
        return rng.randint(500, 2000)

    def output_len(t: int, rng: random.Random) -> int:
        return rng.randint(200, 1000)

    return {
        'name': 'memory_pressure',
        'seed': seed,
        'arrival_rate_func': lambda t: 0.25,
        'prompt_len_func': prompt_len,
        'output_len_func': output_len,
        'priority_func': _default_priority,
        'ttft_slo_mult': 2.0,
        'deadline_mult': 5.0,
        'vram_total_mb': 10000,
        'model_base_memory_mb': 8000,
        'kv_mb_per_token': 0.30,
        'max_concurrent': 4,
        'prefill_speed': 50.0,
        'decode_speed': 2.0,
        'max_steps': 500,
        'description': 'Long-context jobs (500-2000 tokens) under tight 10GB VRAM with slow decode. OOM fires under load, training agent to use memory-protect actions.',
    }


def priority_pressure(seed: int = 42) -> Dict:
    def priority(t: int, rng: random.Random) -> int:
        return rng.choices([1, 2, 3, 4, 5], weights=[10, 5, 1, 1, 1])[0]

    return {
        'name': 'priority_pressure',
        'seed': seed,
        'arrival_rate_func': lambda t: 0.2,
        'prompt_len_func': _default_prompt_len,
        'output_len_func': _default_output_len,
        'priority_func': priority,
        'ttft_slo_mult': 2.0,
        'deadline_mult': 5.0,
        'vram_total_mb': 16384,
        'model_base_memory_mb': 8000,
        'kv_mb_per_token': 0.25,
        'max_concurrent': 4,
        'prefill_speed': 5.0,
        'decode_speed': 10.0,
        'max_steps': 500,
        'description': '10% high-priority mixed with normal jobs, testing priority-aware dispatch',
    }


def adversarial_long_then_short(seed: int = 42) -> Dict:
    def arrival(t: int) -> float:
        if t < 100:
            return 0.12
        elif t < 180:
            return 0.03
        else:
            return 0.3

    def prompt_len(t: int, rng: random.Random) -> int:
        if t < 100:
            return rng.randint(800, 2000)
        elif t < 180:
            return rng.randint(200, 500)
        else:
            return rng.randint(50, 200)

    def output_len(t: int, rng: random.Random) -> int:
        if t < 100:
            return rng.randint(200, 800)
        else:
            return rng.randint(50, 200)

    def priority(t: int, rng: random.Random) -> int:
        if t < 100:
            return rng.choices([1, 2, 3, 4, 5], weights=[1, 1, 3, 2, 1])[0]
        else:
            return rng.choices([1, 2, 3, 4, 5], weights=[8, 4, 1, 1, 1])[0]

    return {
        'name': 'adversarial_long_then_short',
        'seed': seed,
        'arrival_rate_func': arrival,
        'prompt_len_func': prompt_len,
        'output_len_func': output_len,
        'priority_func': priority,
        'ttft_slo_mult': 1.5,
        'deadline_mult': 3.0,
        'vram_total_mb': 16384,
        'model_base_memory_mb': 8000,
        'kv_mb_per_token': 0.25,
        'max_concurrent': 4,
        'prefill_speed': 5.0,
        'decode_speed': 10.0,
        'max_steps': 500,
        'description': 'Long jobs arrive first, then urgent short jobs arrive. Exposes FIFO weakness.',
    }


SCENARIOS = {
    'normal_load': normal_load,
    'burst_load': burst_load,
    'memory_pressure': memory_pressure,
    'priority_pressure': priority_pressure,
    'adversarial_long_then_short': adversarial_long_then_short,
}


def get_scenario(name: str, seed: int = 42) -> Dict:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name](seed=seed)
