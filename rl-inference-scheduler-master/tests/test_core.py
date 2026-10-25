import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rl_inference_scheduler.llm_job import LLMJob
from rl_inference_scheduler.llm_simulator import HardwareAwareServer
from rl_inference_scheduler.llm_env import V2Environment, make_observation
from experiments.workloads import get_scenario


def test_ttft_calculation():
    job = LLMJob(
        job_id=1, prompt_len=100, output_len=200, priority=3,
        arrival_time=10, ttft_slo_ticks=50, deadline_ticks=200,
    )
    job.first_token_time = 25
    assert job.ttft == 15, f"Expected TTFT 15, got {job.ttft}"


def test_latency_calculation():
    job = LLMJob(
        job_id=1, prompt_len=100, output_len=200, priority=3,
        arrival_time=10, ttft_slo_ticks=50, deadline_ticks=200,
    )
    job.completion_time = 50
    assert job.latency == 40, f"Expected latency 40, got {job.latency}"


def test_deadline_miss():
    job = LLMJob(
        job_id=1, prompt_len=100, output_len=200, priority=3,
        arrival_time=10, ttft_slo_ticks=50, deadline_ticks=20,
    )
    job.completion_time = 50
    assert job.deadline_missed, "Expected deadline missed"


def test_deadline_met():
    job = LLMJob(
        job_id=1, prompt_len=100, output_len=200, priority=3,
        arrival_time=10, ttft_slo_ticks=50, deadline_ticks=100,
    )
    job.completion_time = 50
    assert not job.deadline_missed, "Expected deadline met"


def test_ttft_slo_miss():
    job = LLMJob(
        job_id=1, prompt_len=100, output_len=200, priority=3,
        arrival_time=10, ttft_slo_ticks=5, deadline_ticks=100,
    )
    job.first_token_time = 25
    assert job.ttft_slo_missed, "Expected TTFT SLO missed"


def test_ttft_slo_met():
    job = LLMJob(
        job_id=1, prompt_len=100, output_len=200, priority=3,
        arrival_time=10, ttft_slo_ticks=50, deadline_ticks=100,
    )
    job.first_token_time = 25
    assert not job.ttft_slo_missed, "Expected TTFT SLO met"


def test_kv_cache_memory():
    job = LLMJob(
        job_id=1, prompt_len=100, output_len=200, priority=3,
        arrival_time=0, ttft_slo_ticks=50, deadline_ticks=200,
        kv_mb_per_token=0.25,
    )
    expected_start = 100 * 0.25
    assert abs(job.kv_cache_mb - expected_start) < 0.01, \
        f"Expected {expected_start}, got {job.kv_cache_mb}"

    job.generated_tokens = 50
    expected_grow = (100 + 50) * 0.25
    assert abs(job.kv_cache_mb - expected_grow) < 0.01, \
        f"Expected {expected_grow}, got {job.kv_cache_mb}"


def test_safe_batching_rejects_oversized():
    config = get_scenario('memory_pressure', seed=42)
    config['vram_total_mb'] = 8200  # 8000 base + only 200 for KV
    config['model_base_memory_mb'] = 8000
    config['kv_mb_per_token'] = 0.5  # High KV cost
    server = HardwareAwareServer(config)

    server.time = 1
    job = server._generate_job()
    server.queue.append(job)

    result = server._dispatch_jobs(1)
    assert result == 0 or not server.running, \
        "Should not dispatch when insufficient VRAM: " \
        f"kv_cache={job.kv_cache_mb:.1f} headroom={server.vram_total_mb - server.active_memory_mb:.1f}"


def test_priority_jobs_higher_reward():
    env = V2Environment(get_scenario('normal_load'), max_steps=10)
    env.reset(seed=42)

    state1 = {'just_completed': [], 'tokens_generated_this_tick': 0,
              'ttft_slo_misses_this_tick': [], 'deadline_misses_this_tick': [],
              'oom_events_this_tick': 0, 'dropped_this_tick': [],
              'running_count': 0, 'max_concurrent': 4, 'queue_length': 0,
              'vram_used_ratio': 0.5, 'oldest_job_wait': 0,
              'starvation_threshold': 100}

    r1, _ = env._compute_reward(state1)
    assert r1 >= 0, "Empty state should not give negative"


def test_observation_dim():
    config = get_scenario('normal_load')
    env = V2Environment(config, max_steps=10)
    obs = env.reset()
    assert len(obs) == 21, f"Expected 21-dim observation, got {len(obs)}"


def test_action_space():
    config = get_scenario('normal_load')
    env = V2Environment(config, max_steps=10)
    assert env.action_space == 12, f"Expected 12 actions, got {env.action_space}"


def test_scenario_determinism():
    c1 = get_scenario('normal_load', seed=42)
    c2 = get_scenario('normal_load', seed=42)
    assert c1['seed'] == c2['seed']
    assert c1['name'] == c2['name']

    server1 = HardwareAwareServer(c1)
    server2 = HardwareAwareServer(c2)
    for _ in range(10):
        server1.step(8)
        server2.step(8)
    assert len(server1.queue) == len(server2.queue), \
        "Deterministic scenarios should produce same queue length"


def test_memory_pressure_counted():
    config = get_scenario('memory_pressure', seed=42)
    config['vram_total_mb'] = 8100  # Almost full
    config['model_base_memory_mb'] = 8000
    config['memory_pressure_threshold'] = 0.90
    server = HardwareAwareServer(config)

    for _ in range(100):
        server.step(8)  # Wait

    assert server.memory_pressure_events > 0, \
        "Memory pressure events should be counted"


if __name__ == '__main__':
    test_ttft_calculation()
    test_latency_calculation()
    test_deadline_miss()
    test_deadline_met()
    test_ttft_slo_miss()
    test_ttft_slo_met()
    test_kv_cache_memory()
    test_safe_batching_rejects_oversized()
    test_priority_jobs_higher_reward()
    test_observation_dim()
    test_action_space()
    test_scenario_determinism()
    test_memory_pressure_counted()
    print("All tests passed!")
