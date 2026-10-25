import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import tempfile
import torch
import pytest

from rl_inference_scheduler.trace_replay import load_trace_csv, TraceRequest, TraceReplayWorkload


@pytest.fixture
def sample_csv():
    lines = [
        "arrival_time,prompt_len,output_len,priority,deadline_ticks,ttft_slo_ticks,user_type",
        "0,128,256,3,400,80,normal",
        "1,512,128,4,600,120,batch",
        "2,64,64,1,120,30,interactive",
        "5,256,512,2,300,60,normal",
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write('\n'.join(lines))
        path = f.name
    yield path
    os.unlink(path)


def test_trace_csv_loads(sample_csv):
    requests = load_trace_csv(sample_csv)
    assert len(requests) == 4
    assert requests[0].arrival_time == 0
    assert requests[0].prompt_len == 128
    assert requests[0].priority == 3
    assert requests[0].user_type == 'normal'
    assert requests[1].user_type == 'batch'


def test_arrivals_at_correct_tick(sample_csv):
    requests = load_trace_csv(sample_csv)
    wl = TraceReplayWorkload(requests, num_envs=2)

    tick0 = torch.zeros(2, dtype=torch.int32)
    arrivals = wl.generate_arrivals(tick0)
    assert arrivals is not None, "Should have arrivals at tick 0"
    assert arrivals['prompt_len'][0].item() == 128

    tick1 = torch.ones(2, dtype=torch.int32)
    arrivals = wl.generate_arrivals(tick1)
    assert arrivals is not None
    mask = arrivals['prompt_len'] >= 0
    assert mask[1].item(), "Second env should get arrival at tick 1"

    tick2 = torch.full((2,), 2, dtype=torch.int32)
    arrivals = wl.generate_arrivals(tick2)
    assert arrivals is not None
    mask = arrivals['prompt_len'] >= 0
    assert mask[0].item(), "Third arrival should go to env 0 (round-robin)"

    tick5 = torch.full((2,), 5, dtype=torch.int32)
    arrivals = wl.generate_arrivals(tick5)
    assert arrivals is not None
    assert arrivals['prompt_len'][1].item() == 256


def test_deterministic_replay(sample_csv):
    requests = load_trace_csv(sample_csv)
    wl1 = TraceReplayWorkload(requests, num_envs=2)
    wl2 = TraceReplayWorkload(requests, num_envs=2)

    for t in range(10):
        tick = torch.full((2,), t, dtype=torch.int32)
        a1 = wl1.generate_arrivals(tick)
        a2 = wl2.generate_arrivals(tick)
        if a1 is None and a2 is None:
            continue
        assert a1 is not None and a2 is not None
        assert (a1['prompt_len'] == a2['prompt_len']).all()
        assert (a1['output_len'] == a2['output_len']).all()
        assert (a1['priority'] == a2['priority']).all()


def test_defaults_for_optional_fields(sample_csv):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("arrival_time,prompt_len,output_len,priority,deadline_ticks,ttft_slo_ticks\n")
        f.write("0,128,256,3,400,80\n")
        path = f.name

    try:
        requests = load_trace_csv(path)
        assert len(requests) == 1
        assert requests[0].user_type == 'normal'
    finally:
        os.unlink(path)


def test_malformed_trace_raises():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("arrival_time,prompt_len,output_len,priority,deadline_ticks,ttft_slo_ticks\n")
        f.write("abc,128,256,3,400,80\n")
        path = f.name

    try:
        with pytest.raises(ValueError, match="Error parsing trace"):
            load_trace_csv(path)
    finally:
        os.unlink(path)


def test_single_env_equivalence_with_trace(sample_csv):
    requests = load_trace_csv(sample_csv)
    wl = TraceReplayWorkload(requests, num_envs=1)
    wl2 = TraceReplayWorkload(requests, num_envs=1)

    from rl_inference_scheduler.vectorized_env import TensorizedBatchEnv
    env1 = TensorizedBatchEnv(num_envs=1, scenario='normal_load', max_steps=20, device='cpu')
    env1.workload = wl
    obs1 = env1.reset(seed=42)

    env2 = TensorizedBatchEnv(num_envs=1, scenario='normal_load', max_steps=20, device='cpu')
    env2.workload = wl2
    obs2 = env2.reset(seed=42)

    actions = torch.zeros(1, dtype=torch.long)
    for _ in range(15):
        obs1, r1, d1, _ = env1.step(actions)
        obs2, r2, d2, _ = env2.step(actions)
        assert (obs1 == obs2).all(), f"Observation mismatch at time {env1.env_time[0].item()}"
        assert r1 == r2
        assert d1 == d2