import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import math
import torch
import numpy as np

from rl_inference_scheduler.trace_replay import load_trace_csv, TraceReplayWorkload
from rl_inference_scheduler.vectorized_env import TensorizedBatchEnv
from rl_inference_scheduler.ppo import PPOAgent


def collect_metrics(env, submitted_count):
    details = env.completed_details
    n_completed = len(details)
    n_dropped = int(env.env_drops.sum().item())
    total_finished = n_completed + n_dropped

    if n_completed == 0:
        return {
            'avg_latency': 0, 'p95_latency': 0, 'p99_latency': 0,
            'avg_ttft': 0, 'p95_ttft': 0, 'p99_ttft': 0,
            'high_priority_p95_ttft': 0, 'high_priority_avg_ttft': 0,
            'deadline_miss_rate': 0, 'ttft_slo_miss_rate': 0,
            'high_priority_completion_rate': 0,
            'throughput': 0,
            'memory_pressure_event_rate': 0, 'oom_event_rate': 0,
            'starvation_rate': 0,
            'n_completed': 0, 'n_dropped': n_dropped, 'n_submitted': submitted_count,
        }

    latencies = [d['completion_time'] - d['arrival_time'] for d in details]
    ttfts = [d['first_token_time'] - d['arrival_time'] for d in details
             if d['first_token_time'] >= 0]
    high_prio_ttfts = [d['first_token_time'] - d['arrival_time'] for d in details
                       if d['priority'] <= 2 and d['first_token_time'] >= 0]

    total_ticks = max(1, int(env.env_time.max().item()))
    throughput = n_completed / total_ticks

    deadline_misses = sum(1 for d in details
                          if d['completion_time'] - d['arrival_time'] > d['deadline'])
    ttft_misses = sum(1 for d in details
                      if d['first_token_time'] >= 0
                      and d['first_token_time'] - d['arrival_time'] > d['ttft_slo'])

    high_prio_completed = sum(1 for d in details if d['priority'] <= 2)
    high_prio_total = sum(1 for d in details if d['priority'] <= 2)

    mem_pressure_events = int(env.env_memory_pressure_events.sum().item())
    oom_events = int(env.env_oom_events.sum().item())
    gpu_idle = int(env.env_gpu_idle_ticks.sum().item())

    starve_count = 0

    return {
        'avg_latency': float(np.mean(latencies)),
        'p95_latency': float(np.percentile(latencies, 95)),
        'p99_latency': float(np.percentile(latencies, 99)),
        'avg_ttft': float(np.mean(ttfts)) if ttfts else 0,
        'p95_ttft': float(np.percentile(ttfts, 95)) if ttfts else 0,
        'p99_ttft': float(np.percentile(ttfts, 99)) if ttfts else 0,
        'high_priority_avg_ttft': float(np.mean(high_prio_ttfts)) if high_prio_ttfts else 0,
        'high_priority_p95_ttft': float(np.percentile(high_prio_ttfts, 95)) if high_prio_ttfts else 0,
        'deadline_miss_rate': deadline_misses / max(n_completed, 1),
        'ttft_slo_miss_rate': ttft_misses / max(n_completed, 1),
        'high_priority_completion_rate': high_prio_completed / max(high_prio_total, 1),
        'throughput': throughput,
        'memory_pressure_event_rate': mem_pressure_events / total_ticks,
        'oom_event_rate': oom_events / total_ticks,
        'starvation_rate': gpu_idle / total_ticks,
        'n_completed': n_completed,
        'n_dropped': n_dropped,
        'n_submitted': submitted_count,
    }


SCHEDULER_ACTIONS = {
    'fifo': 0,
    'sjf': 2,
    'priority': 1,
    'largest_safe_batch': 6,
    'mem_aware_priority': 11,
}


def run_heuristic(requests, num_envs, action_id, max_steps, seed):
    workload = TraceReplayWorkload(requests, num_envs)
    env = TensorizedBatchEnv(num_envs=num_envs, scenario='normal_load',
                             max_steps=max_steps, device='cpu')
    env.workload = workload
    env.reset(seed=seed)

    actions = torch.full((num_envs,), action_id, dtype=torch.long)

    for step in range(max_steps):
        _, reward, done, info = env.step(actions)
        if done:
            break

    metrics = collect_metrics(env, len(requests))
    return metrics


def run_ppo(requests, num_envs, model_path, max_steps, seed):
    workload = TraceReplayWorkload(requests, num_envs)
    env = TensorizedBatchEnv(num_envs=num_envs, scenario='normal_load',
                             max_steps=max_steps, device='cpu')
    env.workload = workload
    obs = env.reset(seed=seed)

    agent = PPOAgent(obs_dim=21, action_dim=12, device='cpu')
    agent.load(model_path)
    agent.actor_critic.eval()

    for step in range(max_steps):
        action, _, _ = agent.act(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break

    return collect_metrics(env, len(requests))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=str, required=True)
    parser.add_argument('--scheduler', type=str, default=None,
                        help='fifo, sjf, priority, largest_safe_batch, mem_aware_priority, ppo')
    parser.add_argument('--all-schedulers', action='store_true')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--ppo-model', type=str, default='saved_models/ppo_v3.pt')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    requests = load_trace_csv(args.trace)
    trace_name = os.path.splitext(os.path.basename(args.trace))[0]
    print(f"Loaded {len(requests)} requests from {args.trace}")

    if args.all_schedulers:
        schedulers = list(SCHEDULER_ACTIONS.keys()) + ['ppo']
    elif args.scheduler:
        schedulers = [args.scheduler]
    else:
        schedulers = list(SCHEDULER_ACTIONS.keys())

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for sched_name in schedulers:
        for seed in args.seeds:
            print(f"  {sched_name} (seed={seed})...", end=' ', flush=True)
            if sched_name == 'ppo':
                if not os.path.exists(args.ppo_model):
                    print(f"SKIP (model not found at {args.ppo_model})")
                    continue
                metrics = run_ppo(requests, args.num_envs, args.ppo_model,
                                  args.max_steps, seed)
            else:
                metrics = run_heuristic(requests, args.num_envs,
                                        SCHEDULER_ACTIONS[sched_name],
                                        args.max_steps, seed)
            metrics['scheduler'] = sched_name
            metrics['seed'] = seed
            metrics['trace'] = trace_name
            all_results.append(metrics)
            print(f"completed={metrics['n_completed']} throughput={metrics['throughput']:.4f}")

    csv_path = os.path.join(args.output_dir, 'trace_metrics.csv')
    fieldnames = ['trace', 'scheduler', 'seed',
                  'avg_latency', 'p95_latency', 'p99_latency',
                  'avg_ttft', 'p95_ttft', 'p99_ttft',
                  'high_priority_avg_ttft', 'high_priority_p95_ttft',
                  'deadline_miss_rate', 'ttft_slo_miss_rate',
                  'high_priority_completion_rate', 'throughput',
                  'memory_pressure_event_rate', 'oom_event_rate',
                  'starvation_rate',
                  'n_completed', 'n_dropped', 'n_submitted']
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in all_results:
            filtered = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered)

    print(f"\nResults appended to {csv_path}")


if __name__ == '__main__':
    main()