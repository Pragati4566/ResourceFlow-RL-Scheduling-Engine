import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import glob
import torch

from rl_inference_scheduler.trace_replay import load_trace_csv, TraceReplayWorkload
from rl_inference_scheduler.vectorized_env import TensorizedBatchEnv
from rl_inference_scheduler.ppo import PPOAgent
from experiments.replay_trace import run_heuristic, run_ppo, SCHEDULER_ACTIONS, collect_metrics


TRAIN_SCENARIOS = ['normal_load', 'burst_load', 'priority_pressure']
EVAL_SCENARIOS = ['memory_pressure', 'adversarial_long_then_short']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces-dir', type=str, default='traces')
    parser.add_argument('--ppo-model', type=str, default='saved_models/ppo_v3_mixed.pt')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    eval_sources = []

    for sc in EVAL_SCENARIOS:
        eval_sources.append(('scenario', sc))

    trace_files = sorted(glob.glob(f'{args.traces_dir}/*.csv'))
    for tf in trace_files:
        name = os.path.splitext(os.path.basename(tf))[0]
        eval_sources.append(('trace', name, tf))

    schedulers = {
        'FIFO': 0,
        'SJF': 2,
        'Priority': 1,
        'LargestSafeBatch': 6,
        'MemAwarePriority': 11,
    }

    all_results = []

    for source in eval_sources:
        if source[0] == 'scenario':
            sc_name = source[1]
            print(f"\n=== Scenario: {sc_name} ===")
            for sched_name, action_id in schedulers.items():
                for seed in args.seeds:
                    from rl_inference_scheduler.workload_batch import BatchWorkload
                    env = TensorizedBatchEnv(num_envs=args.num_envs, scenario=sc_name,
                                             max_steps=args.max_steps, device='cpu')
                    env.reset(seed=seed)
                    actions = torch.full((args.num_envs,), action_id, dtype=torch.long)
                    for _ in range(args.max_steps):
                        _, _, done, _ = env.step(actions)
                        if done:
                            break
                    metrics = collect_metrics(env, 0)
                    metrics['source'] = sc_name
                    metrics['scheduler'] = sched_name
                    metrics['seed'] = seed
                    all_results.append(metrics)
                    print(f"  {sched_name}: com={metrics['n_completed']} thpt={metrics['throughput']:.3f}")

            if os.path.exists(args.ppo_model):
                for seed in args.seeds:
                    env = TensorizedBatchEnv(num_envs=args.num_envs, scenario=sc_name,
                                             max_steps=args.max_steps, device='cpu')
                    env.reset(seed=seed)
                    agent = PPOAgent(obs_dim=21, action_dim=12, device='cpu')
                    agent.load(args.ppo_model)
                    agent.actor_critic.eval()
                    obs = env.reset(seed=seed)
                    for _ in range(args.max_steps):
                        action, _, _ = agent.act(obs)
                        obs, _, done, _ = env.step(action)
                        if done:
                            break
                    metrics = collect_metrics(env, 0)
                    metrics['source'] = sc_name
                    metrics['scheduler'] = 'PPO'
                    metrics['seed'] = seed
                    all_results.append(metrics)
                    print(f"  PPO: com={metrics['n_completed']} thpt={metrics['throughput']:.3f}")
        else:
            trace_name = source[1]
            trace_path = source[2]
            requests = load_trace_csv(trace_path)
            print(f"\n=== Trace: {trace_name} ({len(requests)} requests) ===")
            for sched_name, action_id in schedulers.items():
                for seed in args.seeds:
                    metrics = run_heuristic(requests, args.num_envs, action_id,
                                            args.max_steps, seed)
                    metrics['source'] = trace_name
                    metrics['scheduler'] = sched_name
                    metrics['seed'] = seed
                    all_results.append(metrics)
                    print(f"  {sched_name}: com={metrics['n_completed']} thpt={metrics['throughput']:.3f}")

            if os.path.exists(args.ppo_model):
                for seed in args.seeds:
                    metrics = run_ppo(requests, args.num_envs, args.ppo_model,
                                      args.max_steps, seed)
                    metrics['source'] = trace_name
                    metrics['scheduler'] = 'PPO'
                    metrics['seed'] = seed
                    all_results.append(metrics)
                    print(f"  PPO: com={metrics['n_completed']} thpt={metrics['throughput']:.3f}")

    csv_path = f'{args.output_dir}/shift_evaluation.csv'
    fieldnames = ['source', 'scheduler', 'seed',
                  'avg_latency', 'p95_latency', 'p99_latency',
                  'avg_ttft', 'p95_ttft', 'p99_ttft',
                  'high_priority_avg_ttft', 'high_priority_p95_ttft',
                  'deadline_miss_rate', 'ttft_slo_miss_rate',
                  'high_priority_completion_rate', 'throughput',
                  'memory_pressure_event_rate', 'oom_event_rate',
                  'starvation_rate',
                  'n_completed', 'n_dropped', 'n_submitted']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_results:
            w.writerow({k: v for k, v in row.items() if k in fieldnames})

    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    main()