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
from experiments.workloads import SCENARIOS


def run_scenario(scenario_name, num_envs, max_steps, action_id, seed):
    from rl_inference_scheduler.workload_batch import BatchWorkload
    env = TensorizedBatchEnv(num_envs=num_envs, scenario=scenario_name,
                             max_steps=max_steps, device='cpu')
    env.reset(seed=seed)
    actions = torch.full((num_envs,), action_id, dtype=torch.long)
    for _ in range(max_steps):
        _, _, done, _ = env.step(actions)
        if done:
            break
    return collect_metrics(env, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces-dir', type=str, default='traces')
    parser.add_argument('--ppo-model', type=str, default='saved_models/ppo_v3_mixed.pt')
    parser.add_argument('--dqn-model', type=str, default=None)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/plots', exist_ok=True)

    trace_files = sorted(glob.glob(f'{args.traces_dir}/*.csv'))
    print(f"Found {len(trace_files)} trace files")

    schedulers = {
        'FIFO': 0,
        'SJF': 2,
        'Priority': 1,
        'LargestSafeBatch': 6,
        'MemAwarePriority': 11,
    }

    all_results = []

    for trace_path in trace_files:
        trace_name = os.path.splitext(os.path.basename(trace_path))[0]
        requests = load_trace_csv(trace_path)
        print(f"\n{trace_name} ({len(requests)} requests)")

        for sched_name, action_id in schedulers.items():
            for seed in args.seeds:
                print(f"  {sched_name} (seed={seed})...", end=' ', flush=True)
                metrics = run_heuristic(requests, args.num_envs, action_id,
                                        args.max_steps, seed)
                metrics['trace'] = trace_name
                metrics['scheduler'] = sched_name
                metrics['seed'] = seed
                all_results.append(metrics)
                print(f"com={metrics['n_completed']} thpt={metrics['throughput']:.3f}")

        if os.path.exists(args.ppo_model):
            for seed in args.seeds:
                print(f"  PPO (seed={seed})...", end=' ', flush=True)
                metrics = run_ppo(requests, args.num_envs, args.ppo_model,
                                  args.max_steps, seed)
                metrics['trace'] = trace_name
                metrics['scheduler'] = 'PPO'
                metrics['seed'] = seed
                all_results.append(metrics)
                print(f"com={metrics['n_completed']} thpt={metrics['throughput']:.3f}")
        else:
            print(f"  PPO: SKIP (model not found at {args.ppo_model})")

    csv_path = f'{args.output_dir}/rl_wins_summary.csv'
    fieldnames = ['trace', 'scheduler', 'seed',
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

    aggregated = {}
    for row in all_results:
        key = (row['trace'], row['scheduler'])
        if key not in aggregated:
            aggregated[key] = {k: [] for k in fieldnames[3:]}
        for k in fieldnames[3:]:
            aggregated[key][k].append(row.get(k, 0))

    print("\n" + "=" * 100)
    print("WHERE RL WINS")
    print("=" * 100)
    for trace_name in sorted(set(r['trace'] for r in all_results)):
        print(f"\n--- {trace_name} ---")
        best = {}
        sch_avgs = {}
        for sched_name in schedulers:
            key = (trace_name, sched_name)
            if key not in aggregated:
                continue
            avgs = {k: sum(v) / len(v) for k, v in aggregated[key].items()}
            sch_avgs[sched_name] = avgs
        key = (trace_name, 'PPO')
        if key in aggregated:
            avgs = {k: sum(v) / len(v) for k, v in aggregated[key].items()}
            sch_avgs['PPO'] = avgs

        for metric in ['p95_ttft', 'deadline_miss_rate', 'throughput',
                       'high_priority_p95_ttft']:
            lower_better = metric != 'throughput' and metric != 'high_priority_completion_rate'
            vals = [(s, d.get(metric, 0)) for s, d in sch_avgs.items()]
            if lower_better:
                best_s, best_v = min(vals, key=lambda x: x[1])
            else:
                best_s, best_v = max(vals, key=lambda x: x[1])
            print(f"  {metric}: best={best_s} ({best_v:.4f})")
            for s, v in vals:
                marker = " <<" if s == best_s else ""
                print(f"    {s:<20}: {v:.4f}{marker}")

    if 'PPO' in [r['scheduler'] for r in all_results]:
        print("\n" + "=" * 100)
        print("WHERE HEURISTICS STILL WIN")
        print("=" * 100)
        for trace_name in sorted(set(r['trace'] for r in all_results)):
            ppo_avg = None
            best_heuristic = None
            key_ppo = (trace_name, 'PPO')
            if key_ppo in aggregated:
                ppo_avg = {k: sum(v) / len(v) for k, v in aggregated[key_ppo].items()}
            for sched_name in schedulers:
                key = (trace_name, sched_name)
                if key not in aggregated:
                    continue
                avg = {k: sum(v) / len(v) for k, v in aggregated[key].items()}
                if ppo_avg and avg.get('p95_latency', 0) < ppo_avg.get('p95_latency', 0):
                    if best_heuristic is None or avg.get('p95_latency', 0) < best_heuristic.get('p95_latency', 0):
                        best_heuristic = avg
                        best_heuristic['name'] = sched_name
            if best_heuristic:
                print(f"\n  {trace_name}: {best_heuristic['name']} has lower p95 latency "
                      f"({best_heuristic.get('p95_latency', 0):.1f} vs "
                      f"{ppo_avg.get('p95_latency', 0):.1f} for PPO)" if ppo_avg else "")


if __name__ == '__main__':
    main()