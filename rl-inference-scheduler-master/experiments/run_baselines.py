import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rl_inference_scheduler.simulator import InferenceServer
from rl_inference_scheduler.schedulers.fifo import FIFOScheduler
from rl_inference_scheduler.schedulers.sjf import SJFScheduler
from rl_inference_scheduler.schedulers.priority import PriorityScheduler
from rl_inference_scheduler.metrics import MetricsTracker
import json
import argparse


def run_episode(scheduler, seed: int, max_steps: int = 500, arrival_rate: float = 0.2):
    server = InferenceServer(seed=seed, arrival_rate=arrival_rate)
    tracker = MetricsTracker()

    for _ in range(max_steps):
        action = scheduler.select_action(server)
        state = server.step(action)

        tracker.completed_count += len(state['just_completed'])
        for job in state['just_completed']:
            if job.latency is not None:
                tracker.latencies.append(job.latency)
            if job.ttft is not None:
                tracker.ttfts.append(job.ttft)

        tracker.memory_violations = state['memory_violations']
        tracker.dropped_count += len(state['dropped_this_tick'])
        tracker.total_ticks += 1

        if state['gpu_active']:
            tracker.gpu_active_ticks += 1

        tracker.queue_lengths.append(len(state['queue']))

    return tracker.compute_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--arrival-rate', type=float, default=0.2)
    parser.add_argument('--output', type=str, default='results/baseline_results.json')
    args = parser.parse_args()

    schedulers = {
        'FIFO': FIFOScheduler(),
        'SJF': SJFScheduler(),
        'Priority': PriorityScheduler(),
    }

    all_results = {name: [] for name in schedulers}

    for name, scheduler in schedulers.items():
        print(f"Running {name}...")
        for seed in args.seeds:
            result = run_episode(scheduler, seed, args.max_steps, args.arrival_rate)
            all_results[name].append(result)
            print(f"  seed {seed}: avg_latency={result['avg_latency']:.1f}, "
                  f"throughput={result['throughput']:.3f}, "
                  f"gpu_util={result['gpu_utilization']:.2f}")

    summary = {}
    for name, results in all_results.items():
        summary[name] = {}
        for key in results[0]:
            values = [r[key] for r in results]
            summary[name][key] = float(np.mean(values))
            summary[name][f'{key}_std'] = float(np.std(values))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output}")
    for name, metrics in summary.items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            if not k.endswith('_std'):
                print(f"  {k}: {v:.3f}")


if __name__ == '__main__':
    main()
