import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rl_inference_scheduler.simulator import InferenceServer
from rl_inference_scheduler.schedulers.fifo import FIFOScheduler
from rl_inference_scheduler.schedulers.sjf import SJFScheduler
from rl_inference_scheduler.schedulers.priority import PriorityScheduler
from rl_inference_scheduler.schedulers.rl import RLScheduler
from rl_inference_scheduler.environment import SchedulingEnvironment
from rl_inference_scheduler.dqn import DQNAgent
from rl_inference_scheduler.metrics import MetricsTracker
import torch
import json


def run_episode_collect_all(scheduler, server, max_steps: int):
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

        tracker.dropped_count += len(state['dropped_this_tick'])
        tracker.total_ticks += 1
        tracker.memory_violations = state['memory_violations']

        if state['gpu_active']:
            tracker.gpu_active_ticks += 1

        tracker.queue_lengths.append(len(state['queue']))

    return tracker


def main():
    sns.set_theme(style='whitegrid')
    max_steps = 500
    num_seeds = 10
    arrival_rate = 0.2
    seeds = [42 + i for i in range(num_seeds)]

    all_results = {}
    all_latencies = {}
    all_ttfts = {}

    schedulers = {
        'FIFO': FIFOScheduler(),
        'SJF': SJFScheduler(),
        'Priority': PriorityScheduler(),
    }

    for name, scheduler in schedulers.items():
        print(f"Evaluating {name}...")
        all_lat = []
        all_ttft = []
        all_res = []

        for seed in seeds:
            server = InferenceServer(seed=seed, arrival_rate=arrival_rate)
            tracker = run_episode_collect_all(scheduler, server, max_steps)
            res = tracker.compute_summary()
            all_res.append(res)
            for job in server.completed:
                if job.latency is not None:
                    all_lat.append(job.latency)
                if job.ttft is not None:
                    all_ttft.append(job.ttft)

        all_results[name] = {k: float(np.mean([r[k] for r in all_res])) for k in all_res[0]}
        all_latencies[name] = all_lat
        all_ttfts[name] = all_ttft

    print("Evaluating DQN Scheduler...")
    if os.path.exists('results/dqn_model.pt'):
        env = SchedulingEnvironment(max_steps=max_steps, arrival_rate=arrival_rate)
        agent = DQNAgent(state_dim=env.observation_dim, action_dim=env.action_space)
        agent.load('results/dqn_model.pt')
        agent.epsilon = 0.0

        all_lat = []
        all_ttft = []
        all_res = []

        for seed in seeds:
            server = InferenceServer(seed=seed, arrival_rate=arrival_rate)
            rl_sched = RLScheduler(agent)
            tracker = run_episode_collect_all(rl_sched, server, max_steps)
            res = tracker.compute_summary()
            all_res.append(res)
            for job in server.completed:
                if job.latency is not None:
                    all_lat.append(job.latency)
                if job.ttft is not None:
                    all_ttft.append(job.ttft)

        all_results['DQN'] = {k: float(np.mean([r[k] for r in all_res])) for k in all_res[0]}
        all_latencies['DQN'] = all_lat
        all_ttfts['DQN'] = all_ttft
    else:
        print("  No trained DQN model found at results/dqn_model.pt, skipping")

    os.makedirs('results/plots', exist_ok=True)
    with open('results/comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print_results(all_results)
    plot_results(all_results, all_latencies, all_ttfts)


def print_results(results: dict):
    print("\n" + "=" * 100)
    header = f"{'Metric':<25}"
    for name in results:
        header += f"{name:<18}"
    print(header)
    print("-" * 100)

    metrics = [
        'avg_latency', 'p95_latency', 'avg_ttft', 'throughput',
        'memory_violations', 'gpu_utilization', 'dropped_jobs'
    ]
    for metric in metrics:
        row = f"{metric:<25}"
        for name in results:
            val = results[name].get(metric, 0)
            if isinstance(val, float):
                row += f"{val:<18.3f}"
            else:
                row += f"{val:<18}"
        print(row)
    print("=" * 100)


def plot_results(results: dict, all_latencies: dict, all_ttfts: dict):
    colors = sns.color_palette('husl', len(results))
    names = list(results.keys())

    plt.figure(figsize=(10, 6))
    for i, name in enumerate(names):
        lats = all_latencies.get(name, [])
        if lats:
            sorted_lats = np.sort(lats)
            cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
            plt.plot(sorted_lats, cdf, label=name, color=colors[i], linewidth=2)
    plt.xlabel('Latency (ticks)', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.title('Latency Distribution', fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('results/plots/latency_cdf.png', dpi=150)
    plt.close()
    print("  Saved latency_cdf.png")

    def bar_chart(metric_name, ylabel, title, filename, fmt='.3f'):
        plt.figure(figsize=(10, 6))
        values = [results[n].get(metric_name, 0) for n in names]
        bars = plt.bar(names, values, color=[colors[i] for i in range(len(names))],
                       edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:{fmt}}', ha='center', va='bottom', fontsize=10)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(f'results/plots/{filename}', dpi=150)
        plt.close()
        print(f"  Saved {filename}")

    bar_chart('throughput', 'Jobs / Tick', 'Throughput Comparison', 'throughput_comparison.png')
    bar_chart('avg_ttft', 'Average TTFT (ticks)', 'Time-to-First-Token Comparison', 'ttft_comparison.png', '.1f')
    bar_chart('memory_violations', 'Violation Count', 'Memory Violations', 'memory_violations.png', '.0f')
    bar_chart('gpu_utilization', 'Utilization', 'GPU Utilization', 'gpu_utilization.png')
    bar_chart('avg_latency', 'Average Latency (ticks)', 'Average Latency Comparison', 'avg_latency_comparison.png', '.1f')
    bar_chart('p95_latency', 'P95 Latency (ticks)', 'P95 Latency Comparison', 'p95_latency_comparison.png', '.1f')


if __name__ == '__main__':
    main()
