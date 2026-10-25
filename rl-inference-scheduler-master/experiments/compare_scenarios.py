import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
import argparse
from collections import defaultdict

from rl_inference_scheduler.llm_simulator import HardwareAwareServer
from rl_inference_scheduler.llm_env import V2Environment
from rl_inference_scheduler.dqn import DQNAgent
from rl_inference_scheduler.schedulers.fifo import FIFOScheduler
from rl_inference_scheduler.schedulers.sjf import SJFScheduler
from rl_inference_scheduler.schedulers.priority import PriorityScheduler
from rl_inference_scheduler.schedulers.largest_safe_batch import LargestSafeBatchScheduler
from rl_inference_scheduler.schedulers.memory_aware_priority import MemoryAwarePriorityScheduler
from rl_inference_scheduler.schedulers.rl import RLScheduler
from rl_inference_scheduler.metrics import MetricsTracker
from experiments.workloads import get_scenario, SCENARIOS
import torch


V2_SCHEDULER_ACTIONS = {
    'FIFO': lambda s: 0 if s.queue else 8,
    'SJF': lambda s: 2 if s.queue else 8,
    'Priority': lambda s: 1 if s.queue else 8,
}


def run_episode(scheduler_fn, config: dict, max_steps: int):
    server = HardwareAwareServer(config)
    tracker = MetricsTracker()

    for _ in range(max_steps):
        action = scheduler_fn(server)
        state = server.step(action)

        tracker.completed_count += len(state['just_completed'])
        for job in state['just_completed']:
            if job.latency is not None:
                tracker.latencies.append(job.latency)
            if job.ttft is not None:
                tracker.ttfts.append(job.ttft)

        tracker.dropped_count += len(state['dropped_this_tick'])
        tracker.total_ticks += 1
        tracker.memory_violations = state.get('oom_count', 0)

        if state['gpu_active']:
            tracker.gpu_active_ticks += 1

        tracker.queue_lengths.append(len(state['queue']))

    completed_jobs = server.completed
    ttft_slo_misses = sum(1 for j in completed_jobs if j.ttft_slo_missed)
    deadline_misses = sum(1 for j in completed_jobs if j.deadline_missed)
    high_prio_completed = sum(1 for j in completed_jobs if j.priority <= 2)

    summary = tracker.compute_summary()
    summary['ttft_slo_miss_rate'] = ttft_slo_misses / max(len(completed_jobs), 1)
    summary['deadline_miss_rate'] = deadline_misses / max(len(completed_jobs), 1)
    summary['high_prio_completion_rate'] = high_prio_completed / max(
        sum(1 for j in server.dropped + completed_jobs if j.priority <= 2), 1)
    summary['oom_count'] = server.oom_count
    summary['memory_pressure_events'] = server.memory_pressure_events
    summary['p95_latency'] = float(np.percentile(tracker.latencies, 95)) if tracker.latencies else 0

    return summary


def run_rl_episode(agent, config: dict, max_steps: int):
    env = V2Environment(config, max_steps=max_steps)
    state = env.reset(seed=config['seed'])
    tracker = MetricsTracker()
    total_reward = 0

    for _ in range(max_steps):
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, done, info = env.step(action)

        state = next_state
        total_reward += reward

        server = env.server
        tracker.completed_count += len(server.just_completed)
        for job in server.just_completed:
            if job.latency is not None:
                tracker.latencies.append(job.latency)
            if job.ttft is not None:
                tracker.ttfts.append(job.ttft)

        tracker.dropped_count += len(server.dropped_this_tick)
        tracker.total_ticks += 1
        tracker.memory_violations = server.oom_count

        if server.gpu_active:
            tracker.gpu_active_ticks += 1

        tracker.queue_lengths.append(len(server.queue))

        if done:
            break

    completed_jobs = env.server.completed
    ttft_slo_misses = sum(1 for j in completed_jobs if j.ttft_slo_missed)
    deadline_misses = sum(1 for j in completed_jobs if j.deadline_missed)
    high_prio_completed = sum(1 for j in completed_jobs if j.priority <= 2)

    summary = tracker.compute_summary()
    summary['ttft_slo_miss_rate'] = ttft_slo_misses / max(len(completed_jobs), 1)
    summary['deadline_miss_rate'] = deadline_misses / max(len(completed_jobs), 1)
    summary['high_prio_completion_rate'] = high_prio_completed / max(
        sum(1 for j in env.server.dropped + completed_jobs if j.priority <= 2), 1)
    summary['oom_count'] = env.server.oom_count
    summary['memory_pressure_events'] = env.server.memory_pressure_events
    summary['p95_latency'] = float(np.percentile(tracker.latencies, 95)) if tracker.latencies else 0
    summary['rl_total_reward'] = total_reward

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--rl-model', type=str, default='results/dqn_v2_model.pt')
    parser.add_argument('--scenarios', type=str, nargs='+',
                        default=list(SCENARIOS.keys()))
    args = parser.parse_args()

    sns.set_theme(style='whitegrid')
    os.makedirs('results/plots', exist_ok=True)

    schedulers = {
        'FIFO': lambda s: 0 if s.queue else 8,
        'SJF': lambda s: 2 if s.queue else 8,
        'Priority': lambda s: 1 if s.queue else 8,
        'LargestSafeBatch': LargestSafeBatchScheduler().select_action,
        'MemAwarePriority': MemoryAwarePriorityScheduler().select_action,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rl_agent = None
    if os.path.exists(args.rl_model):
        env = V2Environment(get_scenario('normal_load'), max_steps=args.max_steps)
        rl_agent = DQNAgent(state_dim=env.observation_dim, action_dim=env.action_space)
        rl_agent.load(args.rl_model)
        rl_agent.epsilon = 0.0
        print(f"Loaded RL model from {args.rl_model}")
    else:
        print(f"No RL model found at {args.rl_model}, skipping DQN evaluation")

    all_results = defaultdict(lambda: defaultdict(list))

    for scenario_name in args.scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*60}")

        config_template = get_scenario(scenario_name)

        for name, scheduler_fn in schedulers.items():
            print(f"  Running {name}...")
            for seed in args.seeds:
                config = get_scenario(scenario_name, seed=seed)
                summary = run_episode(scheduler_fn, config, args.max_steps)
                all_results[scenario_name][name].append(summary)

        if rl_agent is not None:
            print(f"  Running DQN...")
            for seed in args.seeds:
                config = get_scenario(scenario_name, seed=seed)
                summary = run_rl_episode(rl_agent, config, args.max_steps)
                all_results[scenario_name]['DQN'].append(summary)

    aggregated = defaultdict(lambda: defaultdict(dict))
    for scenario_name, sched_results in all_results.items():
        for sched_name, results_list in sched_results.items():
            mean = {}
            for key in results_list[0]:
                values = [r[key] for r in results_list]
                mean[key] = float(np.mean(values))
            aggregated[scenario_name][sched_name] = mean

    os.makedirs('results', exist_ok=True)
    with open('results/scenario_metrics.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    with open('results/scenario_metrics.csv', 'w', newline='') as f:
        fieldnames = ['scenario', 'scheduler', 'avg_latency', 'p95_latency',
                      'avg_ttft', 'throughput', 'ttft_slo_miss_rate',
                      'deadline_miss_rate', 'high_prio_completion_rate',
                      'oom_count', 'memory_pressure_events', 'gpu_utilization',
                      'dropped_jobs']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario_name, sched_results in aggregated.items():
            for sched_name, metrics in sched_results.items():
                row = {'scenario': scenario_name, 'scheduler': sched_name}
                for key in fieldnames[2:]:
                    row[key] = metrics.get(key, 0)
                writer.writerow(row)

    print_results(aggregated)
    plot_scenario_comparison(aggregated)


def print_results(aggregated: dict):
    print("\n" + "=" * 120)
    scenarios = list(aggregated.keys())
    schedulers_list = None
    for s in aggregated.values():
        schedulers_list = list(s.keys())
        break

    for metric in ['avg_latency', 'p95_latency', 'avg_ttft', 'throughput',
                   'ttft_slo_miss_rate', 'deadline_miss_rate',
                   'high_prio_completion_rate', 'oom_count']:
        print(f"\n--- {metric} ---")
        header = f"{'Scenario':<30}"
        for s in schedulers_list or []:
            header += f"{s:<18}"
        print(header)
        print("-" * 120)
        for scenario_name in scenarios:
            row = f"{scenario_name:<30}"
            for s in schedulers_list or []:
                val = aggregated[scenario_name].get(s, {}).get(metric, 0)
                if isinstance(val, float):
                    row += f"{val:<18.4f}"
                else:
                    row += f"{val:<18}"
            print(row)


def plot_scenario_comparison(aggregated: dict):
    scenarios = list(aggregated.keys())
    schedulers_list = None
    for s in aggregated.values():
        schedulers_list = list(s.keys())
        break
    if not schedulers_list:
        return

    colors = sns.color_palette('husl', len(schedulers_list))
    n = len(scenarios)

    metric_configs = [
        ('p95_latency', 'P95 Latency (ticks)', 'P95 Latency by Scenario'),
        ('avg_ttft', 'Avg TTFT (ticks)', 'Average TTFT by Scenario'),
        ('throughput', 'Jobs / Tick', 'Throughput by Scenario'),
        ('deadline_miss_rate', 'Rate', 'Deadline Miss Rate by Scenario'),
        ('high_prio_completion_rate', 'Rate', 'High-Priority Completion Rate by Scenario'),
        ('memory_pressure_events', 'Count', 'Memory Pressure Events by Scenario'),
    ]

    for metric, ylabel, title in metric_configs:
        fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 6))
        x = np.arange(n)
        width = 0.8 / len(schedulers_list)

        for i, sched in enumerate(schedulers_list):
            values = [aggregated[s].get(sched, {}).get(metric, 0) for s in scenarios]
            bars = ax.bar(x + i * width - 0.4 + width / 2, values, width * 0.9,
                          label=sched, color=colors[i], edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=8, loc='best')

        filename = metric.replace('_', '_') + '_by_scenario.png'
        plt.tight_layout()
        plt.savefig(f'results/plots/{filename}', dpi=150)
        plt.close()
        print(f"  Saved results/plots/{filename}")


if __name__ == '__main__':
    main()
