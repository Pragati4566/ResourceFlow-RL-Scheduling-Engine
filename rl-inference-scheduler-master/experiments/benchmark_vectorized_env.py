import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rl_inference_scheduler.vectorized_env import TensorizedBatchEnv
from rl_inference_scheduler.llm_env import V2Environment
from experiments.workloads import get_scenario


def benchmark_single_env(config, num_steps=100):
    env = V2Environment(config, max_steps=num_steps + 1)
    obs = env.reset()

    start = time.perf_counter()
    for _ in range(num_steps):
        action = np.random.randint(0, 12)
        obs, reward, done, info = env.step(action)
    elapsed = time.perf_counter() - start
    return elapsed / num_steps


def benchmark_vectorized(num_envs, num_steps=100, device='cpu'):
    env = TensorizedBatchEnv(num_envs=num_envs, scenario='normal_load',
                             max_steps=num_steps + 1, device=device)
    obs = env.reset()

    start = time.perf_counter()
    for _ in range(num_steps):
        actions = torch.randint(0, 12, (num_envs,))
        obs, reward, done, info = env.step(actions)
    elapsed = time.perf_counter() - start
    return elapsed / num_steps


def main():
    env_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    num_steps = 100 if torch.cuda.is_available() else 50

    config = get_scenario('normal_load')
    single_time = benchmark_single_env(config, num_steps)

    results = []
    print(f"{'Num envs':>10} | {'Single (s)':>10} | {'Vector (s)':>10} | {'Speedup':>8} | {'Env-steps/s':>12}")
    print("-" * 60)

    for n in env_sizes:
        vec_time = benchmark_vectorized(n, num_steps)

        env_steps_per_sec = n / vec_time
        speedup = single_time / (vec_time / n) if vec_time > 0 else 0

        results.append({
            'num_envs': n,
            'single_time_per_step': single_time,
            'vectorized_time_per_step': vec_time,
            'speedup': speedup,
            'env_steps_per_sec': env_steps_per_sec,
        })

        print(f"{n:>10} | {single_time:>10.6f} | {vec_time:>10.6f} | {speedup:>8.1f}x | {env_steps_per_sec:>12.0f}")

    os.makedirs('benchmarks', exist_ok=True)
    import csv
    with open('benchmarks/vectorized_scaling.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['num_envs', 'single_time_per_step',
                                          'vectorized_time_per_step', 'speedup',
                                          'env_steps_per_sec'])
        w.writeheader()
        w.writerows(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog([r['num_envs'] for r in results],
               [r['env_steps_per_sec'] for r in results],
               marker='o', label='Vectorized')
    base_single = results[0]['single_time_per_step']
    ax1.loglog([r['num_envs'] for r in results],
               [r['num_envs'] / base_single for r in results],
               '--', label='Single (ideal linear)')
    ax1.set_xlabel('Number of parallel envs')
    ax1.set_ylabel('Env-steps / second')
    ax1.set_title('Scaling: Vectorized vs Single')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    ax2.semilogx([r['num_envs'] for r in results],
                 [r['speedup'] for r in results],
                 marker='s', color='green')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of parallel envs')
    ax2.set_ylabel('Speedup vs single sequential')
    ax2.set_title('Vectorization speedup')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmarks/vectorized_scaling.png', dpi=150)
    plt.close()
    print(f"\nResults saved to benchmarks/vectorized_scaling.csv")
    print(f"Plot saved to benchmarks/vectorized_scaling.png")


if __name__ == '__main__':
    main()
