import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import random


def make_writer(path):
    f = open(path, 'w', newline='')
    w = csv.writer(f)
    w.writerow(['arrival_time', 'prompt_len', 'output_len', 'priority',
                'deadline_ticks', 'ttft_slo_ticks', 'user_type'])
    return f, w


def generate_interactive_chat(num_requests, path):
    rng = random.Random(seed)
    f, w = make_writer(path)
    t = 0
    for _ in range(num_requests):
        t += 1 + int(rng.expovariate(1 / 7))
        prompt = rng.randint(50, 200)
        output = rng.randint(50, 200)
        prio = rng.choices([1, 2, 3, 4, 5], weights=[1, 2, 3, 2, 1])[0]
        total = prompt / 5 + output / 10
        ttft_slo = max(5, int(prompt * 2))
        deadline = max(10, int(total * 5))
        w.writerow([t, prompt, output, prio, deadline, ttft_slo, 'interactive'])
    f.close()
    print(f"Generated {path} with {num_requests} requests")


def generate_long_context_rag(num_requests, path):
    rng = random.Random(seed + 1)
    f, w = make_writer(path)
    t = 0
    for _ in range(num_requests):
        t += 1 + int(rng.expovariate(1 / 0.12))
        prompt = rng.randint(1000, 4000)
        output = rng.randint(50, 200)
        prio = rng.choices([1, 2, 3, 4, 5], weights=[1, 1, 2, 3, 2])[0]
        total = prompt / 5 + output / 10
        ttft_slo = max(5, int(prompt * 1.5))
        deadline = max(10, int(total * 4))
        w.writerow([t, prompt, output, prio, deadline, ttft_slo, 'rag'])
    f.close()
    print(f"Generated {path} with {num_requests} requests")


def generate_burst_traffic(num_requests, path):
    rng = random.Random(seed + 2)
    f, w = make_writer(path)
    t = 0
    for i in range(num_requests):
        if t < 100:
            rate = 0.03
        elif t < 200:
            rate = 0.6
        else:
            rate = 0.2
        t += 1 + int(rng.expovariate(1 / rate))
        prompt = rng.randint(50, 400)
        output = rng.randint(50, 500)
        prio = rng.choices([1, 2, 3, 4, 5], weights=[1, 2, 3, 2, 1])[0]
        total = prompt / 5 + output / 10
        ttft_slo = max(5, int(prompt * 2))
        deadline = max(10, int(total * 4))
        w.writerow([t, prompt, output, prio, deadline, ttft_slo, 'burst'])
    f.close()
    print(f"Generated {path} with {num_requests} requests")


def generate_mixed_priority(num_requests, path):
    rng = random.Random(seed + 3)
    f, w = make_writer(path)
    t = 0
    for _ in range(num_requests):
        t += 1 + int(rng.expovariate(1 / 0.2))
        prompt = rng.randint(50, 500)
        output = rng.randint(50, 1000)
        prio = rng.choices([1, 2, 3, 4, 5], weights=[8, 4, 2, 1, 1])[0]
        total = prompt / 5 + output / 10
        ttft_slo = max(5, int(prompt * 2))
        deadline = max(10, int(total * 5))
        w.writerow([t, prompt, output, prio, deadline, ttft_slo, 'mixed_priority'])
    f.close()
    print(f"Generated {path} with {num_requests} requests")


def generate_adversarial_shift(num_requests, path):
    rng = random.Random(seed + 4)
    f, w = make_writer(path)
    t = 0
    phase1_end = int(num_requests * 0.4)
    for i in range(num_requests):
        t += 1 + int(rng.expovariate(1 / 0.12))
        if i < phase1_end:
            prompt = rng.randint(800, 2000)
            output = rng.randint(200, 800)
            prio = rng.choices([1, 2, 3, 4, 5], weights=[1, 1, 3, 2, 1])[0]
        else:
            prompt = rng.randint(50, 200)
            output = rng.randint(50, 200)
            prio = rng.choices([1, 2, 3, 4, 5], weights=[1, 1, 1, 3, 3])[0]
        total = prompt / 5 + output / 10
        ttft_slo = max(5, int(prompt * 1.5))
        deadline = max(10, int(total * 3))
        w.writerow([t, prompt, output, prio, deadline, ttft_slo, 'adversarial'])
    f.close()
    print(f"Generated {path} with {num_requests} requests")


def generate_memory_saturation(num_requests, path):
    rng = random.Random(seed + 5)
    f, w = make_writer(path)
    t = 0
    for _ in range(num_requests):
        t += 1 + int(rng.expovariate(1 / 0.18))
        prompt = rng.randint(800, 3000)
        output = rng.randint(100, 500)
        prio = rng.choices([1, 2, 3, 4, 5], weights=[1, 2, 3, 2, 1])[0]
        total = prompt / 5 + output / 10
        ttft_slo = max(5, int(prompt * 2))
        deadline = max(10, int(total * 5))
        w.writerow([t, prompt, output, prio, deadline, ttft_slo, 'memory_saturation'])
    f.close()
    print(f"Generated {path} with {num_requests} requests")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-requests', type=int, default=10000)
    parser.add_argument('--output-dir', type=str, default='traces')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)

    global seed
    seed = 42
    generate_interactive_chat(args.num_requests, 'interactive_chat.csv')
    generate_long_context_rag(args.num_requests, 'long_context_rag.csv')
    generate_burst_traffic(args.num_requests, 'burst_traffic.csv')
    generate_mixed_priority(args.num_requests, 'mixed_priority.csv')
    generate_adversarial_shift(args.num_requests, 'adversarial_shift.csv')
    generate_memory_saturation(args.num_requests, 'memory_saturation.csv')


if __name__ == '__main__':
    main()