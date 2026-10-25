# Adaptive Inference Scheduler
### Reinforcement Learning for Resource-Aware LLM Inference Scheduling

Adaptive Inference Scheduler is a simulation-driven reinforcement learning project that explores how intelligent scheduling policies can improve Large Language Model (LLM) inference serving under variable workloads and hardware constraints.

The simulator processes requests with different priorities, prompt lengths, completion lengths, deadlines, and Time to First Token (TTFT) Service Level Objectives (SLOs). It models prefill and decoding, GPU memory pressure, KV-cache growth, and batch efficiency.

The platform evaluates **Deep Q-Network (DQN) and Proximal Policy Optimization (PPO)** against conventional scheduling strategies, including First-In, First-Out (FIFO), Shortest Job First (SJF), priority-based scheduling, and memory-aware policies.

The main objective is to understand the trade-offs among throughput, latency, deadline adherence, memory safety, and scheduling fairness across different workload conditions.

> **Current finding:** Reinforcement learning does not consistently outperform the strongest heuristic baselines in the current experiments. The project focuses on reproducible benchmarking and identifying the conditions under which adaptive scheduling may be useful.

## Key Features

- **Hardware-aware simulation:** Models prefill and decode phases, request completion, KV-cache memory growth, GPU memory pressure, and batching efficiency.
- **Deep reinforcement learning:** Supports DQN and PPO agents for learning scheduling decisions.
- **Multiple scheduling baselines:** Compares FIFO, SJF, priority-based scheduling, LargestSafeBatch, and MemAwarePriority.
- **Vectorized environments:** Runs multiple simulated environments concurrently using tensor-based state representations.
- **Workload simulation:** Evaluates normal traffic, burst traffic, memory pressure, high-priority requests, and adversarial workload changes.
- **Trace-based evaluation:** Replays deterministic request traces to compare scheduling policies under consistent workloads.
- **Distribution-shift testing:** Examines how policies behave when evaluation workloads differ from training scenarios.
- **Performance benchmarking:** Measures latency, throughput, TTFT SLO violations, deadline misses, high-priority completion, and memory-related failures.
- **Automated testing:** Covers simulator behavior, vectorized-environment equivalence, and trace-replay functionality.
- **Experiment outputs:** Generates machine-readable benchmark results and plots for performance analysis.

## Architecture Overview

The system contains two simulation paths with aligned scheduling behavior.

### Reference Simulation

The reference path models jobs using individual Python objects and processes requests through a hardware-aware server simulation.

It handles:
- Request arrivals and queue management.
- Prefill and decode execution.
- Compute-slot availability.
- KV-cache allocation and memory pressure.
- Scheduling decisions and reward calculation.
- Job completion, deadline violations, and out-of-memory events.

### Vectorized Simulation

The vectorized path represents multiple environments using PyTorch tensors. It is designed to improve training throughput by processing batches of environments together rather than relying on separate Python job objects for every environment.

The training workflow consists of:

1. Generating simulated workloads.
2. Collecting observations from parallel environments.
3. Selecting actions using the scheduling policy.
4. Recording rewards and state transitions.
5. Calculating returns and advantages.
6. Updating the PPO policy.
7. Evaluating learned policies against fixed scheduling baselines.

### Trace-Replay Workflow

Generated request traces can be replayed deterministically through the scheduling environment. This allows different policies to be compared against the same arrival patterns and request characteristics.

## Simulation Model

The environment uses discrete simulation ticks to represent the progress of inference requests.

| Component | Model |
|---|---|
| Time progression | Discrete ticks |
| Request execution | Prefill followed by decoding |
| Compute capacity | Four concurrent execution slots |
| Default prefill speed | 5 tokens per tick |
| Default decode speed | 10 tokens per tick |
| Base model memory | 8 GB |
| KV-cache memory | 0.25 MB per prompt or generated token |
| Observation space | 21-dimensional normalized vector |
| Action space | 12 discrete scheduling actions |

### Prefill and Decoding

During prefill, the simulator processes the input prompt. During decoding, it generates output tokens. Batch efficiency affects the rate at which active requests progress.

The configured per-request batch-efficiency factors are:

| Batch size | Relative efficiency |
|---:|---:|
| 1 | 1.00× |
| 2 | 1.65× |
| 4 | 2.80× |
| 8 | 4.80× |

These are simulator parameters, not measurements of guaranteed real-world GPU performance.

### GPU Memory and KV Cache

Memory consumption is modeled using the base model footprint and the accumulated KV-cache requirements of active requests.

The simulator tracks available VRAM, active requests, and memory pressure. Scheduling actions can protect memory headroom, defer requests, or prevent unsafe batching decisions.

### Request Lifecycle

A typical request follows these stages:

`Arrived → Queued → Prefilling → Decoding → Completed`

The simulator also tracks first-token delivery, TTFT SLO violations, deadline misses, dropped jobs, starvation, and out-of-memory events.

## State and Action Spaces

### Observation Space

The agent receives a normalized, 21-dimensional observation representing scheduling conditions.

The observation includes information about:

- Queue length and active jobs.
- Available compute slots.
- VRAM usage and remaining memory headroom.
- Prompt and output-length characteristics.
- Average and maximum waiting times.
- Request priorities.
- Requests approaching TTFT SLOs or deadlines.
- Recent token generation activity.
- Recent TTFT and deadline violations.
- GPU activity and memory pressure.

The observations provide the agent with information needed to balance immediate throughput against latency, priority, and resource-safety requirements.

### Action Space

The environment provides 12 discrete scheduling actions.

| Action | Description |
|---|---|
| 0 | Dispatch FIFO |
| 1 | Dispatch the highest-priority requests |
| 2 | Dispatch by shortest prefill |
| 3 | Dispatch by shortest total processing time |
| 4 | Use a safe batch size of 2 |
| 5 | Use a safe batch size of 4 |
| 6 | Use a safe batch size of 8 |
| 7 | Reserve capacity for high-priority requests |
| 8 | Wait without dispatching |
| 9 | Defer low-priority requests |
| 10 | Drop expired low-priority requests |
| 11 | Dispatch with memory protection |

This discrete action space allows an agent to select among scheduling and batching strategies rather than directly controlling low-level hardware operations.

## Reinforcement Learning

### Deep Q-Network (DQN)

The initial learning approach uses DQN to estimate the expected value of available scheduling actions.

Its implementation includes:
- An experience replay buffer.
- A target network.
- Epsilon-greedy action selection.
- Reward-driven updates from simulated interactions.

The initial experiments showed that DQN performed similarly to FIFO on the evaluated scenarios. This indicates that the initial training setup did not learn a sufficiently effective policy to surpass the stronger heuristic alternatives.

### Proximal Policy Optimization (PPO)

The PPO agent uses an actor-critic architecture to learn a policy and estimate state values.

The implementation includes:
- A shared two-layer neural network with 128 hidden units per layer.
- Separate policy and value outputs.
- PPO clipped policy updates.
- Generalized Advantage Estimation (GAE).
- A rollout buffer for parallel environment trajectories.
- Minibatch-based training.
- Configurable mixed-workload training.

The documented training configuration uses a discount factor of 0.99, GAE parameter of 0.95, and PPO clipping parameter of 0.2.

### Vectorized Rollout Engine

The vectorized environment stores batched state representations in PyTorch tensors and supports training with multiple concurrent environments.

The benchmark evaluates scaling from 1 to 4,096 environments. Throughput improves as the number of environments increases initially, but per-environment Python processing and other overhead eventually limit efficiency.

The current implementation is most suitable for experimentation with hundreds of parallel environments rather than assuming that the largest batch is always fastest.

## Reward Design

The reward function encourages useful request completion while penalizing latency, resource failures, and poor scheduling decisions.

| Event | Reward or penalty |
|---|---:|
| Completed request | +2 |
| Completed high-priority request | +3 |
| Generated token | +0.05 |
| TTFT SLO violation | -4 |
| Deadline violation | -6 |
| Out-of-memory event | -10 |
| Dropped request | -2 |
| Additional penalty for dropping a high-priority request | -1 |
| Idle compute slot while the queue is non-empty | -0.5 per slot |
| VRAM usage above 90% | Progressive penalty |
| Starvation when the oldest wait exceeds 100 ticks | -3 |

Rewards are clipped to the interval [-10, 10] to support training stability.

The reward structure makes scheduling a multi-objective problem: maximizing request throughput alone is insufficient if the resulting policy violates deadlines or creates memory failures.

## Workload Scenarios

The simulator uses five primary workload configurations.

| Scenario | Arrival pattern | Request characteristics | Main challenge |
|---|---|---|---|
| `normal_load` | Poisson arrivals at 0.2 | Mixed prompts of 50–500 tokens | General scheduling efficiency |
| `burst_load` | Arrival rate increases from 0.05 to 0.5, then decreases to 0.2 | Prompts of 50–400 tokens | Queue spikes and backlog |
| `memory_pressure` | Poisson arrivals at 0.25 | Prompts of 500–2,000 tokens; constrained VRAM | Memory safety and long-running requests |
| `priority_pressure` | Poisson arrivals at 0.2 | 50% high-priority requests | Protecting urgent requests |
| `adversarial_long_short` | Phased arrival pattern | Long jobs arrive before a burst of short, high-priority jobs | Starvation and shifting priorities |

The workload configurations vary arrival patterns, request lengths, priorities, and memory limits so that scheduling strategies can be compared under different operating conditions.

## Evaluation Metrics

The benchmarking pipeline evaluates scheduling policies using several complementary metrics.

- **Average latency:** Average time taken to process requests.
- **P95/P99 latency:** Tail-latency measures that describe slower requests.
- **TTFT SLO miss rate:** Proportion of requests that fail to meet their first-token service target.
- **Deadline miss rate:** Proportion of requests that finish after their deadlines.
- **Throughput:** Completed jobs per simulation tick.
- **High-priority completion rate:** Proportion of high-priority jobs completed.
- **Memory and OOM behavior:** Frequency of memory pressure and out-of-memory events.
- **Starvation:** Whether requests wait excessively long for processing.

Lower latency and violation rates are generally preferred, while throughput and successful completion rates should be higher. Results should be interpreted together because optimizing one metric can worsen another.

## Benchmark Results

### DQN Versus Heuristic Baselines

The initial benchmark compares five scheduling baselines and DQN across five scenarios and five random seeds.

#### Average latency

Lower is better.

| Scenario | FIFO | SJF | Priority | LargestSafeBatch | MemAwarePriority | DQN |
|---|---:|---:|---:|---:|---:|---:|
| Normal load | 255.4 | **162.7** | 205.8 | 255.2 | 205.8 | 254.6 |
| Burst load | 240.1 | **152.9** | 195.4 | 240.0 | 195.4 | 239.2 |
| Memory pressure | 372.4 | **328.5** | 368.0 | 372.3 | 372.4 | 371.6 |
| Priority pressure | 255.4 | **162.7** | 234.0 | 255.2 | 234.0 | 254.6 |
| Adversarial long/short | 421.4 | **277.6** | 401.4 | 421.2 | 394.2 | 420.4 |

SJF achieves the lowest average latency in these experiments.

#### TTFT SLO miss rate

Lower is better.

| Scenario | FIFO | SJF | Priority | LargestSafeBatch | MemAwarePriority | DQN |
|---|---:|---:|---:|---:|---:|---:|
| Normal load | 0.190 | 0.103 | **0.025** | 0.190 | **0.025** | 0.190 |
| Burst load | 0.071 | 0.139 | **0.048** | 0.071 | **0.048** | 0.071 |
| Memory pressure | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Priority pressure | 0.190 | 0.103 | **0.041** | 0.190 | **0.041** | 0.190 |
| Adversarial long/short | **0.000** | 0.419 | 0.183 | **0.000** | 0.183 | **0.000** |

Priority-aware policies achieve lower TTFT violation rates in several scenarios, although performance depends on the workload.

#### Deadline miss rate

Lower is better.

| Scenario | FIFO | SJF | Priority | LargestSafeBatch | MemAwarePriority | DQN |
|---|---:|---:|---:|---:|---:|---:|
| Normal load | 0.150 | **0.034** | 0.046 | 0.150 | 0.046 | 0.150 |
| Burst load | 0.131 | 0.165 | **0.079** | 0.131 | **0.079** | 0.131 |
| Memory pressure | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Priority pressure | 0.150 | **0.034** | 0.036 | 0.150 | 0.036 | 0.150 |
| Adversarial long/short | **0.000** | 0.481 | 0.183 | **0.000** | 0.233 | **0.000** |

#### Throughput

Higher is better.

| Scenario | FIFO | SJF | Priority | LargestSafeBatch | MemAwarePriority | DQN |
|---|---:|---:|---:|---:|---:|---:|
| Normal load | 0.023 | **0.033** | 0.022 | 0.023 | 0.022 | 0.023 |
| Burst load | 0.022 | **0.034** | 0.023 | 0.022 | 0.023 | 0.022 |
| Memory pressure | 0.007 | **0.008** | 0.006 | 0.007 | 0.007 | 0.007 |
| Priority pressure | 0.023 | **0.033** | 0.023 | 0.023 | 0.023 | 0.023 |
| Adversarial long/short | 0.006 | **0.014** | 0.007 | 0.006 | 0.008 | 0.006 |

#### High-priority completion rate

Higher is better.

| Scenario | FIFO | SJF | Priority | LargestSafeBatch | MemAwarePriority | DQN |
|---|---:|---:|---:|---:|---:|---:|
| Normal load | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Burst load | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Memory pressure | 0.750 | **1.000** | 0.750 | 0.750 | 0.750 | 0.750 |
| Priority pressure | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Adversarial long/short | 0.750 | **1.000** | **1.000** | 0.750 | **1.000** | 0.750 |

### Interpretation

The initial evaluation highlights several findings:

1. SJF performs strongly on average latency and throughput under the evaluated workload configurations.
2. Priority-aware policies reduce TTFT violations in several scenarios.
3. SJF can produce significant deadline misses when long jobs arrive before a burst of short, high-priority jobs.
4. No single heuristic is best across every metric and workload.
5. DQN, after the initial training budget, behaves similarly to FIFO and does not consistently beat the best heuristic.

These observations motivate further experimentation with longer training, richer curricula, and more robust evaluation rather than assuming that reinforcement learning automatically improves scheduling.

### Vectorized Environment Scaling

The vectorized benchmark measures simulation throughput on CPU as the environment count increases.

| Environments | Single-environment reference (s) | Vectorized execution (s) | Reported speedup | Environment steps/s |
|---:|---:|---:|---:|---:|
| 1 | 0.000395 | 0.002460 | 0.2× | 406 |
| 4 | 0.000395 | 0.003955 | 0.4× | 1,011 |
| 16 | 0.000395 | 0.010150 | 0.6× | 1,576 |
| 64 | 0.000395 | 0.036953 | 0.7× | 1,732 |
| 256 | 0.000395 | 0.130558 | 0.8× | 1,961 |
| 1,024 | 0.000395 | 0.478172 | 0.8× | 2,141 |
| 4,096 | 0.000395 | 2.403284 | 0.7× | 1,704 |

The reported measurements indicate that throughput improves up to a point, but larger environment batches do not guarantee better efficiency. The Python loop used during action application is an identified bottleneck.

### PPO Versus Heuristics on Trace-Based Workloads

The later benchmark compares PPO with the heuristic baselines using generated request traces and a fixed seed.

#### Throughput

Higher is better.

| Trace | FIFO | SJF | Priority | LargestSafeBatch | MemAwarePriority | PPO |
|---|---:|---:|---:|---:|---:|---:|
| Adversarial shift | 0.144 | **0.156** | 0.144 | 0.144 | 0.144 | 0.148 |
| Burst traffic | 0.788 | 0.788 | 0.788 | 0.788 | 0.788 | 0.788 |
| Interactive chat | 0.114 | 0.114 | 0.114 | 0.114 | 0.114 | 0.114 |
| Long-context RAG | 0.052 | 0.052 | 0.052 | 0.052 | 0.052 | 0.052 |
| Memory saturation | 0.092 | 0.092 | 0.092 | 0.092 | 0.092 | 0.092 |
| Mixed priority | 0.457 | 0.457 | 0.450 | 0.457 | 0.457 | 0.453 |

#### P95 latency

Lower is better.

| Trace | FIFO | SJF | Priority | LargestSafeBatch | MemAwarePriority | PPO |
|---|---:|---:|---:|---:|---:|---:|
| Adversarial shift | 271.1 | 271.1 | 271.1 | 271.1 | 271.1 | 271.1 |
| Burst traffic | 141.7 | 141.7 | 141.7 | 141.7 | 141.7 | 141.7 |
| Interactive chat | 53.9 | 53.9 | 53.9 | 53.9 | 53.9 | 53.9 |
| Long-context RAG | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Memory saturation | 273.7 | 273.7 | 273.7 | 273.7 | 273.7 | 273.7 |
| Mixed priority | 199.8 | 199.8 | 200.2 | 199.8 | 199.8 | 200.0 |

The reported deadline miss rate is zero across the schedulers and traces in this benchmark. That result is specific to these evaluated trace configurations.

#### Key observations

- On the adversarial-shift trace, SJF achieves the highest throughput, while PPO is second.
- On several traces, all schedulers produce identical results, indicating limited differentiation under those simulator conditions.
- The long-context RAG trace completes zero requests across all schedulers because its parameters exceed the simulator's capacity. Its zero latency values should therefore not be interpreted as successful performance.
- PPO currently matches the strongest heuristic on some scenarios but does not consistently surpass it.
- Training on a mixture of generated scenarios does not necessarily transfer to every trace distribution.

The current benchmarks demonstrate why scenario design, metrics, capacity validation, and baseline comparisons are essential when evaluating learned scheduling policies.

## Experiment Workflow

The project supports several independent experiments.

### 1. Compare Scheduling Strategies

Runs the scheduling baselines and compares them across configured workloads.

```bash
python -m experiments.compare_scenarios
```

### 2. Benchmark Vectorized Environments

Measures simulation performance across different parallel-environment counts.

```bash
python -m experiments.benchmark_vectorized_env
```

### 3. Train PPO

Trains the PPO policy with vectorized environments.

```bash
python -m experiments.train_ppo_vectorized \
    --num-envs 256 \
    --epochs 50
```

### 4. Generate Synthetic Traces

```bash
python -m experiments.generate_traces \
    --num-requests 10000
```

### 5. Replay a Trace

```bash
python -m experiments.replay_trace \
    --trace traces/adversarial_shift.csv \
    --all-schedulers
```

### 6. Run the RL Benchmark

```bash
python -m experiments.rl_wins_benchmark
```

### 7. Train with the Mixed Workload Curriculum

```bash
python -m experiments.train_ppo_vectorized \
    --scenario mixed_curriculum \
    --num-envs 64 \
    --epochs 100
```

Training results can vary with random seeds, configuration, dependencies, and compute resources. Benchmark results should be regenerated when the implementation or experiment settings change.

## Installation and Setup

### Requirements

- Python 3.11 or a compatible version.
- pip.
- Dependencies specified in `requirements.txt`.
- A compatible PyTorch installation.

### Install Dependencies

Clone your repository and navigate to the project directory.

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_DIRECTORY
```

Create and activate a virtual environment.

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the automated tests:

```bash
pytest tests/ -v
```

Check the test results before relying on the project benchmarks.

## Project Structure

```text
rl_inference_scheduler/
├── job.py
├── simulator.py
├── environment.py
├── dqn.py
├── trainer.py
├── metrics.py
├── llm_job.py
├── llm_simulator.py
├── llm_env.py
├── llm_trainer.py
├── workload_batch.py
├── vectorized_env.py
├── rollout_buffer.py
├── ppo.py
├── trace_replay.py
├── schedulers/
│   ├── fifo.py
│   ├── sjf.py
│   ├── priority.py
│   ├── largest_safe_batch.py
│   └── memory_aware_priority.py
├── experiments/
│   ├── run_baselines.py
│   ├── compare.py
│   ├── train_rl.py
│   ├── workloads.py
│   ├── compare_scenarios.py
│   ├── train_ppo_vectorized.py
│   ├── benchmark_vectorized_env.py
│   ├── generate_traces.py
│   ├── replay_trace.py
│   ├── rl_wins_benchmark.py
│   └── evaluate_shift.py
├── tests/
│   ├── test_core.py
│   ├── test_vectorized_equivalence.py
│   └── test_trace_replay.py
└── traces/
    ├── interactive_chat.csv
    ├── burst_traffic.csv
    ├── mixed_priority.csv
    ├── adversarial_shift.csv
    ├── long_context_rag.csv
    └── memory_saturation.csv
```

## Testing and Quality Checks

The test suite covers three broad areas:

- **Core simulator tests:** Validate job execution, scheduling behavior, and environment logic.
- **Vectorized equivalence tests:** Compare vectorized-environment behavior with the reference implementation.
- **Trace-replay tests:** Check deterministic workload replay and policy evaluation.

The project also includes a continuous-integration workflow that runs tests on repository updates.

Passing tests helps validate expected behavior, but it does not guarantee that the agent has learned an effective policy or that simulation results will transfer to production inference systems.

## Limitations

The current implementation has several important limitations:

- **RL performance:** DQN does not consistently outperform established heuristics, and PPO's improvements remain limited in the reported trace benchmarks.
- **Simulation fidelity:** The workload and hardware model is simulated; real inference behavior may differ.
- **Trace coverage:** Synthetic traces cannot capture every real-world request distribution.
- **Capacity limitations:** Some long-context workloads exceed the current simulator's configured capacity.
- **CPU execution:** The vectorized environment still has CPU-side bottlenecks.
- **Single-system scope:** The current simulator does not implement multi-GPU scheduling.
- **Generalization:** Training across selected scenarios does not guarantee performance under unseen distributions.

## Future Improvements

Potential improvements include:

- Longer PPO training runs and improved learning-rate schedules.
- More effective curriculum design and exploration strategies.
- Evaluation over additional random seeds.
- Expanded testing for workload shifts and out-of-memory conditions.
- Reducing Python-loop overhead in action processing.
- GPU-accelerated simulation.
- Integration with real inference traces from serving systems.
- Multi-GPU scheduling and resource coordination.
- Stronger statistical comparisons between learned policies and heuristic baselines.
- Improved trace validation to identify infeasible workloads before benchmarking.

## Conclusion

Adaptive Inference Scheduler provides an experimental framework for studying resource-aware scheduling through simulation and reinforcement learning.

By combining hardware-aware request modeling, conventional scheduling strategies, DQN and PPO agents, vectorized training, and trace-based evaluation, it enables systematic analysis of the trade-offs between throughput, latency, service objectives, and memory safety.

Its most important outcome is not the assumption that reinforcement learning always wins, but the ability to measure where learned policies perform well, where conventional heuristics remain stronger, and which limitations should guide the next experiment.
