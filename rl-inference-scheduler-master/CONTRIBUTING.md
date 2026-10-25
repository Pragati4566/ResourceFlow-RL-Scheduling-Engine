# Contributing

## Setup

```bash
pip install -r requirements.txt
pytest tests/ -v   # all 25 tests should pass
```

## Adding a new scenario

1. Add a scenario function in `experiments/workloads.py` returning a config dict with `arrival_rate_func`, `prompt_len_func`, `output_len_func`, `priority_func`, VRAM params, and timing params.
2. Register it in the `SCENARIOS` dict.
3. It will automatically be available to `compare_scenarios.py`, `TensorizedBatchEnv`, and the mixed curriculum.

## Adding a new scheduler

1. Create a class in `rl_inference_scheduler/schedulers/` implementing `select_action(server) -> int`.
2. Add it to the scheduler dict in `experiments/compare_scenarios.py`.
3. For the vectorized env: map it to one of the 12 existing action IDs, or add a new dispatch method to `TensorizedBatchEnv._apply_actions()`.

## Running benchmarks

```bash
# v0.2 comparison (5 schedulers x 5 scenarios x 5 seeds)
python -m experiments.compare_scenarios

# v0.3 vectorized scaling
python -m experiments.benchmark_vectorized_env

# v0.4 trace replay
python -m experiments.generate_traces --num-requests 10000
python -m experiments.replay_trace --trace traces/adversarial_shift.csv --all-schedulers

# Full RL-wins benchmark (requires trained model)
python -m experiments.rl_wins_benchmark
```

## Testing

- All tests in `tests/`.
- Run `pytest tests/ -v` before submitting a PR.
- New features should include tests, especially for the vectorized env equivalence (match v0.2 single-env output).

## Code style

- No required formatter, but keep code readable.
- Type hints for all function signatures.
- Avoid adding third-party dependencies without discussion.
