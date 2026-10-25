import random
from typing import List, Dict, Optional, Callable
from .llm_job import LLMJob


BATCH_EFFICIENCY = {1: 1.0, 2: 1.65, 4: 2.80, 8: 4.80}


class HardwareAwareServer:
    def __init__(self, config: dict):
        self.config = config
        self.rng = random.Random(config.get('seed', 42))

        self.vram_total_mb = config.get('vram_total_mb', 16384)
        self.model_base_memory_mb = config.get('model_base_memory_mb', 8000)
        self.kv_mb_per_token = config.get('kv_mb_per_token', 0.25)
        self.max_concurrent = config.get('max_concurrent', 4)
        self.prefill_speed = config.get('prefill_speed', 5.0)
        self.decode_speed = config.get('decode_speed', 10.0)
        self.memory_pressure_threshold = config.get('memory_pressure_threshold', 0.90)
        self.starvation_threshold = config.get('starvation_threshold', 100)
        self.memory_protect_threshold = config.get('memory_protect_threshold', 0.25)

        self.arrival_rate_func: Callable = config.get('arrival_rate_func', lambda t: 0.2)
        self.prompt_len_func: Callable = config.get(
            'prompt_len_func', lambda t, rng: rng.randint(50, 500))
        self.output_len_func: Callable = config.get(
            'output_len_func', lambda t, rng: rng.randint(50, 1000))
        self.priority_func: Callable = config.get(
            'priority_func', lambda t, rng: rng.choices([1, 2, 3, 4, 5], weights=[1, 2, 3, 2, 1])[0])
        self.ttft_slo_mult = config.get('ttft_slo_mult', 5.0)
        self.deadline_mult = config.get('deadline_mult', 15.0)

        self.time = 0
        self.queue: List[LLMJob] = []
        self.running: List[LLMJob] = []
        self.completed: List[LLMJob] = []
        self.dropped: List[LLMJob] = []
        self.next_job_id = 0

        self.just_completed: List[LLMJob] = []
        self.just_completed_prefill: List[LLMJob] = []
        self.dropped_this_tick: List[LLMJob] = []
        self.ttft_slo_misses_this_tick: List[LLMJob] = []
        self.deadline_misses_this_tick: List[LLMJob] = []
        self.oom_events_this_tick: int = 0
        self.tokens_generated_this_tick: int = 0
        self.gpu_idle_ticks: int = 0
        self.memory_pressure_events: int = 0
        self.deadline_miss_count: int = 0
        self.ttft_slo_miss_count: int = 0
        self.oom_count: int = 0

    @property
    def active_memory_mb(self) -> float:
        return self.model_base_memory_mb + sum(j.kv_cache_mb for j in self.running)

    @property
    def vram_used_ratio(self) -> float:
        return self.active_memory_mb / self.vram_total_mb

    @property
    def gpu_active(self) -> bool:
        return len(self.running) > 0

    @property
    def gpu_idle(self) -> bool:
        return len(self.running) == 0

    @property
    def current_batch_size(self) -> int:
        return len(self.running)

    @property
    def efficiency_multiplier(self) -> float:
        size = self.current_batch_size
        if size <= 1:
            return 1.0
        if size <= 2:
            return BATCH_EFFICIENCY[2] / 2
        if size <= 4:
            return BATCH_EFFICIENCY[4] / 4
        return BATCH_EFFICIENCY[8] / 8

    def _generate_job(self) -> LLMJob:
        t = self.time
        prompt_len = self.prompt_len_func(t, self.rng)
        output_len = self.output_len_func(t, self.rng)
        priority = self.priority_func(t, self.rng)
        total_work = prompt_len / self.prefill_speed + output_len / self.decode_speed

        return LLMJob(
            job_id=self.next_job_id,
            prompt_len=prompt_len,
            output_len=output_len,
            priority=priority,
            arrival_time=t,
            ttft_slo_ticks=max(5, int(prompt_len * self.ttft_slo_mult)),
            deadline_ticks=max(10, int(total_work * self.deadline_mult)),
            kv_mb_per_token=self.kv_mb_per_token,
        )

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = random.Random(seed)
            self.config['seed'] = seed
        self.time = 0
        self.queue.clear()
        self.running.clear()
        self.completed.clear()
        self.dropped.clear()
        self.next_job_id = 0
        self.just_completed.clear()
        self.just_completed_prefill.clear()
        self.dropped_this_tick.clear()
        self.ttft_slo_misses_this_tick.clear()
        self.deadline_misses_this_tick.clear()
        self.oom_events_this_tick = 0
        self.tokens_generated_this_tick = 0
        self.gpu_idle_ticks = 0
        self.memory_pressure_events = 0
        self.deadline_miss_count = 0
        self.ttft_slo_miss_count = 0
        self.oom_count = 0

    def step(self, action: int) -> Dict:
        self.just_completed.clear()
        self.just_completed_prefill.clear()
        self.dropped_this_tick.clear()
        self.ttft_slo_misses_this_tick.clear()
        self.deadline_misses_this_tick.clear()
        self.oom_events_this_tick = 0
        self.tokens_generated_this_tick = 0

        if self.rng.random() < self.arrival_rate_func(self.time):
            new_job = self._generate_job()
            self.queue.append(new_job)

        for job in self.running:
            if job.phase == 'prefilling':
                processed = self.prefill_speed * self.efficiency_multiplier
                job.prefill_remaining -= processed
                if job.prefill_remaining <= 0:
                    job.first_token_time = self.time
                    job.phase = 'decoding'
                    self.just_completed_prefill.append(job)

                    if job.ttft_slo_missed:
                        self.ttft_slo_misses_this_tick.append(job)
                        self.ttft_slo_miss_count += 1

            elif job.phase == 'decoding':
                generated = self.decode_speed * self.efficiency_multiplier
                actual = min(generated, job.decode_remaining)
                job.decode_remaining -= actual
                job.generated_tokens += actual
                self.tokens_generated_this_tick += actual

                if job.decode_remaining <= 0:
                    job.completion_time = self.time
                    job.phase = 'completed'
                    self.completed.append(job)
                    self.just_completed.append(job)

                    if job.deadline_missed:
                        self.deadline_misses_this_tick.append(job)
                        self.deadline_miss_count += 1

        self.running = [j for j in self.running if j.phase in ('prefilling', 'decoding')]

        if self.active_memory_mb > self.vram_total_mb:
            self.oom_events_this_tick = 1
            self.oom_count += 1

        if self.vram_used_ratio > self.memory_pressure_threshold:
            self.memory_pressure_events += 1

        self._apply_action(action)

        if self.gpu_idle and len(self.queue) > 0:
            self.gpu_idle_ticks += 1

        self.time += 1

        return self._get_state()

    def _start_job(self, job: LLMJob):
        self.queue.remove(job)
        job.phase = 'prefilling'
        self.running.append(job)

    def _can_run(self, job: LLMJob) -> bool:
        return (self.active_memory_mb + job.kv_cache_mb <= self.vram_total_mb
                and len(self.running) < self.max_concurrent)

    def _dispatch_jobs(self, count: int, sort_key=None):
        if not self.queue:
            return 0
        if sort_key:
            self.queue.sort(key=sort_key)
        dispatched = 0
        for job in list(self.queue):
            if dispatched >= count:
                break
            if self._can_run(job):
                self._start_job(job)
                dispatched += 1
        return dispatched

    def _apply_action(self, action: int):
        if action == 0:
            self._dispatch_jobs(1, lambda j: j.arrival_time)
        elif action == 1:
            self._dispatch_jobs(1, lambda j: (j.priority, j.arrival_time))
        elif action == 2:
            self._dispatch_jobs(1, lambda j: (j.prefill_remaining, j.arrival_time))
        elif action == 3:
            self._dispatch_jobs(1, lambda j: (j.prefill_remaining + j.decode_remaining, j.arrival_time))
        elif action == 4:
            self._dispatch_jobs(2, lambda j: j.arrival_time)
        elif action == 5:
            self._dispatch_jobs(4, lambda j: j.arrival_time)
        elif action == 6:
            self._dispatch_jobs(8, lambda j: j.arrival_time)
        elif action == 7:
            if len(self.running) < self.max_concurrent - 1:
                self._dispatch_jobs(1, lambda j: (j.priority, j.arrival_time))
        elif action == 8:
            pass
        elif action == 9:
            if self.queue:
                self.queue.sort(key=lambda j: (-j.priority, j.arrival_time))
                job = self.queue.pop(0)
                self.queue.append(job)
        elif action == 10:
            expired = [j for j in self.queue
                       if (self.time - j.arrival_time > j.ttft_slo_ticks)
                       and j.priority >= 3 and j.can_drop]
            for job in expired[:2]:
                self.queue.remove(job)
                job.dropped = True
                self.dropped.append(job)
                self.dropped_this_tick.append(job)
        elif action == 11:
            threshold = self.vram_total_mb * self.memory_protect_threshold
            protect = [j for j in self.queue if j.kv_cache_mb <= threshold]
            if protect:
                protect.sort(key=lambda j: j.arrival_time)
                job = protect[0]
                if self._can_run(job):
                    self._start_job(job)

    def oldest_wait(self) -> int:
        return max((self.time - j.arrival_time for j in self.queue), default=0)

    def _get_state(self) -> Dict:
        return {
            'time': self.time,
            'queue': list(self.queue),
            'running': list(self.running),
            'completed': list(self.completed),
            'dropped': list(self.dropped),
            'just_completed': list(self.just_completed),
            'just_completed_prefill': list(self.just_completed_prefill),
            'dropped_this_tick': list(self.dropped_this_tick),
            'ttft_slo_misses_this_tick': list(self.ttft_slo_misses_this_tick),
            'deadline_misses_this_tick': list(self.deadline_misses_this_tick),
            'oom_events_this_tick': self.oom_events_this_tick,
            'tokens_generated_this_tick': self.tokens_generated_this_tick,
            'active_memory_mb': self.active_memory_mb,
            'vram_used_ratio': self.vram_used_ratio,
            'vram_total_mb': self.vram_total_mb,
            'gpu_active': self.gpu_active,
            'gpu_idle': self.gpu_idle,
            'max_concurrent': self.max_concurrent,
            'running_count': len(self.running),
            'queue_length': len(self.queue),
            'active_jobs': len(self.running),
            'oldest_job_wait': self.oldest_wait(),
            'starvation_threshold': self.starvation_threshold,
            'memory_pressure_events': self.memory_pressure_events,
            'ttft_slo_miss_count': self.ttft_slo_miss_count,
            'deadline_miss_count': self.deadline_miss_count,
            'oom_count': self.oom_count,
        }
