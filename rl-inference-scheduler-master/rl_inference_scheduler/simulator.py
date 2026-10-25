import random
from typing import List, Dict, Optional
from .job import Job


class InferenceServer:
    def __init__(self, seed: int = 42, arrival_rate: float = 0.2):
        self.rng = random.Random(seed)
        self.arrival_rate = arrival_rate
        self.time = 0
        self.vram_total = 16384.0
        self.max_concurrent = 4

        self.queue: List[Job] = []
        self.running: List[Job] = []
        self.completed: List[Job] = []
        self.dropped: List[Job] = []
        self.next_job_id = 0

        self.gpu_idle_ticks = 0
        self.memory_violations = 0
        self.dropped_this_tick: List[Job] = []
        self.just_completed: List[Job] = []

    @property
    def vram_used(self) -> float:
        return sum(j.memory_mb for j in self.running)

    @property
    def vram_free(self) -> float:
        return self.vram_total - self.vram_used

    @property
    def gpu_active(self) -> bool:
        return len(self.running) > 0

    @property
    def gpu_idle(self) -> bool:
        return len(self.running) == 0

    def _generate_job(self) -> Job:
        prompt_len = self.rng.randint(50, 500)
        output_len = self.rng.randint(50, 1000)
        priority = self.rng.choices([1, 2, 3, 4, 5], weights=[1, 2, 3, 2, 1])[0]
        memory_mb = self.rng.uniform(500, 4000)
        compute_units = (prompt_len + output_len) * self.rng.uniform(0.01, 0.03)
        return Job(
            job_id=self.next_job_id,
            prompt_len=prompt_len,
            output_len=output_len,
            priority=priority,
            memory_mb=memory_mb,
            compute_units=compute_units,
            arrival_time=self.time,
        )

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.time = 0
        self.queue.clear()
        self.running.clear()
        self.completed.clear()
        self.dropped.clear()
        self.next_job_id = 0
        self.gpu_idle_ticks = 0
        self.memory_violations = 0

    def step(self, action: int) -> Dict:
        self.just_completed.clear()
        self.dropped_this_tick.clear()

        if self.rng.random() < self.arrival_rate:
            new_job = self._generate_job()
            self.queue.append(new_job)

        for job in self.running:
            job.compute_units -= 1.0
            if job.compute_units <= 0:
                job.finish_time = self.time
                self.completed.append(job)
                self.just_completed.append(job)

        self.running = [j for j in self.running if j.compute_units > 0]

        if self.vram_used > self.vram_total * 0.95:
            self.memory_violations += 1

        self._apply_action(action)

        if self.gpu_idle and len(self.queue) > 0:
            self.gpu_idle_ticks += 1

        self.time += 1

        return self._get_state()

    def _can_run(self, job: Job) -> bool:
        return (self.vram_used + job.memory_mb <= self.vram_total
                and len(self.running) < self.max_concurrent)

    def _take_job(self, job: Job):
        self.queue.remove(job)
        job.start_time = self.time
        self.running.append(job)

    def _apply_action(self, action: int):
        if not self.queue:
            return

        if action == 0:
            self.queue.sort(key=lambda j: j.arrival_time)
            for job in self.queue:
                if self._can_run(job):
                    self._take_job(job)
                    return

        elif action == 1:
            self.queue.sort(key=lambda j: (j.priority, j.arrival_time))
            for job in self.queue:
                if self._can_run(job):
                    self._take_job(job)
                    return

        elif action == 2:
            self.queue.sort(key=lambda j: (j.compute_units, j.arrival_time))
            for job in self.queue:
                if self._can_run(job):
                    self._take_job(job)
                    return

        elif action == 3:
            self.queue.sort(key=lambda j: j.arrival_time)
            taken = 0
            for job in list(self.queue):
                if taken >= 2:
                    break
                if self._can_run(job):
                    self._take_job(job)
                    taken += 1

        elif action == 4:
            pass

        elif action == 5:
            self.queue.sort(key=lambda j: (-j.priority, j.arrival_time))
            job = self.queue.pop(0)
            job.dropped = True
            self.dropped.append(job)
            self.dropped_this_tick.append(job)

    def _get_state(self) -> Dict:
        return {
            'time': self.time,
            'queue': list(self.queue),
            'running': list(self.running),
            'completed': list(self.completed),
            'dropped': list(self.dropped),
            'just_completed': list(self.just_completed),
            'dropped_this_tick': list(self.dropped_this_tick),
            'vram_used': self.vram_used,
            'vram_free': self.vram_free,
            'vram_total': self.vram_total,
            'gpu_idle': self.gpu_idle,
            'gpu_active': self.gpu_active,
            'gpu_idle_ticks': self.gpu_idle_ticks,
            'memory_violations': self.memory_violations,
            'max_concurrent': self.max_concurrent,
            'running_count': len(self.running),
            'queue_length': len(self.queue),
        }
