from dataclasses import dataclass
from typing import Optional


@dataclass
class Job:
    job_id: int
    prompt_len: int
    output_len: int
    priority: int
    memory_mb: float
    compute_units: float
    arrival_time: int
    start_time: Optional[int] = None
    finish_time: Optional[int] = None
    dropped: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def wait_time(self) -> Optional[int]:
        if self.start_time is not None:
            return self.start_time - self.arrival_time
        return None

    @property
    def latency(self) -> Optional[int]:
        if self.finish_time is not None:
            return self.finish_time - self.arrival_time
        return None

    @property
    def ttft(self) -> Optional[int]:
        if self.start_time is not None:
            return self.start_time - self.arrival_time
        return None
