from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMJob:
    job_id: int
    prompt_len: int
    output_len: int
    priority: int
    arrival_time: int
    ttft_slo_ticks: int
    deadline_ticks: int
    kv_mb_per_token: float = 0.25
    can_drop: bool = True

    prefill_remaining: int = field(init=False)
    decode_remaining: int = field(init=False)
    generated_tokens: int = 0
    first_token_time: Optional[int] = None
    completion_time: Optional[int] = None
    phase: str = 'queued'
    dropped: bool = False

    def __post_init__(self):
        self.prefill_remaining = self.prompt_len
        self.decode_remaining = self.output_len

    @property
    def kv_cache_mb(self) -> float:
        return (self.prompt_len + self.generated_tokens) * self.kv_mb_per_token

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def ttft(self) -> Optional[int]:
        if self.first_token_time is not None:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def latency(self) -> Optional[int]:
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None

    @property
    def deadline_missed(self) -> bool:
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time > self.deadline_ticks
        return False

    @property
    def ttft_slo_missed(self) -> bool:
        if self.first_token_time is not None:
            return self.first_token_time - self.arrival_time > self.ttft_slo_ticks
        return False
