import numpy as np
from typing import List, Dict
from .job import Job


class MetricsTracker:
    def __init__(self):
        self.latencies: List[float] = []
        self.ttfts: List[float] = []
        self.completed_count: int = 0
        self.memory_violations: int = 0
        self.gpu_active_ticks: int = 0
        self.total_ticks: int = 0
        self.dropped_count: int = 0
        self.queue_lengths: List[int] = []

    def compute_summary(self) -> Dict[str, float]:
        return {
            'avg_latency': float(np.mean(self.latencies)) if self.latencies else 0.0,
            'p95_latency': float(np.percentile(self.latencies, 95)) if self.latencies else 0.0,
            'median_latency': float(np.median(self.latencies)) if self.latencies else 0.0,
            'avg_ttft': float(np.mean(self.ttfts)) if self.ttfts else 0.0,
            'throughput': self.completed_count / max(self.total_ticks, 1),
            'memory_violations': self.memory_violations,
            'gpu_utilization': self.gpu_active_ticks / max(self.total_ticks, 1),
            'dropped_jobs': self.dropped_count,
        }


def compute_metrics(jobs_completed: List[Job]) -> Dict[str, float]:
    latencies = [j.latency for j in jobs_completed if j.latency is not None]
    ttfts = [j.ttft for j in jobs_completed if j.ttft is not None]
    return {
        'avg_latency': float(np.mean(latencies)) if latencies else 0.0,
        'p95_latency': float(np.percentile(latencies, 95)) if latencies else 0.0,
        'median_latency': float(np.median(latencies)) if latencies else 0.0,
        'avg_ttft': float(np.mean(ttfts)) if ttfts else 0.0,
    }
