from ..llm_simulator import HardwareAwareServer


class MemoryAwarePriorityScheduler:
    def __init__(self, headroom_ratio: float = 0.25):
        self.headroom_ratio = headroom_ratio

    def select_action(self, server: HardwareAwareServer) -> int:
        if not server.queue:
            return 8

        vram_headroom = 1.0 - server.vram_used_ratio
        if vram_headroom < self.headroom_ratio:
            short_jobs = [j for j in server.queue if j.total_tokens < 300]
            if short_jobs:
                return 2
            return 11

        has_high_prio = any(j.priority <= 2 for j in server.queue)
        if has_high_prio:
            return 1
        return 0
