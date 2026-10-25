from .base import BaseScheduler
from ..simulator import InferenceServer


class PriorityScheduler(BaseScheduler):
    def select_action(self, server: InferenceServer) -> int:
        if not server.queue:
            return 4
        return 1
