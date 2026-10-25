from .base import BaseScheduler
from ..simulator import InferenceServer


class FIFOScheduler(BaseScheduler):
    def select_action(self, server: InferenceServer) -> int:
        if not server.queue:
            return 4
        return 0
