from abc import ABC, abstractmethod
from typing import Dict
from ..simulator import InferenceServer


class BaseScheduler(ABC):
    @abstractmethod
    def select_action(self, server: InferenceServer) -> int:
        pass
