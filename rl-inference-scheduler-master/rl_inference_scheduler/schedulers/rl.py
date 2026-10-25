from .base import BaseScheduler
from ..simulator import InferenceServer
from ..dqn import DQNAgent
from ..environment import make_observation


class RLScheduler(BaseScheduler):
    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def select_action(self, server: InferenceServer) -> int:
        obs = make_observation(server)
        return self.agent.select_action(obs, eval_mode=True)
