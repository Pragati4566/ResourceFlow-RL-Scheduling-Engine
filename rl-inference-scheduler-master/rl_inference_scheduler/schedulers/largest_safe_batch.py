from ..llm_simulator import HardwareAwareServer


class LargestSafeBatchScheduler:
    def select_action(self, server: HardwareAwareServer) -> int:
        if not server.queue:
            return 8

        free_slots = server.max_concurrent - len(server.running)
        if free_slots >= 8:
            return 6
        elif free_slots >= 4:
            return 5
        elif free_slots >= 2:
            return 4
        else:
            return 0
