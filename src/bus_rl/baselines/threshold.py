from bus_rl.control.actions import action_id


class ThresholdController:
    def act(self, observation, mask):
        queues = observation["stops"][:, :, :, 0].sum(axis=(1, 2)) * 40
        route = int(queues.argmax())
        if queues[route] >= 40:
            for bus in range(16):
                choice = action_id("DISPATCH", bus, route)
                if mask[choice]:
                    return choice
        return 0
