from bus_rl.control.actions import action_id


class ProportionalController:
    """Allocate the next reserve to the route with the largest observed queue."""

    def act(self, observation, mask):
        route = int(observation["stops"][:, :, :, 0].sum(axis=(1, 2)).argmax())
        for bus_id in range(16):
            candidate = action_id("DISPATCH", bus_id, route)
            if mask[candidate]:
                return candidate
        return 0
