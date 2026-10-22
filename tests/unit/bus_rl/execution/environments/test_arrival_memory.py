import numpy as np

from bus_rl.execution.environments.arrival_memory import ArrivalMemory


def test_causal_sparse_slot_history_reset_and_snapshot_isolation():
    memory = ArrivalMemory(3, 10)
    obs = {"arrival_history": np.zeros((2, 4, 2, 8, 5), np.float32)}
    memory.apply(obs, [2, 0], reset=True)
    obs["arrival_history"][0, ..., 0] = 2
    obs["arrival_history"][1, ..., 0] = 1
    first = memory.apply(obs, [2, 0])
    for step in range(2, 13):
        one = {"arrival_history": np.full((1, 4, 2, 8, 5), step, np.float32)}
        latest = memory.apply(one, [0])
    np.testing.assert_array_equal(latest["arrival_memory"][0, 0, 0, 0], np.arange(12, 2, -1))
    np.testing.assert_array_equal(first["arrival_memory"][1, 0, 0, 0], [1, *([0] * 9)])
    reset = memory.apply({"arrival_history": obs["arrival_history"][:1]}, [0], reset=True)
    assert not reset["arrival_memory"].any()
    assert memory.values[2, 0, 0, 0, 0] == 2
    assert not memory.values[1].any()
