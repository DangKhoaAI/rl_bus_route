from bus_sim.oracle.actions import ACTION_TABLE


def test_action_table_has_221_slots():
    assert len(ACTION_TABLE) == 221
