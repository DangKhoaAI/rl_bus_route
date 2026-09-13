from tests.support.factories import empty_scenario


def test_scenario_hash_ignores_control_stage_flags():
    assert empty_scenario("M1").scenario_hash == empty_scenario("M2").scenario_hash
    assert empty_scenario("M1").scenario_hash == empty_scenario("M3").scenario_hash
