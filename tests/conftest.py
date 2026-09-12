import pytest

from bus_rl import domain


@pytest.fixture(autouse=True)
def _enable_conservation_checks():
    domain.CONSERVATION_CHECKS = True
    yield
    domain.CONSERVATION_CHECKS = False
