import pytest

from bus_rl import domain


@pytest.fixture(autouse=True, scope="session")
def _small_torch_threads():
    """Tests are not benchmarks: 2 threads is much faster here (see R4 tuning)."""
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _enable_conservation_checks():
    domain.CONSERVATION_CHECKS = True
    yield
    domain.CONSERVATION_CHECKS = False
