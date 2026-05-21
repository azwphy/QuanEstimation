import pytest
from quanestimation import ComprehensiveOpt


def test_comprehensiveopt_factory_methods():
    """Verify all supported factory methods exist."""
    for method in ["AD", "DE", "PSO"]:
        copt = ComprehensiveOpt(
            savefile=False,
            method=method,
            max_episode=5,
            seed=1234,
        )
        assert copt is not None


def test_comprehensiveopt_invalid_method():
    """Verify factory raises on unsupported method."""
    with pytest.raises(ValueError, match="is not a valid value for method"):
        ComprehensiveOpt(savefile=False, method="INVALID")
