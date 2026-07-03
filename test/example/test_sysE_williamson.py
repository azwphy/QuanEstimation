import numpy as np
from quanestimation.base.AsymptoticBound.CramerRao import (
    QFIM_Gauss,
    Williamson_form,
)


def test_sysE_williamson() -> None:
    _, c3 = Williamson_form(3.0 * np.eye(2))
    assert len(c3) == 1
    assert np.isclose(c3[0], 3.0, rtol=1e-10)

    _, c5 = Williamson_form(5.0 * np.eye(2))
    assert len(c5) == 1
    assert np.isclose(c5[0], 5.0, rtol=1e-10)

    nu1, nu2 = 3.0, 5.0
    R1 = nu1 * np.eye(2)
    R2 = nu2 * np.eye(2)
    dR1 = [np.array([[1.0, 0.0], [0.0, 1.0]])]
    dR2 = [np.array([[1.0, 0.0], [0.0, 1.0]])]
    mu = np.array([0.0, 0.0])
    dmu = [np.array([0.0, 0.0])]

    F1 = QFIM_Gauss(mu, dmu, R1, dR1)
    F2 = QFIM_Gauss(mu, dmu, R2, dR2)
    assert np.allclose(F1, 1.0 / (2.0 * (nu1**2 - 0.25)))
    assert np.allclose(F2, 1.0 / (2.0 * (nu2**2 - 0.25)))
