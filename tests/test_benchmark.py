from __future__ import annotations

import numpy as np
import pytest

from messops import IntegralContext


@pytest.fixture
def h2o():
    z = np.array([1, 1, 8])
    pos = np.array(
        [
            [0.0, -0.757, 0.587],
            [0.0, 0.757, 0.587],
            [0.0, 0.0, 0.0],
        ]
    )
    return (z, pos)


# basis_name_cases = ["6-31g", "cc-pvdz", "def2-dzvppd"]
basis_name_cases = ["6-31g", "cc-pvdz"]


@pytest.mark.parametrize("basis_name", basis_name_cases)
def test_one_body_integrals(benchmark, h2o, basis_name):
    z, pos = h2o

    def compute():
        ctx = IntegralContext(z, pos, basis_name)
        S = ctx.overlap()
        T = ctx.kinetic()
        V = ctx.nuclear()
        return S, T, V

    S, T, V = benchmark(compute)

    assert np.allclose(S, S.T)
    assert np.allclose(T, T.T)
    assert np.allclose(V, V.T)
