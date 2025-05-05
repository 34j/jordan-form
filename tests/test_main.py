import numpy as np
import sympy as sp

from jordan_form import (
    AlgebraicMultiplicity,
    all_canonical_jordan_chains,
    geig_func,
    group_close_eigvals,
    multiplicity,
)


def test_ordinary():
    A = np.diag([1, 1, 1, 0, 1, 1, 0, 1, 0], k=1)
    A = A + np.diag(np.ones(A.shape[0]), k=0)
    eigval, eigvec = np.linalg.eig(A)
    group_close_eigvals(eigval, rtol=1e-3)
    multiplicities = multiplicity(
        eigval,
        eigvec,
        atol_algebraic=1e-3,
        rtol_algebraic=1e-3,
        atol_geometric=1e-3,
        rtol_geometric=1e-3,
    )
    assert multiplicities[0].algebraic_multiplicity == 10
    assert multiplicities[0].geometric_multiplicity == 4
    chains_all = all_canonical_jordan_chains(
        geig_func(A),
        multiplicities[0],
        rtol_norm=1e-3,
        rtol_rank=1e-3,
        atol_norm=1e-3,
        atol_rank=1e-3,
    )
    assert list(chains_all.chain_lengths) == [4, 3, 2, 1]


def test_complicated_ordinary():
    A = np.array(
        [
            [0, 0, 0, 0, -1, -1],
            [0, -8, 4, -3, 1, -3],
            [-3, 13, -8, 6, 2, 9],
            [-2, 14, -7, 4, 2, 10],
            [1, -18, 11, -11, 2, -6],
            [-1, 19, -11, 10, -2, 7],
        ],
        dtype=float,
    )
    eigval, eigvec = np.linalg.eig(A)
    multiplicities = multiplicity(
        eigval,
        eigvec,
        atol_algebraic=1e-3,
        rtol_algebraic=1e-3,
        atol_geometric=1e-3,
        rtol_geometric=1e-3,
    )
    print(multiplicities[0].geometric_multiplicity)
    chains_all = all_canonical_jordan_chains(
        geig_func(A),
        multiplicities[0],
        rtol_norm=1e-3,
        rtol_rank=1e-3,
        atol_norm=1e-3,
        atol_rank=1e-3,
    )
    assert list(chains_all.chain_lengths) == [3, 2]


def test_nonlinear():
    def f(
        eigval: float, derv: int, /
    ) -> np.ndarray[tuple[int, int], np.dtype[np.number]] | None:
        x = sp.symbols("x")
        mat = sp.Matrix([[x**2, 1, 0], [0, x, 0], [0, 0, x]])
        mat_deriv = sp.diff(mat, x, derv)
        return np.array(mat_deriv.subs(x, eigval)).astype(np.float64)

    chains_all = all_canonical_jordan_chains(
        f,
        AlgebraicMultiplicity(eigvals=np.zeros(4)),
        rtol_norm=1e-3,
        rtol_rank=1e-3,
        atol_norm=1e-3,
        atol_rank=1e-3,
    )
    print(chains_all)
