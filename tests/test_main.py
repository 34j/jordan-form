import matplotlib.pyplot as plt
import numpy as np
import pytest
import sympy as sp

from jordan_form import (
    canonical_jordan_chains,
    geig_func,
    group_close_eigvals,
    multiplicity,
    sympy_func,
)
from jordan_form.plot import plot_eigval_with_multiplicity


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
    chains = canonical_jordan_chains(
        geig_func(A),
        multiplicities[0].eigval,
        rtol_norm=1e-3,
        rtol_rank=1e-3,
        atol_norm=1e-3,
        atol_rank=1e-3,
        algebraic_multiplicity=multiplicities[0].algebraic_multiplicity,
    )
    assert list(chains.chain_lengths) == [4, 3, 2, 1]


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
    assert multiplicities[0].geometric_multiplicity == 2
    chains = canonical_jordan_chains(
        geig_func(A),
        multiplicities[0].eigval,
        rtol_norm=1e-3,
        rtol_rank=1e-3,
        atol_norm=1e-3,
        atol_rank=1e-3,
        algebraic_multiplicity=multiplicities[0].algebraic_multiplicity,
    )
    assert list(chains.chain_lengths) == [3, 2]


@pytest.mark.parametrize("pass_eigvec", [True, False])
def test_plot(pass_eigvec: bool) -> None:
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
        eigvec if pass_eigvec else None,
        atol_algebraic=1e-3,
        rtol_algebraic=1e-3,
        atol_geometric=1e-3,
        rtol_geometric=1e-3,
    )
    fig, ax = plt.subplots()
    plot_eigval_with_multiplicity(
        multiplicities,
        ax=ax,
        text_filter=lambda m: m.algebraic_multiplicity > 1,
        text_add_clue=True,
    )
    fig.savefig(f"tests/test_complicated_ordinary_{pass_eigvec}.png", dpi=300)
    plt.close(fig)


@pytest.mark.parametrize("wrong_eigval", [True, False])
def test_nonlinear(wrong_eigval: bool) -> None:
    x = sp.symbols("x")
    mat = sp.Matrix([[x**2, 1, 0], [0, x, 0], [0, 0, x]])
    if wrong_eigval:
        multiplicity(
            np.empty(0),
            np.empty((3, 0)),
            atol_algebraic=1e-3,
            rtol_algebraic=1e-3,
            atol_geometric=1e-3,
            rtol_geometric=1e-3,
        )
    chains = canonical_jordan_chains(
        sympy_func(mat),
        1 if wrong_eigval else 0,
        rtol_norm=1e-3,
        rtol_rank=1e-3,
        atol_norm=1e-3,
        atol_rank=1e-3,
        algebraic_multiplicity=0 if wrong_eigval else 4,
    )
    assert list(chains.chain_lengths) == [] if wrong_eigval else [3, 1]
