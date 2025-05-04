import numpy as np

from jordan_form import (
    all_canonical_jordan_chains,
    canonoical_jordan_chains,
    geig_func,
    group_close_eigvals,
    multiplicity,
)


def test_ordinary():
    A = np.diag([1, 1, 1, 0, 1, 1, 0, 1, 0], k=1)
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
    chains = [
        canonoical_jordan_chains(
            np.stack(
                [A - m.eigval * np.eye(A.shape[0]), -np.eye(A.shape[0])]
                + [np.zeros_like(A) for _ in range(2)],
                axis=0,
            ),
            rtol_norm=1e-3,
            rtol_rank=1e-3,
            atol_norm=1e-3,
            atol_rank=1e-3,
        )
        for m in multiplicities
    ]
    assert [c.shape[0] for c in chains[0]] == [4, 3, 2, 1]
    chains_all = all_canonical_jordan_chains(
        geig_func(A),
        multiplicities,
        rtol_norm=1e-3,
        rtol_rank=1e-3,
        atol_norm=1e-3,
        atol_rank=1e-3,
    )
    print(chains_all)


# chainss = get_canonoicaljordan_chain(
#     np.asarray(
#         [
#             [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
#             [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
#             [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
#         ]
#     )
# )
# print(chainss)
