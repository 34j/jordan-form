import numpy as np

from jordan_form._main import get_canonoical_jordan_chain, get_multiplicity


def test_ordinary():
    A = np.diag([1, 1, 1, 0, 1, 1, 0, 1], k=1)
    eigval, eigvec = np.linalg.eig(A)
    multiplicities = get_multiplicity(
        eigval, eigvec, atol_algebraic=1e-3, rtol_geometric=1e-3
    )
    assert multiplicities[0].algebraic_multiplicity == 9
    assert multiplicities[0].geometric_multiplicity == 3
    for m in multiplicities:
        chain = get_canonoical_jordan_chain(
            np.stack(
                [A - m.eigval * np.eye(A.shape[0]), -np.eye(A.shape[0])]
                + [np.zeros_like(A) for _ in range(2)],
                axis=0,
            )
        )
        print(chain)


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
