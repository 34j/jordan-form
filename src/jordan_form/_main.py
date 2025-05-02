import warnings
from collections.abc import Sequence
from typing import Any

import attrs
import numpy as np


@attrs.frozen(kw_only=True)
class AlgebraicMultiplicity:
    eigvals: np.ndarray[tuple[int], np.dtype[np.number]]
    """The eigenvalues of shape (algebraic_multiplicity,)."""

    @property
    def eigval(self) -> float:
        """The mean eigenvalue."""
        return np.mean(self.eigvals, axis=0)

    @property
    def algebraic_multiplicity(self) -> int:
        """
        The algebraic multiplicity of the eigenvalue.

        The number of times the eigenvalue appears
        as a root of the characteristic polynomial.

        """
        return self.eigvals.shape[0]


@attrs.frozen(kw_only=True)
class Multiplicity(AlgebraicMultiplicity):
    eigvecs: np.ndarray[tuple[int, int], np.dtype[np.number]]
    """The eigenvectors of shape (n, algebraic_multiplicity)."""
    eigvec_orthogonal: np.ndarray[tuple[int, int], np.dtype[np.number]]
    """The orthogonal eigenvectors of shape (n, geometric_multiplicity)."""

    @property
    def geometric_multiplicity(self) -> int:
        """
        The geometric multiplicity of the eigenvalue.

        The dimension of the eigenspace of the eigenvalue.
        Less than or equal to the algebraic multiplicity.

        """
        return self.eigvec_orthogonal.shape[1]


def group_close_eigval(
    eigval: Sequence[float],
    /,
    *,
    atol: float | None = None,
) -> set[set[int]]:
    """
    Group the eigenvalues that are close to each other.

    Parameters
    ----------
    eigval : Sequence[float]
        The eigenvalues.
    atol : float | None, optional
        The threshold to treat eigenvalues as the same.

    Returns
    -------
    set[set[int]]
        The indices of the eigenvalues that are close to each other.

    """
    if atol is None:
        atol = 0
    eigval_ = np.asarray(eigval)
    eigval_dists = np.abs(eigval_[:, None] - eigval_[None, :])
    eigval_dists_close = eigval_dists < atol
    eigval_left_index = set(np.arange(eigval_.shape[0]))
    result = set()
    while eigval_left_index:
        i = eigval_left_index.pop()
        close_index = eigval_dists_close[i, :].nonzero()[0]
        s = set()
        for j in close_index:
            if i == j:
                continue
            elif j in eigval_left_index:
                eigval_left_index.remove(j)
            else:
                warnings.warn(
                    "atol is too large or too small.", RuntimeWarning, stacklevel=2
                )
                continue
            s.add(j)
        result.add(s)
    return result


def _matrix_rank_from_s(A: Any, S: Any, /, *, tol: Any = None, rtol: Any = None) -> Any:
    if tol is None:
        if rtol is None:
            rtol = max(A.shape[-2:]) * np.finfo(S.dtype).eps
        else:
            rtol = np.asarray(rtol)[..., None]
        tol = np.max(S, axis=-1, keepdims=True) * rtol
    else:
        tol = np.asarray(tol)[..., None]

    return np.count_nonzero(S > tol, axis=-1)


def get_multiplicity(
    eigval: np.ndarray[tuple[int], np.dtype[np.number]],
    eigvec: np.ndarray[tuple[int], np.dtype[np.number]] | None = None,
    /,
    *,
    atol: float | None = None,
) -> list[Multiplicity] | list[AlgebraicMultiplicity]:
    """
    Get the multiplicity of the eigenvalue.

    Does not support batched eigenvalues.

    Parameters
    ----------
    eigval : Array | NativeArray
        The eigenvalues of shape (n_eig,).
    eigvec : Array | NativeArray | None, optional
        The eigenvectors of shape (n, n_eig), by default None.
    atol : float | None, optional
        The threshold to treat eigenvalues as the same.

    Returns
    -------
    int
        The multiplicity of the eigenvalue.

    """
    atol = atol or 0
    if eigval.ndim != 1:
        raise ValueError("eigval should be 1D array.")
    if eigvec is not None:
        if eigvec.ndim != 2:
            raise ValueError("eigvec should be 2D array.")
        if eigval.shape[0] != eigvec.shape[1]:
            raise ValueError(
                f"{eigval.shape[0]=} should be equal to {eigvec.shape[1]=}."
            )
    groups = group_close_eigval(eigval, atol=atol)
    result: list[Multiplicity] | list[AlgebraicMultiplicity] = []
    for group in groups:
        group_list = list(group)
        eigvals_group = eigval[group_list]
        if eigvec is None:
            result.append(
                AlgebraicMultiplicity(  # type: ignore
                    eigvals=eigvals_group,
                )
            )
        else:
            eigvecs_group = eigvec[:, group_list]
            u, s, _ = np.linalg.svd(eigvecs_group)
            rank = _matrix_rank_from_s(eigvecs_group, s)
            eigvec_orthogonal = u[:, :rank]
            result.append(
                Multiplicity(
                    eigvals=eigvals_group,
                    eigvecs=eigvecs_group,
                    eigvec_orthogonal=eigvec_orthogonal,
                )
            )
    return result
