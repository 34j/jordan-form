import warnings
from collections.abc import Sequence
from typing import Protocol, overload

import attrs
import numpy as np
from numpy.typing import NDArray


class AlgebraicMultiplicityProtocol(Protocol):
    @property
    def eigval(self) -> float:
        """The main eigenvalue."""
        ...

    @property
    def algebraic_multiplicity(self) -> int:
        """
        The algebraic multiplicity of the eigenvalue.

        The number of times the eigenvalue appears
        as a root of the characteristic polynomial.

        The number of generalized eigenvectors
        in a canonical system of Jordan chains.

        """
        ...


class MultiplicityProtocol(AlgebraicMultiplicityProtocol, Protocol):
    @property
    def eigvec_orthogonal(self) -> NDArray[np.number]:
        """The orthogonal eigenvectors of shape (n, geometric_multiplicity)."""
        ...

    @property
    def geometric_multiplicity(self) -> int:
        """
        The geometric multiplicity of the eigenvalue.

        The dimension of the eigenspace of the eigenvalue.
        Less than or equal to the algebraic multiplicity.

        """
        ...


@attrs.frozen(kw_only=True)
class AlgebraicMultiplicity(AlgebraicMultiplicityProtocol):
    eigvals: np.ndarray[tuple[int], np.dtype[np.number]]
    """The eigenvalues of shape (algebraic_multiplicity,)."""

    @property
    def eigval(self) -> float:
        return np.mean(self.eigvals, axis=0)

    @property
    def algebraic_multiplicity(self) -> int:
        return self.eigvals.shape[0]


@attrs.frozen(kw_only=True)
class Multiplicity(AlgebraicMultiplicity, MultiplicityProtocol):
    eigvecs: np.ndarray[tuple[int, int], np.dtype[np.number]]
    """The eigenvectors of shape (n, algebraic_multiplicity)."""
    eigvec_orthogonal: np.ndarray[tuple[int, int], np.dtype[np.number]]

    @property
    def geometric_multiplicity(self) -> int:
        return self.eigvec_orthogonal.shape[1]


def get_tol(
    base: NDArray[np.floating],
    /,
    *,
    rtol: NDArray[np.floating] | None = None,
    atol: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    if rtol is None:
        rtol = np.finfo(base.dtype).eps
    if atol is None:
        atol = np.finfo(base.dtype).eps
    return rtol * np.abs(base) + atol


def group_close_eigvals(
    eigvals: Sequence[float],
    /,
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> list[list[int]]:
    """
    Group the eigenvalues that are close to each other.

    Parameters
    ----------
    eigvals : Sequence[float]
        The eigenvalues.
    atol : float | None, optional
        The threshold to treat eigenvalues as the same.
        Defaults to ``np.finfo(eigval.dtype).eps``.
    rtol : float | None, optional
        The threshold to treat eigenvalues as the same.
        Defaults to ``np.finfo(eigval.dtype).eps``.

    Returns
    -------
    list[list[int]]
        The indices of the eigenvalues that are close to each other.

    """
    if len(eigvals) == 0:
        return []
    eigval_ = np.asarray(eigvals)
    tol = get_tol(eigval_.max(), rtol=rtol, atol=atol)
    eigval_dists = np.abs(eigval_[:, None] - eigval_[None, :])
    eigval_dists_close = eigval_dists < tol
    eigval_left_index = set(np.arange(eigval_.shape[0]))
    result = []
    while eigval_left_index:
        i = eigval_left_index.pop()
        close_index = eigval_dists_close[i, :].nonzero()[0]
        s = []
        for j in close_index:
            if i == j:
                pass
            elif j in eigval_left_index:
                eigval_left_index.remove(j)
            else:
                warnings.warn(
                    "atol is too large or too small.", RuntimeWarning, stacklevel=2
                )
                continue
            s.append(j)
        result.append(s)
    return result


def _matrix_rank_from_s(
    M: NDArray[np.number],
    s: NDArray[np.floating],
    /,
    *,
    atol: NDArray[np.floating] | None = None,
    rtol: NDArray[np.floating] | None = None,
) -> int:
    tol = get_tol(s.max() * max(M.shape[-2:]), rtol=rtol, atol=atol)
    return np.count_nonzero(s > tol, axis=-1)


@overload
def multiplicity(
    eigvals: NDArray[np.number],
    eigvecs: NDArray[np.number] = ...,
    /,
    *,
    atol_algebraic: float | None = ...,
    rtol_algebraic: float | None = ...,
    atol_geometric: float | None = ...,
    rtol_geometric: float | None = ...,
) -> list[Multiplicity]: ...
@overload
def multiplicity(  # type: ignore
    eigvals: NDArray[np.number],
    eigvecs: None = ...,
    /,
    *,
    atol_algebraic: float | None = ...,
    rtol_algebraic: float | None = ...,
    atol_geometric: float | None = ...,
    rtol_geometric: float | None = ...,
) -> list[AlgebraicMultiplicity]: ...
def multiplicity(
    eigvals: np.ndarray[tuple[int], np.dtype[np.number]],
    eigvecs: np.ndarray[tuple[int, int], np.dtype[np.number]] | None = None,
    /,
    *,
    atol_algebraic: float | None = None,
    rtol_algebraic: float | None = None,
    atol_geometric: float | None = None,
    rtol_geometric: float | None = None,
) -> list[Multiplicity] | list[AlgebraicMultiplicity]:
    """
    Get the multiplicity of the eigenvalue.

    Does not support batched eigenvalues.

    Parameters
    ----------
    eigvals : Array | NativeArray
        The eigenvalues of shape (n_eig,).
    eigvecs : Array | NativeArray | None, optional
        The eigenvectors of shape (n, n_eig), by default None.
    atol_algebraic : float | None, optional
        The threshold to treat eigenvalues as the same.
        Defaults to ``np.finfo(eigval.dtype).eps``.
    rtol_algebraic : float | None, optional
        The threshold to treat eigenvalues as the same.
        Defaults to ``np.finfo(eigval.dtype).eps``.
    atol_geometric : float, optional
        Threshold below which SVD values are considered zero.
        Defaults to ``np.finfo(A.dtype).eps``.
    rtol_geometric : float, optional
        Threshold below which SVD values are considered zero.
        Defaults to ``np.finfo(A.dtype).eps``.

    Returns
    -------
    int
        The multiplicity of the eigenvalue.

    """
    if eigvals.ndim != 1:
        raise ValueError("eigval should be 1D array.")
    if eigvecs is not None:
        if eigvecs.ndim != 2:
            raise ValueError("eigvec should be 2D array.")
        if eigvals.shape[0] != eigvecs.shape[1]:
            raise ValueError(
                f"{eigvals.shape[0]=} should be equal to {eigvecs.shape[1]=}."
            )
    groups = group_close_eigvals(eigvals, atol=atol_algebraic, rtol=rtol_algebraic)
    result: list[Multiplicity] | list[AlgebraicMultiplicity] = []
    for group in groups:
        eigvals_group = eigvals[group]
        if eigvecs is None:
            result.append(
                AlgebraicMultiplicity(  # type: ignore
                    eigvals=eigvals_group,
                )
            )
        else:
            eigvecs_group = eigvecs[:, group]
            u, s, _ = np.linalg.svd(eigvecs_group)
            rank = _matrix_rank_from_s(
                eigvecs_group, s, atol=atol_geometric, rtol=rtol_geometric
            )
            eigvec_orthogonal = u[:, :rank]
            result.append(
                Multiplicity(
                    eigvals=eigvals_group,
                    eigvecs=eigvecs_group,
                    eigvec_orthogonal=eigvec_orthogonal,
                )
            )
    if eigvecs is None:
        result.sort(key=lambda x: x.algebraic_multiplicity, reverse=True)
    else:
        result.sort(
            key=lambda x: (x.algebraic_multiplicity, x.geometric_multiplicity),  # type: ignore
            reverse=True,
        )
    return result
