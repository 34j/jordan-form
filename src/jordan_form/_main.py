import warnings
from collections.abc import Sequence
from typing import overload

import attrs
import numpy as np
import scipy
import scipy.special
from numpy.typing import NDArray


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
    eigvecs: np.ndarray[tuple[int], np.dtype[np.number]] | None = None,
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


def fixed_jordan_chains(A: NDArray[np.number], /) -> NDArray[np.number]:
    """
    Get the Jordans chain of the matrix of fixed length.

    Parameters
    ----------
    A : NDArray[np.number]
        The matrix derivatives of shape (l_chain, n, n).

    Returns
    -------
    NDArray[np.number]
        The Jordan chains of shape (n_chain, l_chain, n).

    """
    m = A.shape[0]
    n = A.shape[1]
    mat = np.stack(
        [
            np.moveaxis(
                np.concat(
                    (
                        np.flip(
                            A[: j + 1, :, :]
                            / scipy.special.factorial(np.arange(j + 1)[:, None, None]),
                            axis=0,
                        ),
                        np.zeros((m - j - 1, n, n), dtype=A.dtype, device=A.device),
                    ),
                    axis=0,
                ),
                0,
                1,
            ).reshape(n, m * n)
            for j in range(m)
        ],
        axis=0,
    ).reshape(m * n, m * n)
    # (m*n, n_jordan_chain)
    chain = scipy.linalg.null_space(mat)
    # (n_jordan_chain, m*n)
    chain = np.moveaxis(chain, -1, 0)
    # (n_jordan_chain, m, n)
    chain = chain.reshape(chain.shape[0], m, n)
    return chain


def proj(a_from: NDArray[np.number], a_to: NDArray[np.number], /) -> NDArray[np.number]:
    """
    Project a_from to a_to.

    Parameters
    ----------
    a_from : NDArray[np.number]
        The vector to be projected of shape (..., n).
    a_to : NDArray[np.number]
        The vector space to project to of shape (..., n, n_dim).

    Returns
    -------
    NDArray[np.number]
        The projected vector of shape (..., n).

    """
    a_to = a_to / np.linalg.norm(a_to, axis=-2, keepdims=True)
    return np.sum(np.dot(a_from, a_to) * a_to, axis=-1)


def canonoical_jordan_chains(
    A: NDArray[np.number],
    /,
    *,
    hermitian: bool | None = None,
    atol_rank: float | None = None,
    rtol_rank: float | None = None,
    atol_norm: float | None = None,
    rtol_norm: float | None = None,
    flatten: bool = True,
) -> list[NDArray[np.number]]:
    """
    Get the Jordan chains of the matrix.

    Parameters
    ----------
    A : NDArray[np.number]
        The matrix derivatives of shape (multiplicity, n, n).
    hermitian : bool, optional
        If True, `A` is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.
        Defaults to False.
    atol_rank : float, optional
        Threshold below which SVD values are considered zero.
        Defaults to ``np.finfo(A.dtype).eps``.
    rtol_rank : float, optional
        Threshold below which SVD values are considered zero.
        Defaults to ``np.finfo(A.dtype).eps``.
    atol_norm : float, optional
        Threshold below which norm values are considered zero.
        Defaults to ``np.finfo(A.dtype).eps``.
    rtol_norm : float, optional
        Threshold below which norm values are considered zero.
        Defaults to ``np.finfo(A.dtype).eps``.
    flatten : bool, optional
        If True, flatten the chains. Defaults to True.

    Returns
    -------
    list[NDArray[np.number]]
        The Jordan chains.
        (-l_chain)-th element has Jordan chains of length l_chain
        of shape (n_chain, l_chain, n).

    """
    chains: list[NDArray[np.number]] = []
    for i in range(A.shape[0], 0, -1):
        A = A[:i, :, :]
        chain = fixed_jordan_chains(A)
        # filter and normalize based on the first element
        norm = np.linalg.norm(chain[:, 0, :], axis=-1)
        norm_filter = norm > get_tol(np.max(norm), rtol=rtol_norm, atol=atol_norm)
        chain = chain[norm_filter, :, :]
        chain = chain / norm[norm_filter, None, None]
        if not chain.size:
            continue
        if chains:
            cut_chain = _get_space(chains, i)
            # [n_chain_cut, l_chain, n]
            # [n_chain, n], [1, n_chain_cut, n] -> [n_chain, n_chain_cut]
            d = np.sum(chain[:, None, 0, :] * cut_chain[None, :, 0, :], axis=-1)
            # [n_chain, l_chain, n]
            chain = chain - np.sum(
                d[:, :, None, None] * cut_chain[None, :, :, :], axis=1
            )
        # svd, remove duplicated chains
        chain = np.swapaxes(chain, 0, 1)
        u, s, _ = np.linalg.svd(chain[0, :, :], hermitian=hermitian)
        rank = _matrix_rank_from_s(chain[0, :, :], s, atol=atol_rank, rtol=rtol_rank)
        chain = u.T[:rank] @ chain
        chain = np.swapaxes(chain, 0, 1)
        if chain.size:
            chains.append(chain)
    if flatten:
        chains = [c for chain_ in chains for c in chain_]
    return chains


def _get_space(
    chains: list[NDArray[np.number]],
    length: int,
    /,
) -> NDArray[np.number]:
    """
    Get the cut chains.

    Parameters
    ----------
    chains : list[NDArray[np.number]]
        The Jordan chains.
    length : int
        The length to cut.

    Returns
    -------
    NDArray[np.number]
        The cut chains of shape (n_chain, l_chain, n).

    """
    res = np.concat([chains_[:, :length, :] for chains_ in chains], axis=0)
    return res
