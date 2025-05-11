from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ._multiplicity import AlgebraicMultiplicityProtocol

TMultiplicity = TypeVar("TMultiplicity", bound=AlgebraicMultiplicityProtocol)


def plot_eigval_with_multiplicity(
    multiplicities: Sequence[TMultiplicity],
    /,
    *,
    ax: Axes | None = None,
    text_filter: Callable[[TMultiplicity], bool] | None = None,
    text_additionial: Callable[[TMultiplicity], str] | None = None,
    text_add_clue: bool = False,
    text_format_string: str = "{:.3g}",
    text_kwargs: dict[str, Any] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Plot eigenvalues with annotation of the multiplicity.

    Does not support batched eigenvalues.

    Parameters
    ----------
    multiplicities : Sequence[Tultiplicity]
        The multiplicities to plot.
    ax : plt.Axes | None, optional
        The axes to plot, by default None.
    text_filter : Callable[[TMultiplicity], bool] | None, optional
        A function to filter the text to be plotted, by default None.
        If None, all multiplicities will be plotted.
        If a function is provided, only the multiplicities that return True
        will be plotted.
    text_additionial : Callable[[TMultiplicity], str] | None, optional
        A function to add additional text to the plot, by default None.
        If None, no additional text will be plotted.
    text_add_clue : bool, optional
        Whether to add maximum distance between eigenvalues
        and singular values of the eigenvectors (if provided),
        by default False.
    text_format_string : str, optional
        The format string for the text, by default "{:.3g}".
        This is used to format the maximum distance between eigenvalues
        and singular values of the eigenvectors (if provided).
    text_kwargs : dict[str, Any] | None, optional
        The keyword arguments to pass to the text function, by default None.
        If None, no additional keyword arguments will be passed.
    scatter_kwargs : dict[str, Any] | None, optional
        The keyword arguments to pass to the scatter function, by default None.
        If None, no additional keyword arguments will be passed.

    """
    ax_ = ax or plt.gca()
    eigvals = np.stack(
        [x for m in multiplicities for x in getattr(m, "eigvals", [m.eigval])]
    )

    # label
    label = "Eigenvalues\n(Algebraic multiplicity"
    if text_add_clue:
        label += " (max |λ_i - λ_j|)"
    if any(hasattr(m, "geometric_multiplicity") for m in multiplicities):
        label += ", Geometric multiplicity"
        if text_add_clue:
            label += " (σ)"  # noqa
    label += ")"

    # plot
    ax_.scatter(
        eigvals.real, eigvals.imag, marker="x", label=label, **(scatter_kwargs or {})
    )

    for m in multiplicities:
        if text_filter is not None and not text_filter(m):
            continue

        # text
        text = f"{m.algebraic_multiplicity:g}"
        if m.algebraic_multiplicity > 1 and text_add_clue and hasattr(m, "eigvals"):
            max_dist = np.max(np.abs(m.eigvals[:, None] - m.eigvals[None, :]))
            text += f" ({text_format_string.format(max_dist)})"
        geometric_multiplicity = getattr(m, "geometric_multiplicity", None)
        if geometric_multiplicity is not None:
            text += f", {geometric_multiplicity:g}"
            if m.algebraic_multiplicity > 1 and text_add_clue and hasattr(m, "eigvecs"):
                s = np.linalg.svd(m.eigvecs, compute_uv=False)
                text += f" ({','.join([text_format_string.format(x) for x in s])})"
        if text_additionial is not None:
            text += f", {text_additionial(m)}"

        # add text
        ax_.text(m.eigval.real, m.eigval.imag, text, **(text_kwargs or {}))

    ax_.set_title("Eigenvalues in the complex plane")
    ax_.set_xlabel("Re λ")
    ax_.set_ylabel("Im λ")
    ax_.legend()
