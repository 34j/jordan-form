from collections.abc import Sequence
from typing import Callable, TypeVar

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

    """
    ax_ = ax or plt.gca()
    eigvals = np.stack(
        [x for m in multiplicities for x in getattr(m, "eigvals", [m.eigval])]
    )
    ax_.scatter(
        eigvals.real,
        eigvals.imag,
        marker="x",
        label="Eigenvalues\n(Algebraic multiplicity"
        + (
            ", Geometric multiplicity"
            if any(hasattr(m, "geometric_multiplicity") for m in multiplicities)
            else ""
        )
        + ")",
    )
    for m in multiplicities:
        if text_filter is not None and not text_filter(m):
            continue
        ax_.text(
            m.eigval.real,
            m.eigval.imag,
            f"{m.algebraic_multiplicity}"
            + (
                f", {getattr(m, 'geometric_multiplicity', '')}"
                if hasattr(m, "geometric_multiplicity")
                else ""
            ),
        )
    ax_.set_title("Eigenvalues in the complex plane")
    ax_.set_xlabel("Re λ")
    ax_.set_ylabel("Im λ")
    ax_.legend()
