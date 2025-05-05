__version__ = "0.0.0"
from ._jordan_chain import (
    CanonicalJordanChains,
    canonical_jordan_chains,
    canonoical_jordan_chains_from_unrestricted,
    geig_func,
)
from ._multiplicity import (
    AlgebraicMultiplicity,
    Multiplicity,
    group_close_eigvals,
    multiplicity,
)

__all__ = [
    "AlgebraicMultiplicity",
    "CanonicalJordanChains",
    "Multiplicity",
    "canonical_jordan_chains",
    "canonoical_jordan_chains_from_unrestricted",
    "geig_func",
    "group_close_eigvals",
    "multiplicity",
]
