import sympy as sp
import typer
from rich import print
from sympy.parsing.mathematica import parse_mathematica

from ._jordan_chain import canonical_jordan_chains, sympy_func

app = typer.Typer()


@app.command()
def main(expr: str, eigval: float, rtol: float = 1e-3, atol: float = 1e-3) -> None:
    """
    Main function to run the script.

    Parameters
    ----------
    expr : str
        The input function.
    eigval : float
        The eigenvalue.
    rtol : float
        The relative tolerance.
    atol : float
        The absolute tolerance.

    """
    mat = sp.Matrix(parse_mathematica(expr))
    chains = canonical_jordan_chains(
        sympy_func(mat),
        eigval,
        rtol_norm=rtol,
        rtol_rank=rtol,
        atol_norm=atol,
        atol_rank=atol,
    )
    print(f"Algebraic multiplicity: {chains.algebraic_multiplicity}")
    print(f"Geometric multiplicity: {chains.geometric_multiplicity}")
    print(f"Dimension of generalized eigenspace: {chains.dim_generalized_eigenspace}")
    print(
        "Dimension of ith generalized eigenvectors: "
        f"{chains.dim_ith_generalized_eigenvectors}"
    )
    print(f"Chain lengths: {chains.chain_lengths}")
    for i, chain in enumerate(chains.chains):
        print(f"Chain {i}:")
        print(chain)
