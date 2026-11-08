# Jordan Form

<p align="center">
  <a href="https://github.com/34j/jordan-form/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/jordan-form/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://jordan-form.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/jordan-form.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
</p>
<p align="center">
  <a href="https://codecov.io/gh/34j/jordan-form">
    <img src="https://img.shields.io/codecov/c/github/34j/jordan-form.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/j178/prek">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json" alt="prek">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/jordan-form/">
    <img src="https://img.shields.io/pypi/v/jordan-form.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/jordan-form.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/jordan-form.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://jordan-form.readthedocs.io" target="_blank">https://jordan-form.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/jordan-form" target="_blank">https://github.com/34j/jordan-form </a>

---

Compute jordan canonical (normal) form of (nonlinear / generalized) eigenvalue problem

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install jordan-form
```

## Usage

Consider a function

$$
T(\lambda) = \begin{pmatrix}
\lambda^2 & -\lambda \\
0 & \lambda^2
\end{pmatrix}
$$

which eigenvalue is `0`. (Example 1. from [Nonlinear Eivenvalue Problems](https://www.mat.tuhh.de/forschung/rep/rep174.pdf) which is from Matrix Polynomials (Computer Science and Applied Mathematics). (1982). Academic Press.)

```python
import numpy as np

from jordan_form import canonical_jordan_chains


def f(
    eigval: float, derv: int
) -> np.ndarray[tuple[int, int], np.dtype[np.number]] | None:
    if derv == 0:
        return np.array([[eigval**2, -eigval], [0, eigval**2]])
    elif derv == 1:
        return np.array([[2 * eigval, -1], [0, 2 * eigval]])
    elif derv == 2:
        return np.array([[2, 0], [0, 2]])
    else:
        return np.zeros((2, 2))


chains = canonical_jordan_chains(
    f, 0, atol_rank=1e-3, rtol_rank=1e-3, atol_norm=1e-3, rtol_norm=1e-3
)
print(chains)
```

```python
CanonicalJordanChains(eigval=0, chains=[array([[1., 0.],
       [0., 1.],
       [0., 0.]]), array([[0., 1.]])])
```

## CLI Usage

```shell
> jordan-form "{{x^2,-x},{0,x^2}}" 0
Algebraic multiplicity: 4
Geometric multiplicity: 2
Dimension of generalized eigenspace: [2 3 4]
Dimension of ith generalized eigenvectors: [2 1 1]
Chain lengths: [3 1]
Chain 0:
[[1. 0.]
 [0. 1.]
 [0. 0.]]
Chain 1:
[[0. 1.]]
```

## References

- ["canonical set of Jordan chains" - Google Search](https://www.google.com/search?q=%22canonical+set+of+Jordan+chains%22)
- [Nonlinear Eivenvalue Problems](https://www.mat.tuhh.de/forschung/rep/rep174.pdf)
