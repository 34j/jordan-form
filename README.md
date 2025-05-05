# Jordan Form

<p align="center">
  <a href="https://github.com/34j/jordan-form/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/jordan-form/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://jordan-form.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/jordan-form.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/jordan-form">
    <img src="https://img.shields.io/codecov/c/github/34j/jordan-form.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
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

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
