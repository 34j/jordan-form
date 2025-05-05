== Definition

Let $T: CC -> CC^n$ an matrix function analytic in a neighborhood of $lambda$.

$x_0,dots,x_(r-1) in CC^n$ is called a *Generalized Jordan chain* (of length $r$) coressponding to $lambda$ if

$
sum_(k=0)^l 1/k! (d^k T(lambda))/(d lambda^k) x_(l-k) = 0
$

$x_0,dots,x_(r-1)$ is called a *Jordan chain* if $x_0,dots,x_(r-1)$ is a generalized Jordan chain and $x_0 != 0$.

== Example

Let
$
T(lambda) = mat(
  lambda^2, 1, 0;
  0, lambda, 0;
  0, 0, lambda
)
==>
T'(lambda) = mat(
  2 lambda, 1, 0;
  0, 1, 0;
  0, 0, 1
),
T''(lambda) = mat(
  2, 0, 0;
  0, 0, 0;
  0, 0, 0
)
$

Then $lambda = 0$ is an eigenvalue of $T$ with algebraic multiplicity $3$.

$
T(0) = mat(
  0, 1, 0;
  0, 0, 0;
  0, 0, 0
),
T'(0) = mat(
  0, 0, 0;
  0, 1, 0;
  0, 0, 1
),
T''(0) = mat(
  2, 0, 0;
  0, 0, 0;
  0, 0, 0
)
$

The generalized Jordan chains of length $1$ are:
$
x_0 in {
  vec(1, 0, 0),
  vec(0, 1, 0)
}
$
