
# `gpAEP`

`gpAEP` is a sub-tool of `Ard` for the computation of expected production based
on:

- probabilistic resource distributions $p$, and
- hard-to-sample evaluation of the yield response $f$.

For example:
$$
\mathrm{P} = \int \cdots \int p(x_1, \ldots, x_d) f(x_1, \ldots, x_d) \mathrm{\,d} x_d \cdots \mathrm{\,d} x_1
$$

The fundamental use case is for *A*nnual *E*nergy *P*roduction (AEP) for wind farms.
For wind farms, it is typical to represent the wind resource in terms of the wind resource joint probability density function (pdf), $p(\psi, V)$, and wind farm power response function, $f(\psi, V)$, primarily.
In this case, we would have $x_1 = \psi$ where $\psi$ represents the wind
direction and $x_2 = V$, where $V$ represents the wind speed[^additionalvariables].
Accurate evaluations of the farm power $f(\psi, V)$ can be very cost-intensive due to complex aerodynamic interactions, and this project seeks to make the evaluation of $\mathrm{P}= \mathrm{AEP}$ with minimal numbers of evaluations.

[^additionalvariables]: In addition to wind speed and direction, it is rather likely that things like turbulence intensity (TI), and thermal condition variables are of considerable importance to production estimates and the approach herein is likely to enable improved research approaches to capture their effects.

Taking a more general view, we re-interpret gpAEP to refer to *A*ssessment *E*fficient *P*roduction using Gaussian Processes (*gp*), and treat the integral as an assessment of some well known underlying measure of potential $p$ and exploitability response of that potential $f$.
In this view, the potential could be anything that is well characterizable and the exploitability could represent any quantity that represents a multiplicitive factor that characterizes the exploitation of the potential and that is *hard to assess accurately*.
