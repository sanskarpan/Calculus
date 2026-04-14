"""
Calculus Library for AI/ML/DL
==============================

A comprehensive calculus library built from scratch in Python,
designed specifically for understanding the mathematical foundations
of optimization, backpropagation, and deep learning.

Stable Modules:
---------------
- differentiation: Numerical differentiation (derivatives, gradients, Jacobians, Hessians)
- integration: Numerical integration (Riemann, trapezoidal, Simpson's, Monte Carlo)
- autograd: Reverse-mode automatic differentiation with computational graphs

Available via direct import:
- taylor_series: Taylor expansions and function approximation

This is an educational package for learning scalar calculus and toy autograd.
Not intended for production use.
"""

from .differentiation import (
    derivative,
    partial_derivative,
    gradient,
    jacobian,
    hessian,
    gradient_descent_step,
)

from .integration import (
    riemann_sum,
    trapezoidal_rule,
    simpsons_rule,
    monte_carlo_integration,
    double_integral,
)

from .autograd import Variable, ComputationGraph, backward

from .taylor_series import (
    taylor_series,
    taylor_series_numerical,
    exp_taylor,
    sin_taylor,
    cos_taylor,
    log_taylor,
    linear_approximation,
    quadratic_approximation,
)

__version__ = "1.0.0"
__all__ = [
    "derivative",
    "partial_derivative",
    "gradient",
    "jacobian",
    "hessian",
    "gradient_descent_step",
    "riemann_sum",
    "trapezoidal_rule",
    "simpsons_rule",
    "monte_carlo_integration",
    "double_integral",
    "Variable",
    "ComputationGraph",
    "backward",
    "taylor_series",
    "taylor_series_numerical",
    "exp_taylor",
    "sin_taylor",
    "cos_taylor",
    "log_taylor",
    "linear_approximation",
    "quadratic_approximation",
]
