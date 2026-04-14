"""
Calculus & Multivariable Calculus Library for AI/ML/DL
=======================================================

A comprehensive calculus library built from scratch in Python,
designed specifically for understanding the mathematical foundations
of optimization, backpropagation, and deep learning.

Modules:
--------
- differentiation: Numerical and automatic differentiation
- integration: Numerical integration methods
- vector_calculus: Gradient, divergence, curl operations
- taylor_series: Taylor expansion and function approximation
- autograd: Automatic differentiation and computational graphs
- optimization_utils: Critical points, convexity checking
"""

from .differentiation import (
    derivative,
    partial_derivative,
    gradient,
    jacobian,
    hessian,
    gradient_descent_step
)

from .integration import (
    riemann_sum,
    trapezoidal_rule,
    simpsons_rule,
    monte_carlo_integration,
    double_integral
)

from .autograd import (
    Variable,
    ComputationGraph,
    backward
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
    "backward"
]
