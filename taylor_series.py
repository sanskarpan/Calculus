"""
Taylor Series and Function Approximation Module
================================================

Implements Taylor series expansions and function approximation methods.

Used for:
- Function approximation
- Understanding model linearization
- Error analysis
- Second-order optimization methods
"""

from typing import Callable, List
import math


def factorial(n: int) -> int:
    """Compute factorial n!"""
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")
    # Use the stdlib implementation (fast + no recursion depth issues).
    return math.factorial(n)


def taylor_series(
    f: Callable[[float], float],
    derivatives: List[Callable[[float], float]],
    a: float,
    x: float,
    n: int,
) -> float:
    """
    Compute Taylor series approximation of f around point a.

    Formula:
        f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ... + f⁽ⁿ⁾(a)(x-a)ⁿ/n!

    Used in:
    - Function approximation
    - Newton's method
    - Model simplification

    Args:
        f: Function to approximate
        derivatives: List of derivative functions [f', f'', f''', ...]
        a: Center point of expansion
        x: Point to evaluate at
        n: Number of terms

    Returns:
        Taylor approximation of f(x)

    Example:
        >>> # Approximate exp(x) around x=0
        >>> f = lambda x: math.exp(x)
        >>> derivs = [lambda x: math.exp(x)] * 5  # All derivatives are exp
        >>> taylor_series(f, derivs, a=0, x=1, n=5)
        2.708...  # ≈ e
    """
    result = f(a)  # 0th term

    for i in range(min(n, len(derivatives))):
        term = derivatives[i](a) * ((x - a) ** (i + 1)) / factorial(i + 1)
        result += term

    return result


def taylor_series_numerical(
    f: Callable[[float], float], a: float, x: float, n: int, h: float = 1e-5
) -> float:
    """
    Taylor series using numerical derivatives.

    More practical when analytical derivatives are unavailable.

    Args:
        f: Function to approximate
        a: Center point
        x: Evaluation point
        n: Number of terms
        h: Step size for numerical differentiation

    Returns:
        Taylor approximation
    """
    from .differentiation import derivative, second_derivative

    result = f(a)  # 0th term
    result += derivative(f, a, h) * (x - a)  # 1st order term

    if n >= 2:
        result += second_derivative(f, a, h) * ((x - a) ** 2) / 2  # 2nd order

    # Note: Higher order terms (n >= 3) require stable numerical differentiation
    # which is beyond simple finite differences. For now, we limit to 2nd order.

    return result


def maclaurin_series(
    f: Callable[[float], float],
    derivatives: List[Callable[[float], float]],
    x: float,
    n: int,
) -> float:
    """
    Maclaurin series (Taylor series centered at 0).

    Formula:
        f(x) ≈ f(0) + f'(0)x + f''(0)x²/2! + ...

    Args:
        f: Function to approximate
        derivatives: Derivative functions
        x: Evaluation point
        n: Number of terms

    Returns:
        Maclaurin approximation

    Common series:
        - exp(x) = 1 + x + x²/2! + x³/3! + ...
        - sin(x) = x - x³/3! + x⁵/5! - ...
        - cos(x) = 1 - x²/2! + x⁴/4! - ...
    """
    return taylor_series(f, derivatives, a=0, x=x, n=n)


def exp_taylor(x: float, n: int = 10) -> float:
    """
    Exponential function using Taylor series.

    e^x = 1 + x + x²/2! + x³/3! + ...

    Args:
        x: Input value
        n: Number of terms

    Returns:
        Approximation of e^x
    """
    result = 0.0
    for k in range(n):
        result += (x**k) / factorial(k)
    return result


def sin_taylor(x: float, n: int = 10) -> float:
    """
    Sine function using Taylor series.

    sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...

    Args:
        x: Input value (in radians)
        n: Number of terms

    Returns:
        Approximation of sin(x)
    """
    result = 0.0
    for k in range(n):
        term = ((-1) ** k) * (x ** (2 * k + 1)) / factorial(2 * k + 1)
        result += term
    return result


def cos_taylor(x: float, n: int = 10) -> float:
    """
    Cosine function using Taylor series.

    cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...

    Args:
        x: Input value (in radians)
        n: Number of terms

    Returns:
        Approximation of cos(x)
    """
    result = 0.0
    for k in range(n):
        term = ((-1) ** k) * (x ** (2 * k)) / factorial(2 * k)
        result += term
    return result


def log_taylor(x: float, n: int = 20) -> float:
    """
    Natural logarithm using Taylor series (valid for 0 < x <= 2).

    ln(x) = (x-1) - (x-1)²/2 + (x-1)³/3 - ... for |x-1| < 1

    Args:
        x: Input value (0 < x <= 2)
        n: Number of terms

    Returns:
        Approximation of ln(x)
    """
    if x <= 0 or x > 2:
        raise ValueError("Taylor series for log valid only for 0 < x <= 2")

    result = 0.0
    z = x - 1  # Shift to center at 1
    for k in range(1, n + 1):
        term = ((-1) ** (k + 1)) * (z**k) / k
        result += term
    return result


def linear_approximation(
    f: Callable[[float], float], a: float, x: float, h: float = 1e-5
) -> float:
    """
    First-order (linear) Taylor approximation.

    f(x) ≈ f(a) + f'(a)(x-a)

    Used in:
    - Newton's method
    - Gradient descent
    - Local linearization

    Args:
        f: Function to approximate
        a: Center point
        x: Evaluation point
        h: Step size for numerical derivative

    Returns:
        Linear approximation
    """
    from .differentiation import derivative

    return f(a) + derivative(f, a, h) * (x - a)


def quadratic_approximation(
    f: Callable[[float], float], a: float, x: float, h: float = 1e-5
) -> float:
    """
    Second-order (quadratic) Taylor approximation.

    f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2

    Used in:
    - Newton's method
    - Second-order optimization
    - Better local approximation

    Args:
        f: Function to approximate
        a: Center point
        x: Evaluation point
        h: Step size

    Returns:
        Quadratic approximation
    """
    from .differentiation import derivative, second_derivative

    f_a = f(a)
    f_prime_a = derivative(f, a, h)
    f_double_prime_a = second_derivative(f, a, h)

    return f_a + f_prime_a * (x - a) + 0.5 * f_double_prime_a * ((x - a) ** 2)


def taylor_error_bound(f_max_derivative: float, a: float, x: float, n: int) -> float:
    """
    Estimate error bound for Taylor series (Lagrange remainder).

    Error ≤ M * |x-a|^(n+1) / (n+1)!

    where M is max of |f^(n+1)| on interval [a, x]

    Args:
        f_max_derivative: Maximum value of (n+1)th derivative
        a: Center point
        x: Evaluation point
        n: Number of terms used

    Returns:
        Upper bound on approximation error
    """
    return f_max_derivative * (abs(x - a) ** (n + 1)) / factorial(n + 1)


def multivariate_taylor_first_order(
    f: Callable[[List[float]], float], a: List[float], x: List[float], h: float = 1e-5
) -> float:
    """
    First-order multivariate Taylor approximation.

    f(x) ≈ f(a) + ∇f(a) · (x - a)

    Used in:
    - Multivariable optimization
    - Gradient descent
    - Model linearization

    Args:
        f: Multivariate function
        a: Center point
        x: Evaluation point
        h: Step size

    Returns:
        Linear approximation
    """
    from .differentiation import gradient

    f_a = f(a)
    grad = gradient(f, a, h)

    # Compute dot product: grad · (x - a)
    dot_product = sum(g * (xi - ai) for g, xi, ai in zip(grad, x, a))

    return f_a + dot_product


def multivariate_taylor_second_order(
    f: Callable[[List[float]], float], a: List[float], x: List[float], h: float = 1e-5
) -> float:
    """
    Second-order multivariate Taylor approximation.

    f(x) ≈ f(a) + ∇f(a) · (x-a) + 0.5 * (x-a)^T H (x-a)

    Used in:
    - Newton's method in multiple dimensions
    - Second-order optimization

    Args:
        f: Multivariate function
        a: Center point
        x: Evaluation point
        h: Step size

    Returns:
        Quadratic approximation
    """
    from .differentiation import gradient, hessian

    f_a = f(a)
    grad = gradient(f, a, h)
    H = hessian(f, a, h)

    # First-order term
    first_order = sum(g * (xi - ai) for g, xi, ai in zip(grad, x, a))

    # Second-order term: (x-a)^T H (x-a)
    delta = [xi - ai for xi, ai in zip(x, a)]
    n = len(delta)

    second_order = 0.0
    for i in range(n):
        for j in range(n):
            second_order += delta[i] * H[i][j] * delta[j]

    return f_a + first_order + 0.5 * second_order


def approximate_function(
    f: Callable[[float], float], x_values: List[float], degree: int = 3
) -> Callable[[float], float]:
    """
    Create a polynomial approximation of a function.

    Uses Taylor series at the midpoint.

    Args:
        f: Function to approximate
        x_values: Domain of interest
        degree: Degree of polynomial

    Returns:
        Polynomial approximation function

    Example:
        >>> f = lambda x: math.exp(x)
        >>> approx = approximate_function(f, [0, 1, 2], degree=3)
        >>> approx(0.5)  # Approximate e^0.5
    """
    # Use midpoint as center
    a = (min(x_values) + max(x_values)) / 2

    # Compute derivatives numerically
    from .differentiation import derivative, second_derivative

    coeffs = [f(a)]  # c0
    if degree >= 1:
        coeffs.append(derivative(f, a))  # c1 (f'(a))
    if degree >= 2:
        coeffs.append(
            second_derivative(f, a)
        )  # c2 (f''(a)) - will be divided by 2! in poly()

    # Return polynomial function
    def poly(x):
        result = coeffs[0]
        delta = x - a
        for i in range(1, len(coeffs)):
            result += coeffs[i] * (delta**i) / factorial(i)
        return result

    return poly
