"""
Numerical Integration Module
=============================

Implements numerical integration methods for:
- Definite integrals
- Multiple integrals
- Monte Carlo integration
- Applications to probability

These are essential for:
- Computing expectations
- Probability distributions
- Loss function integration
- Normalizing constants
"""

from typing import Callable, Tuple, List
import random
import math


def riemann_sum(f: Callable[[float], float], a: float, b: float,
                n: int = 1000, method: str = 'midpoint') -> float:
    """
    Compute definite integral using Riemann sums.

    Used in:
    - Basic numerical integration
    - Understanding integral definition

    Args:
        f: Function to integrate
        a: Lower limit
        b: Upper limit
        n: Number of rectangles
        method: 'left', 'right', or 'midpoint'

    Returns:
        Approximate integral ∫[a,b] f(x) dx

    Methods:
        - Left: Use left endpoint of each subinterval
        - Right: Use right endpoint of each subinterval
        - Midpoint: Use midpoint of each subinterval (more accurate)
    """
    dx = (b - a) / n
    total = 0.0

    for i in range(n):
        if method == 'left':
            x = a + i * dx
        elif method == 'right':
            x = a + (i + 1) * dx
        elif method == 'midpoint':
            x = a + (i + 0.5) * dx
        else:
            raise ValueError(f"Unknown method: {method}")

        total += f(x) * dx

    return total


def trapezoidal_rule(f: Callable[[float], float], a: float, b: float,
                     n: int = 1000) -> float:
    """
    Compute definite integral using trapezoidal rule.

    More accurate than Riemann sums!

    Used in:
    - Numerical integration
    - Computing expectations
    - Probability calculations

    Args:
        f: Function to integrate
        a: Lower limit
        b: Upper limit
        n: Number of trapezoids

    Returns:
        Approximate integral ∫[a,b] f(x) dx

    Formula:
        ∫f(x)dx ≈ (dx/2) * [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]

    Error: O(n⁻²)
    """
    dx = (b - a) / n
    total = 0.5 * (f(a) + f(b))

    for i in range(1, n):
        x = a + i * dx
        total += f(x)

    return total * dx


def simpsons_rule(f: Callable[[float], float], a: float, b: float,
                  n: int = 1000) -> float:
    """
    Compute definite integral using Simpson's rule.

    Most accurate basic method (uses parabolic approximation)!

    Used in:
    - High-accuracy integration
    - Scientific computing
    - Probability integrals

    Args:
        f: Function to integrate
        a: Lower limit
        b: Upper limit
        n: Number of intervals (must be even)

    Returns:
        Approximate integral ∫[a,b] f(x) dx

    Formula:
        ∫f(x)dx ≈ (dx/3) * [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + f(xₙ)]

    Error: O(n⁻⁴)
    """
    if n % 2 != 0:
        n += 1  # Make even

    dx = (b - a) / n
    total = f(a) + f(b)

    for i in range(1, n):
        x = a + i * dx
        if i % 2 == 0:
            total += 2 * f(x)
        else:
            total += 4 * f(x)

    return total * dx / 3


def monte_carlo_integration(f: Callable[[float], float], a: float, b: float,
                            n_samples: int = 10000) -> float:
    """
    Compute integral using Monte Carlo sampling.

    Powerful for high-dimensional integrals!

    Used in:
    - High-dimensional integration
    - Expectations in probability
    - Bayesian inference

    Args:
        f: Function to integrate
        a: Lower limit
        b: Upper limit
        n_samples: Number of random samples

    Returns:
        Approximate integral ∫[a,b] f(x) dx

    Formula:
        ∫[a,b] f(x) dx ≈ (b-a) * (1/n) * Σ f(xᵢ) where xᵢ ~ Uniform(a,b)

    Error: O(n⁻⁰·⁵) but dimension-independent!
    """
    total = 0.0

    for _ in range(n_samples):
        x = random.uniform(a, b)
        total += f(x)

    return (b - a) * total / n_samples


def adaptive_integration(f: Callable[[float], float], a: float, b: float,
                         tolerance: float = 1e-6, max_depth: int = 10) -> float:
    """
    Adaptive integration with automatic refinement.

    Refines intervals where function varies rapidly.

    Args:
        f: Function to integrate
        a: Lower limit
        b: Upper limit
        tolerance: Desired accuracy
        max_depth: Maximum recursion depth

    Returns:
        Approximate integral
    """
    def integrate_segment(left, right, depth):
        mid = (left + right) / 2

        # Simpson's rule on whole interval
        whole = simpsons_rule(f, left, right, n=2)

        # Simpson's rule on each half
        left_half = simpsons_rule(f, left, mid, n=2)
        right_half = simpsons_rule(f, mid, right, n=2)

        # Error estimate
        error = abs(whole - (left_half + right_half))

        if error < tolerance or depth >= max_depth:
            return left_half + right_half
        else:
            # Subdivide further
            return (integrate_segment(left, mid, depth + 1) +
                    integrate_segment(mid, right, depth + 1))

    return integrate_segment(a, b, 0)


def double_integral(f: Callable[[float, float], float],
                    x_range: Tuple[float, float],
                    y_range: Tuple[float, float],
                    nx: int = 100, ny: int = 100) -> float:
    """
    Compute double integral using 2D trapezoidal rule.

    Used in:
    - 2D probability distributions
    - Surface integrals
    - Joint distributions

    Args:
        f: Function of two variables f(x, y)
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        nx: Number of x subdivisions
        ny: Number of y subdivisions

    Returns:
        Approximate integral ∫∫ f(x,y) dx dy
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    total = 0.0

    for i in range(nx):
        for j in range(ny):
            x = x_min + (i + 0.5) * dx
            y = y_min + (j + 0.5) * dy
            total += f(x, y) * dx * dy

    return total


def triple_integral(f: Callable[[float, float, float], float],
                    x_range: Tuple[float, float],
                    y_range: Tuple[float, float],
                    z_range: Tuple[float, float],
                    n: int = 50) -> float:
    """
    Compute triple integral.

    Args:
        f: Function of three variables f(x, y, z)
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        z_range: (z_min, z_max)
        n: Number of subdivisions per dimension

    Returns:
        Approximate integral ∫∫∫ f(x,y,z) dx dy dz
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    dx = (x_max - x_min) / n
    dy = (y_max - y_min) / n
    dz = (z_max - z_min) / n

    total = 0.0

    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = x_min + (i + 0.5) * dx
                y = y_min + (j + 0.5) * dy
                z = z_min + (k + 0.5) * dz
                total += f(x, y, z) * dx * dy * dz

    return total


def monte_carlo_multidim(f: Callable[[List[float]], float],
                         bounds: List[Tuple[float, float]],
                         n_samples: int = 100000) -> float:
    """
    Monte Carlo integration for multidimensional functions.

    This is where Monte Carlo shines - handles any dimension!

    Used in:
    - High-dimensional probability
    - Bayesian inference
    - Expectation computation

    Args:
        f: Function of d variables
        bounds: List of (min, max) for each dimension
        n_samples: Number of random samples

    Returns:
        Approximate integral

    Example:
        >>> # Integrate f(x,y,z) = x²+y²+z² over [0,1]³
        >>> f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
        >>> bounds = [(0,1), (0,1), (0,1)]
        >>> monte_carlo_multidim(f, bounds)
        1.0  # Approximate
    """
    d = len(bounds)
    volume = 1.0
    for low, high in bounds:
        volume *= (high - low)

    total = 0.0

    for _ in range(n_samples):
        x = [random.uniform(low, high) for low, high in bounds]
        total += f(x)

    return volume * total / n_samples


def expectation(f: Callable[[float], float], pdf: Callable[[float], float],
                a: float, b: float, n: int = 1000) -> float:
    """
    Compute expectation E[f(X)] where X has PDF p(x).

    Used in:
    - Probability theory
    - Expected loss
    - Decision theory

    Args:
        f: Function to take expectation of
        pdf: Probability density function
        a: Lower limit of support
        b: Upper limit of support
        n: Number of integration points

    Returns:
        E[f(X)] = ∫ f(x) p(x) dx

    Example:
        >>> # E[X²] where X ~ Uniform(0,1)
        >>> f = lambda x: x**2
        >>> pdf = lambda x: 1.0  # Uniform on [0,1]
        >>> expectation(f, pdf, 0, 1)
        0.333...  # Approximately 1/3
    """
    def integrand(x):
        return f(x) * pdf(x)

    return simpsons_rule(integrand, a, b, n)


def variance(f: Callable[[float], float], pdf: Callable[[float], float],
             a: float, b: float, n: int = 1000) -> float:
    """
    Compute variance Var[f(X)] where X has PDF p(x).

    Args:
        f: Function
        pdf: Probability density function
        a: Lower limit
        b: Upper limit
        n: Number of integration points

    Returns:
        Var[f(X)] = E[f(X)²] - E[f(X)]²
    """
    # E[f(X)²]
    def f_squared(x):
        return f(x) ** 2

    E_f_squared = expectation(f_squared, pdf, a, b, n)

    # E[f(X)]²
    E_f = expectation(f, pdf, a, b, n)
    E_f_squared_alt = E_f ** 2

    return E_f_squared - E_f_squared_alt


def line_integral(F: Callable[[float, float], Tuple[float, float]],
                  curve: Callable[[float], Tuple[float, float]],
                  t_range: Tuple[float, float],
                  n: int = 1000) -> float:
    """
    Compute line integral of vector field along a curve.

    Used in:
    - Work calculations
    - Path integrals
    - Physics applications

    Args:
        F: Vector field F(x,y) = (P, Q)
        curve: Parametric curve r(t) = (x(t), y(t))
        t_range: Parameter range (t_min, t_max)
        n: Number of subdivisions

    Returns:
        Line integral ∫_C F · dr

    Example:
        >>> # F = (y, x) along circle r(t) = (cos(t), sin(t))
        >>> F = lambda x, y: (y, x)
        >>> curve = lambda t: (math.cos(t), math.sin(t))
        >>> line_integral(F, curve, (0, 2*math.pi))
    """
    t_min, t_max = t_range
    dt = (t_max - t_min) / n

    total = 0.0

    for i in range(n):
        t = t_min + (i + 0.5) * dt

        # Position and derivative at t
        x, y = curve(t)

        # Numerical derivative of curve
        h = 1e-5
        x_plus, y_plus = curve(t + h)
        x_minus, y_minus = curve(t - h)

        dx_dt = (x_plus - x_minus) / (2 * h)
        dy_dt = (y_plus - y_minus) / (2 * h)

        # Vector field at position
        P, Q = F(x, y)

        # F · dr = P dx + Q dy
        total += (P * dx_dt + Q * dy_dt) * dt

    return total


def cumulative_integral(f: Callable[[float], float], a: float,
                        x_values: List[float], n: int = 1000) -> List[float]:
    """
    Compute cumulative integral F(x) = ∫[a,x] f(t) dt for multiple x values.

    Used in:
    - Cumulative distribution functions
    - Antiderivatives
    - Probability calculations

    Args:
        f: Function to integrate
        a: Starting point
        x_values: List of upper limits
        n: Subdivisions per integral

    Returns:
        List of F(x) values

    Example:
        >>> # CDF of standard normal (approximate)
        >>> f = lambda x: math.exp(-x**2/2) / math.sqrt(2*math.pi)
        >>> cdf_values = cumulative_integral(f, -10, [0, 1, 2, 3])
    """
    results = []

    for x in x_values:
        if x >= a:
            integral = simpsons_rule(f, a, x, n)
        else:
            integral = -simpsons_rule(f, x, a, n)

        results.append(integral)

    return results


def integrate_to_find_constant(f: Callable[[float], float],
                               a: float, b: float,
                               target_integral: float = 1.0,
                               n: int = 1000) -> float:
    """
    Find normalization constant c such that ∫[a,b] c*f(x) dx = target.

    Used in:
    - Normalizing probability distributions
    - Finding partition functions

    Args:
        f: Unnormalized function
        a: Lower limit
        b: Upper limit
        target_integral: Desired integral value
        n: Number of integration points

    Returns:
        Normalization constant c

    Example:
        >>> # Normalize f(x) = x² on [0,1] to be a PDF
        >>> f = lambda x: x**2
        >>> c = integrate_to_find_constant(f, 0, 1, target_integral=1.0)
        >>> # Now c*f(x) integrates to 1
    """
    current_integral = simpsons_rule(f, a, b, n)

    if abs(current_integral) < 1e-10:
        raise ValueError("Function integrates to zero, cannot normalize")

    return target_integral / current_integral
