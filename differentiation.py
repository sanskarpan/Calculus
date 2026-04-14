"""
Differentiation Module
======================

Implements numerical differentiation methods essential for AI/ML/DL:
- Finite differences (forward, backward, central)
- Partial derivatives
- Gradient computation
- Jacobian matrix
- Hessian matrix
- Directional derivatives

These are fundamental for:
- Gradient descent optimization
- Backpropagation
- Sensitivity analysis
- Newton's method
"""

from typing import Callable, List, Optional, Union
import math


def derivative(f: Callable[[float], float], x: float, h: float = 1e-5,
               method: str = 'central') -> float:
    """
    Compute the derivative of a function at a point using finite differences.

    Used in:
    - Gradient descent
    - Optimization algorithms
    - Sensitivity analysis

    Args:
        f: Function to differentiate
        x: Point at which to compute derivative
        h: Step size (smaller = more accurate, but numerical errors)
        method: 'forward', 'backward', or 'central'

    Returns:
        Approximate derivative f'(x)

    Methods:
        - Forward: f'(x) ≈ (f(x+h) - f(x)) / h
        - Backward: f'(x) ≈ (f(x) - f(x-h)) / h
        - Central: f'(x) ≈ (f(x+h) - f(x-h)) / (2h) [most accurate]
    """
    if method == 'forward':
        return (f(x + h) - f(x)) / h
    elif method == 'backward':
        return (f(x) - f(x - h)) / h
    elif method == 'central':
        return (f(x + h) - f(x - h)) / (2 * h)
    else:
        raise ValueError(f"Unknown method: {method}")


def second_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """
    Compute the second derivative using central differences.

    Used in:
    - Newton's method
    - Convexity checking
    - Acceleration in optimization

    Args:
        f: Function to differentiate
        x: Point at which to compute second derivative
        h: Step size

    Returns:
        Approximate second derivative f''(x)

    Formula: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
    """
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)


def partial_derivative(f: Callable[[List[float]], float], x: List[float],
                       variable_index: int, h: float = 1e-5) -> float:
    """
    Compute partial derivative with respect to one variable.

    Used in:
    - Multivariable optimization
    - Neural network gradients
    - Sensitivity analysis

    Args:
        f: Multivariate function
        x: Point at which to compute partial derivative
        variable_index: Index of variable to differentiate with respect to
        h: Step size

    Returns:
        Partial derivative ∂f/∂x_i
    """
    x_plus = x.copy()
    x_minus = x.copy()

    x_plus[variable_index] += h
    x_minus[variable_index] -= h

    return (f(x_plus) - f(x_minus)) / (2 * h)


def gradient(f: Callable[[List[float]], float], x: List[float],
             h: float = 1e-5) -> List[float]:
    """
    Compute the gradient (vector of partial derivatives).

    The gradient is THE most important concept in machine learning!

    Used in:
    - Gradient descent
    - Backpropagation
    - All optimization algorithms

    Args:
        f: Multivariate function R^n → R
        x: Point at which to compute gradient
        h: Step size

    Returns:
        Gradient vector ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

    Example:
        >>> f = lambda x: x[0]**2 + x[1]**2  # f(x,y) = x² + y²
        >>> gradient(f, [1, 2])  # ∇f = [2x, 2y] at (1,2)
        [2.0, 4.0]
    """
    n = len(x)
    grad = []

    for i in range(n):
        grad.append(partial_derivative(f, x, i, h))

    return grad


def jacobian(f: Callable[[List[float]], List[float]], x: List[float],
             h: float = 1e-5) -> List[List[float]]:
    """
    Compute the Jacobian matrix of a vector-valued function.

    Used in:
    - Neural network layers (chain rule)
    - Nonlinear least squares
    - Newton's method for systems

    Args:
        f: Vector function R^n → R^m
        x: Point at which to compute Jacobian
        h: Step size

    Returns:
        Jacobian matrix J where J[i][j] = ∂f_i/∂x_j

    Shape: (m, n) where m = output dim, n = input dim

    Example:
        >>> f = lambda x: [x[0]**2, x[0]*x[1]]  # [x², xy]
        >>> jacobian(f, [1, 2])
        [[2.0, 0.0], [2.0, 1.0]]
    """
    n = len(x)
    f_x = f(x)
    m = len(f_x)

    J = [[0.0] * n for _ in range(m)]

    for i in range(m):
        # Partial derivative of i-th output with respect to each input
        def f_i(x_val):
            return f(x_val)[i]

        for j in range(n):
            J[i][j] = partial_derivative(f_i, x, j, h)

    return J


def hessian(f: Callable[[List[float]], float], x: List[float],
            h: float = 1e-5) -> List[List[float]]:
    """
    Compute the Hessian matrix (matrix of second partial derivatives).

    Used in:
    - Newton's method
    - Convexity analysis
    - Second-order optimization

    Args:
        f: Multivariate function R^n → R
        x: Point at which to compute Hessian
        h: Step size

    Returns:
        Hessian matrix H where H[i][j] = ∂²f/(∂x_i ∂x_j)

    The Hessian is symmetric for smooth functions.
    Positive definite Hessian → local minimum
    Negative definite Hessian → local maximum
    Indefinite Hessian → saddle point
    """
    n = len(x)
    H = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):  # Exploit symmetry
            if i == j:
                # Diagonal: pure second partial derivative
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h

                H[i][i] = (f(x_plus) - 2 * f(x) + f(x_minus)) / (h ** 2)
            else:
                # Off-diagonal: mixed partial derivative
                # ∂²f/(∂x_i ∂x_j) ≈ (f(x+h_i+h_j) - f(x+h_i) - f(x+h_j) + f(x)) / h²
                x_pp = x.copy()  # +h_i, +h_j
                x_pm = x.copy()  # +h_i, -h_j
                x_mp = x.copy()  # -h_i, +h_j
                x_mm = x.copy()  # -h_i, -h_j

                x_pp[i] += h
                x_pp[j] += h

                x_pm[i] += h
                x_pm[j] -= h

                x_mp[i] -= h
                x_mp[j] += h

                x_mm[i] -= h
                x_mm[j] -= h

                H[i][j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h ** 2)
                H[j][i] = H[i][j]  # Symmetry

    return H


def directional_derivative(f: Callable[[List[float]], float], x: List[float],
                          direction: List[float], h: float = 1e-5) -> float:
    """
    Compute directional derivative in a given direction.

    Used in:
    - Optimization (line search)
    - Sensitivity analysis
    - Trust region methods

    Args:
        f: Multivariate function
        x: Point at which to compute derivative
        direction: Direction vector (will be normalized)
        h: Step size

    Returns:
        Directional derivative D_v f(x) = ∇f(x) · v
    """
    # Normalize direction
    norm = math.sqrt(sum(d ** 2 for d in direction))
    if norm == 0:
        raise ValueError("Direction vector cannot be zero")

    unit_direction = [d / norm for d in direction]

    # f(x + h*v) - f(x - h*v) / (2h)
    x_plus = [x[i] + h * unit_direction[i] for i in range(len(x))]
    x_minus = [x[i] - h * unit_direction[i] for i in range(len(x))]

    return (f(x_plus) - f(x_minus)) / (2 * h)


def gradient_descent_step(f: Callable[[List[float]], float], x: List[float],
                          learning_rate: float = 0.01, h: float = 1e-5) -> List[float]:
    """
    Perform one step of gradient descent.

    This is the fundamental update rule in machine learning!

    Args:
        f: Function to minimize
        x: Current point
        learning_rate: Step size (α)
        h: Finite difference step

    Returns:
        New point: x_new = x - α * ∇f(x)

    Example:
        >>> f = lambda x: x[0]**2 + x[1]**2
        >>> x = [10, 10]
        >>> for _ in range(100):
        >>>     x = gradient_descent_step(f, x, learning_rate=0.1)
        >>> # x converges to [0, 0]
    """
    grad = gradient(f, x, h)
    x_new = [x[i] - learning_rate * grad[i] for i in range(len(x))]
    return x_new


def check_gradient(f: Callable[[List[float]], float],
                   analytical_grad: Callable[[List[float]], List[float]],
                   x: List[float], h: float = 1e-5, tolerance: float = 1e-5) -> bool:
    """
    Verify analytical gradient implementation against numerical gradient.

    Critical for debugging neural network implementations!

    Args:
        f: Function
        analytical_grad: Analytical gradient function
        x: Point to check
        h: Finite difference step
        tolerance: Maximum allowed error

    Returns:
        True if gradients match within tolerance

    Example:
        >>> f = lambda x: x[0]**2 + x[1]**2
        >>> grad_f = lambda x: [2*x[0], 2*x[1]]
        >>> check_gradient(f, grad_f, [1, 2])
        True
    """
    numerical_grad = gradient(f, x, h)
    analytical = analytical_grad(x)

    if len(numerical_grad) != len(analytical):
        return False

    for i in range(len(numerical_grad)):
        if abs(numerical_grad[i] - analytical[i]) > tolerance:
            print(f"Gradient mismatch at index {i}: "
                  f"numerical={numerical_grad[i]:.6f}, "
                  f"analytical={analytical[i]:.6f}")
            return False

    return True


def chain_rule(outer_derivative: float, inner_derivatives: List[float]) -> List[float]:
    """
    Apply the chain rule for composition of functions.

    Used in:
    - Backpropagation
    - Computing gradients through layers

    Args:
        outer_derivative: df/dy where y = g(x)
        inner_derivatives: [dg/dx₁, dg/dx₂, ...]

    Returns:
        [df/dx₁, df/dx₂, ...] = outer_derivative * inner_derivatives

    Example (neural network):
        >>> # y = relu(Wx + b), L = (y - target)²
        >>> # dL/dW = dL/dy * dy/d(Wx+b) * d(Wx+b)/dW
    """
    return [outer_derivative * inner_deriv for inner_deriv in inner_derivatives]


def laplacian(f: Callable[[List[float]], float], x: List[float],
              h: float = 1e-5) -> float:
    """
    Compute the Laplacian (sum of second partial derivatives).

    Used in:
    - Physics-informed neural networks
    - Diffusion equations
    - Heat equation

    Args:
        f: Multivariate function
        x: Point at which to compute Laplacian
        h: Step size

    Returns:
        Laplacian: ∇²f = ∂²f/∂x₁² + ∂²f/∂x₂² + ... + ∂²f/∂xₙ²
    """
    n = len(x)
    laplacian_value = 0.0

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h

        second_partial = (f(x_plus) - 2 * f(x) + f(x_minus)) / (h ** 2)
        laplacian_value += second_partial

    return laplacian_value


def divergence(F: Callable[[List[float]], List[float]], x: List[float],
               h: float = 1e-5) -> float:
    """
    Compute divergence of a vector field.

    Used in:
    - Fluid dynamics
    - Normalizing flows
    - Vector calculus

    Args:
        F: Vector field R^n → R^n
        x: Point at which to compute divergence
        h: Step size

    Returns:
        Divergence: div(F) = ∂F₁/∂x₁ + ∂F₂/∂x₂ + ... + ∂Fₙ/∂xₙ
    """
    n = len(x)
    div = 0.0

    for i in range(n):
        def F_i(x_val):
            return F(x_val)[i]

        div += partial_derivative(F_i, x, i, h)

    return div


def curl(F: Callable[[List[float]], List[float]], x: List[float],
         h: float = 1e-5) -> List[float]:
    """
    Compute curl of a 3D vector field.

    Args:
        F: Vector field R^3 → R^3
        x: Point at which to compute curl
        h: Step size

    Returns:
        Curl vector

    Formula:
        curl(F) = [∂F₃/∂x₂ - ∂F₂/∂x₃,
                   ∂F₁/∂x₃ - ∂F₃/∂x₁,
                   ∂F₂/∂x₁ - ∂F₁/∂x₂]
    """
    if len(x) != 3:
        raise ValueError("Curl is only defined for 3D vector fields")

    def F_1(x_val):
        return F(x_val)[0]

    def F_2(x_val):
        return F(x_val)[1]

    def F_3(x_val):
        return F(x_val)[2]

    curl_x = partial_derivative(F_3, x, 1, h) - partial_derivative(F_2, x, 2, h)
    curl_y = partial_derivative(F_1, x, 2, h) - partial_derivative(F_3, x, 0, h)
    curl_z = partial_derivative(F_2, x, 0, h) - partial_derivative(F_1, x, 1, h)

    return [curl_x, curl_y, curl_z]


def critical_points_1d(f: Callable[[float], float], a: float, b: float,
                       num_samples: int = 1000, tolerance: float = 1e-5) -> List[float]:
    """
    Find critical points (where f'(x) = 0) in an interval.

    Used in:
    - Optimization
    - Finding local minima/maxima

    Args:
        f: Function to analyze
        a: Left endpoint
        b: Right endpoint
        num_samples: Number of points to check
        tolerance: Threshold for considering derivative zero

    Returns:
        List of approximate critical points
    """
    critical = []
    dx = (b - a) / num_samples

    for i in range(num_samples):
        x = a + i * dx
        deriv = derivative(f, x)

        if abs(deriv) < tolerance:
            # Check if we haven't already found this point
            if not critical or abs(x - critical[-1]) > dx:
                critical.append(x)

    return critical


def is_convex(f: Callable[[float], float], a: float, b: float,
              num_samples: int = 100) -> bool:
    """
    Check if a function is convex on an interval.

    A function is convex if f''(x) >= 0 for all x.

    Used in:
    - Optimization theory
    - Checking loss function properties

    Args:
        f: Function to check
        a: Left endpoint
        b: Right endpoint
        num_samples: Number of points to check

    Returns:
        True if function appears convex
    """
    dx = (b - a) / num_samples

    for i in range(num_samples):
        x = a + i * dx
        second_deriv = second_derivative(f, x)

        if second_deriv < -1e-6:  # Small negative tolerance
            return False

    return True
