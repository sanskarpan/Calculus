"""
Unit tests for differentiation module
"""

import unittest
import math
from Calculus.differentiation import (
    derivative, second_derivative, partial_derivative,
    gradient, jacobian, hessian, directional_derivative,
    gradient_descent_step, check_gradient, laplacian,
    divergence, curl, is_convex
)


class TestBasicDerivatives(unittest.TestCase):
    """Test basic derivative computations."""

    def test_derivative_linear(self):
        """Test derivative of linear function."""
        f = lambda x: 2 * x + 3
        # f'(x) = 2
        self.assertAlmostEqual(derivative(f, 5), 2.0, places=4)

    def test_derivative_quadratic(self):
        """Test derivative of quadratic function."""
        f = lambda x: x ** 2
        # f'(x) = 2x
        self.assertAlmostEqual(derivative(f, 3), 6.0, places=4)

    def test_derivative_exp(self):
        """Test derivative of exponential."""
        f = lambda x: math.exp(x)
        # f'(x) = exp(x)
        x = 2
        self.assertAlmostEqual(derivative(f, x), math.exp(x), places=4)

    def test_derivative_sin(self):
        """Test derivative of sine."""
        f = lambda x: math.sin(x)
        # f'(x) = cos(x)
        x = math.pi / 4
        self.assertAlmostEqual(derivative(f, x), math.cos(x), places=4)

    def test_second_derivative(self):
        """Test second derivative."""
        f = lambda x: x ** 3
        # f''(x) = 6x
        x = 2
        self.assertAlmostEqual(second_derivative(f, x), 6 * x, places=3)


class TestMultivariateCalculus(unittest.TestCase):
    """Test multivariable calculus operations."""

    def test_partial_derivative(self):
        """Test partial derivative."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        # ∂f/∂x₀ = 2x₀, ∂f/∂x₁ = 2x₁
        point = [3, 4]
        self.assertAlmostEqual(partial_derivative(f, point, 0), 6.0, places=4)
        self.assertAlmostEqual(partial_derivative(f, point, 1), 8.0, places=4)

    def test_gradient(self):
        """Test gradient computation."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        point = [3, 4]
        grad = gradient(f, point)
        self.assertAlmostEqual(grad[0], 6.0, places=4)
        self.assertAlmostEqual(grad[1], 8.0, places=4)

    def test_gradient_complex(self):
        """Test gradient of complex function."""
        f = lambda x: x[0] * x[1] + x[0] ** 2
        # ∇f = [y + 2x, x]
        point = [2, 3]
        grad = gradient(f, point)
        self.assertAlmostEqual(grad[0], 7.0, places=4)  # 3 + 2*2
        self.assertAlmostEqual(grad[1], 2.0, places=4)


class TestJacobianHessian(unittest.TestCase):
    """Test Jacobian and Hessian computation."""

    def test_jacobian(self):
        """Test Jacobian matrix."""
        # f(x, y) = [x², xy]
        f = lambda x: [x[0] ** 2, x[0] * x[1]]
        point = [2, 3]
        J = jacobian(f, point)

        # J = [[2x, 0], [y, x]]
        self.assertAlmostEqual(J[0][0], 4.0, places=3)  # ∂(x²)/∂x = 2x = 4
        self.assertAlmostEqual(J[0][1], 0.0, places=3)  # ∂(x²)/∂y = 0
        self.assertAlmostEqual(J[1][0], 3.0, places=3)  # ∂(xy)/∂x = y = 3
        self.assertAlmostEqual(J[1][1], 2.0, places=3)  # ∂(xy)/∂y = x = 2

    def test_hessian_quadratic(self):
        """Test Hessian of quadratic function."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        # H = [[2, 0], [0, 2]]
        point = [1, 1]
        H = hessian(f, point)

        self.assertAlmostEqual(H[0][0], 2.0, places=2)
        self.assertAlmostEqual(H[1][1], 2.0, places=2)
        self.assertAlmostEqual(H[0][1], 0.0, places=2)


class TestVectorCalculus(unittest.TestCase):
    """Test vector calculus operations."""

    def test_directional_derivative(self):
        """Test directional derivative."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        point = [1, 0]
        direction = [1, 0]  # x-direction

        dir_deriv = directional_derivative(f, point, direction)
        self.assertAlmostEqual(dir_deriv, 2.0, places=4)  # ∂f/∂x = 2x = 2

    def test_laplacian(self):
        """Test Laplacian."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        # ∇²f = 2 + 2 = 4
        point = [1, 1]
        lap = laplacian(f, point)
        self.assertAlmostEqual(lap, 4.0, places=2)

    def test_divergence(self):
        """Test divergence of vector field."""
        # F(x,y) = [x, y]
        F = lambda x: [x[0], x[1]]
        # div(F) = 1 + 1 = 2
        point = [1, 1]
        div = divergence(F, point)
        self.assertAlmostEqual(div, 2.0, places=3)


class TestOptimization(unittest.TestCase):
    """Test optimization utilities."""

    def test_gradient_descent_step(self):
        """Test gradient descent step."""
        f = lambda x: (x[0] - 2) ** 2 + (x[1] - 3) ** 2
        # Minimum at (2, 3)

        x = [0, 0]
        for _ in range(100):
            x = gradient_descent_step(f, x, learning_rate=0.1)

        # Should converge to (2, 3)
        self.assertAlmostEqual(x[0], 2.0, places=1)
        self.assertAlmostEqual(x[1], 3.0, places=1)

    def test_check_gradient(self):
        """Test gradient checking."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        grad_f = lambda x: [2 * x[0], 2 * x[1]]

        point = [1, 2]
        self.assertTrue(check_gradient(f, grad_f, point))

    def test_is_convex(self):
        """Test convexity check."""
        # Convex function
        f_convex = lambda x: x ** 2
        self.assertTrue(is_convex(f_convex, -2, 2))

        # Non-convex function
        f_nonconvex = lambda x: -x ** 2
        self.assertFalse(is_convex(f_nonconvex, -2, 2))


if __name__ == '__main__':
    unittest.main()
