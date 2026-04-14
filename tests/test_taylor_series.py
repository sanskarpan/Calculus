"""
Unit tests for Taylor series module
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest
import math
from Calculus.taylor_series import (
    taylor_series,
    taylor_series_numerical,
    exp_taylor,
    sin_taylor,
    cos_taylor,
    log_taylor,
    linear_approximation,
    quadratic_approximation,
    approximate_function,
    multivariate_taylor_first_order,
    multivariate_taylor_second_order,
)


class TestAnalyticalTaylors(unittest.TestCase):
    """Test analytical Taylor series implementations."""

    def test_exp_taylor(self):
        """Test exponential Taylor series."""
        for x in [0.1, 0.5, 1.0]:
            for n in [5, 10, 20]:
                result = exp_taylor(x, n)
                expected = math.exp(x)
                self.assertAlmostEqual(
                    result,
                    expected,
                    places=5,
                    msg=f"exp_taylor({x}, {n}) = {result}, expected {expected}",
                )

    def test_sin_taylor(self):
        """Test sine Taylor series."""
        test_cases = [
            (0.0, 0.0),
            (0.5, math.sin(0.5)),
            (math.pi / 6, 0.5),  # sin(π/6) = 0.5
        ]
        for x, expected in test_cases:
            result = sin_taylor(x, n=10)
            self.assertAlmostEqual(
                result,
                expected,
                places=5,
                msg=f"sin_taylor({x}) = {result}, expected {expected}",
            )

    def test_cos_taylor(self):
        """Test cosine Taylor series."""
        test_cases = [
            (0.0, 1.0),
            (0.5, math.cos(0.5)),
            (math.pi / 3, 0.5),  # cos(π/3) = 0.5
        ]
        for x, expected in test_cases:
            result = cos_taylor(x, n=10)
            self.assertAlmostEqual(
                result,
                expected,
                places=5,
                msg=f"cos_taylor({x}) = {result}, expected {expected}",
            )

    def test_log_taylor(self):
        """Test logarithm Taylor series (valid for 0 < x <= 2)."""
        test_cases = [
            (0.5, math.log(0.5)),
            (1.0, 0.0),
            (1.5, math.log(1.5)),
        ]
        for x, expected in test_cases:
            result = log_taylor(x, n=20)
            self.assertAlmostEqual(
                result,
                expected,
                places=4,
                msg=f"log_taylor({x}) = {result}, expected {expected}",
            )

    def test_log_taylor_invalid_domain(self):
        """Test log_taylor raises for invalid inputs."""
        with self.assertRaises(ValueError):
            log_taylor(0)
        with self.assertRaises(ValueError):
            log_taylor(-1)
        with self.assertRaises(ValueError):
            log_taylor(3)


class TestNumericalTaylors(unittest.TestCase):
    """Test numerical Taylor series approximations."""

    def test_linear_approximation(self):
        """Test linear approximation for known polynomial."""
        f = lambda x: 2 * x + 3
        result = linear_approximation(f, 1, 1)  # f(1) + f'(1)*(1-1) = 5 + 2*0 = 5
        self.assertAlmostEqual(result, 5.0, places=3)

    def test_linear_approximation_exp(self):
        """Test linear approximation for exp."""
        f = math.exp
        result = linear_approximation(f, 0, 0.1)  # ≈ 1 + 1*0.1 = 1.1
        expected = 1.105170  # actual e^0.1
        self.assertAlmostEqual(result, expected, places=2)

    def test_quadratic_approximation(self):
        """Test quadratic approximation for x^2."""
        f = lambda x: x**2
        # Taylor at a=0: f(0)=0, f'(0)=0, f''(0)=2
        # Quadratic: 0 + 0*(x-0) + 2*(x-0)^2/2 = x^2
        result = quadratic_approximation(f, 0, 0.5)
        self.assertAlmostEqual(result, 0.25, places=4)

    def test_quadratic_approximation_exp(self):
        """Test quadratic approximation for exp."""
        f = math.exp
        # At a=0: f(0)=1, f'(0)=1, f''(0)=1
        # Approx: 1 + 1*x + 1*x^2/2 = 1 + x + x^2/2
        result = quadratic_approximation(f, 0, 0.1)  # = 1 + 0.1 + 0.005 = 1.105
        expected = 1.105170
        self.assertAlmostEqual(result, expected, places=3)

    def test_taylor_series_numerical_basic(self):
        """Test basic numerical Taylor series."""
        f = lambda x: x**2
        # Taylor at a=0: 0 + 0*x + 2*x^2/2! = x^2
        result = taylor_series_numerical(f, 0, 0.5, 3)
        self.assertAlmostEqual(result, 0.25, places=2)

    def test_taylor_series_numerical_exp(self):
        """Test numerical Taylor for exp."""
        f = math.exp
        result = taylor_series_numerical(f, 0, 0.2, 5)
        expected = math.exp(0.2)
        self.assertAlmostEqual(result, expected, places=3)


class TestApproximateFunction(unittest.TestCase):
    """Test polynomial approximation function."""

    def test_approximate_constant(self):
        """Test approximation of constant function."""
        f = lambda x: 5.0
        approx = approximate_function(f, [0, 1], degree=0)
        self.assertAlmostEqual(approx(0.5), 5.0, places=3)

    def test_approximate_linear(self):
        """Test approximation of linear function."""
        f = lambda x: 3 * x + 2
        approx = approximate_function(f, [0, 1], degree=1)
        # Should be exact for linear
        self.assertAlmostEqual(approx(0.5), 3 * 0.5 + 2, places=2)

    def test_approximate_quadratic(self):
        """Test approximation of quadratic function."""
        f = lambda x: x**2
        approx = approximate_function(f, [-1, 1], degree=2)
        # Should be exact for quadratic
        self.assertAlmostEqual(approx(0.5), 0.25, places=3)
        self.assertAlmostEqual(approx(0.0), 0.0, places=3)


class TestMultivariateTaylors(unittest.TestCase):
    """Test multivariate Taylor approximations."""

    def test_multivariate_first_order(self):
        """Test multivariate linear approximation."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        a = [1.0, 1.0]
        x = [1.1, 1.1]
        result = multivariate_taylor_first_order(f, a, x)
        # f(a) + grad·(x-a) = 2 + [2,2]·[0.1,0.1] = 2 + 0.4 = 2.4
        self.assertAlmostEqual(result, 2.4, places=2)

    def test_multivariate_second_order(self):
        """Test multivariate quadratic approximation."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        a = [1.0, 1.0]
        x = [1.1, 1.1]
        result = multivariate_taylor_second_order(f, a, x)
        # f(a) + grad·(x-a) + 0.5*(x-a)ᵀH(x-a)
        # = 2 + [2,2]·[0.1,0.1] + 0.5*[0.1,0.1]·[4 0; 0 4]·[0.1;0.1]
        # = 2 + 0.4 + 0.5 * 0.04 = 2.42
        self.assertAlmostEqual(result, 2.42, places=2)


class TestTaylorsEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_taylor_series_numerical_n_equals_1(self):
        """Test n=1 case."""
        f = lambda x: x**2
        result = taylor_series_numerical(f, 0, 0.5, 1)
        # Only linear term: f(0) + f'(0)*x = 0 + 0*0.5 = 0
        self.assertAlmostEqual(result, 0.0, places=2)

    def test_taylor_series_numerical_n_equals_2(self):
        """Test n=2 case."""
        f = lambda x: x**2
        result = taylor_series_numerical(f, 0, 0.5, 2)
        # Linear + quadratic: 0 + 0*0.5 + 2*0.5²/2 = 0.25
        self.assertAlmostEqual(result, 0.25, places=2)


if __name__ == "__main__":
    unittest.main()
