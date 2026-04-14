"""
Unit tests for integration module
"""

import unittest
import math
from Calculus.integration import (
    riemann_sum,
    trapezoidal_rule,
    simpsons_rule,
    monte_carlo_integration,
    double_integral,
    expectation,
    adaptive_integration,
)


class TestBasicIntegration(unittest.TestCase):
    """Test basic integration methods."""

    def test_riemann_constant(self):
        """Test Riemann sum for constant function."""
        f = lambda x: 5
        # ∫₀² 5 dx = 10
        result = riemann_sum(f, 0, 2, n=1000)
        self.assertAlmostEqual(result, 10.0, places=1)

    def test_riemann_linear(self):
        """Test Riemann sum for linear function."""
        f = lambda x: x
        # ∫₀² x dx = x²/2 |₀² = 2
        result = riemann_sum(f, 0, 2, n=1000)
        self.assertAlmostEqual(result, 2.0, places=1)

    def test_trapezoidal_quadratic(self):
        """Test trapezoidal rule for quadratic."""
        f = lambda x: x**2
        # ∫₀¹ x² dx = x³/3 |₀¹ = 1/3
        result = trapezoidal_rule(f, 0, 1, n=1000)
        self.assertAlmostEqual(result, 1 / 3, places=3)

    def test_simpsons_rule(self):
        """Test Simpson's rule (most accurate)."""
        f = lambda x: x**3
        # ∫₀² x³ dx = x⁴/4 |₀² = 4
        result = simpsons_rule(f, 0, 2, n=1000)
        self.assertAlmostEqual(result, 4.0, places=3)

    def test_simpsons_exp(self):
        """Test Simpson's rule on exponential."""
        f = lambda x: math.exp(x)
        # ∫₀¹ eˣ dx = e - 1
        result = simpsons_rule(f, 0, 1, n=1000)
        self.assertAlmostEqual(result, math.e - 1, places=3)

    def test_adaptive_integration_sin(self):
        """Test adaptive Simpson integration on sin."""
        f = math.sin
        # ∫₀^π sin(x) dx = 2
        result = adaptive_integration(f, 0.0, math.pi, tolerance=1e-8, max_depth=20)
        self.assertAlmostEqual(result, 2.0, places=6)


class TestMonteCarloIntegration(unittest.TestCase):
    """Test Monte Carlo integration."""

    def test_monte_carlo_simple(self):
        """Test Monte Carlo on simple function."""
        f = lambda x: x
        # ∫₀¹ x dx = 0.5
        result, _ = monte_carlo_integration(f, 0, 1, n_samples=100000)
        self.assertAlmostEqual(result, 0.5, places=1)

    def test_monte_carlo_quadratic(self):
        """Test Monte Carlo on quadratic."""
        f = lambda x: x**2
        # ∫₀¹ x² dx = 1/3
        result, _ = monte_carlo_integration(f, 0, 1, n_samples=100000)
        self.assertAlmostEqual(result, 1 / 3, places=1)


class TestMultipleIntegrals(unittest.TestCase):
    """Test double and triple integrals."""

    def test_double_integral_constant(self):
        """Test double integral of constant."""
        f = lambda x, y: 1
        # ∫₀¹∫₀¹ 1 dx dy = 1
        result = double_integral(f, (0, 1), (0, 1), nx=100, ny=100)
        self.assertAlmostEqual(result, 1.0, places=2)

    def test_double_integral_xy(self):
        """Test double integral of xy."""
        f = lambda x, y: x * y
        # ∫₀¹∫₀¹ xy dx dy = 1/4
        result = double_integral(f, (0, 1), (0, 1), nx=100, ny=100)
        self.assertAlmostEqual(result, 0.25, places=2)


class TestProbabilityIntegration(unittest.TestCase):
    """Test integration for probability."""

    def test_expectation_uniform(self):
        """Test expectation for uniform distribution."""
        f = lambda x: x
        pdf = lambda x: 1.0  # Uniform on [0, 1]
        # E[X] = 0.5 for Uniform(0,1)
        result = expectation(f, pdf, 0, 1, n=1000)
        self.assertAlmostEqual(result, 0.5, places=2)

    def test_expectation_quadratic(self):
        """Test E[X²] for uniform distribution."""
        f = lambda x: x**2
        pdf = lambda x: 1.0
        # E[X²] = 1/3 for Uniform(0,1)
        result = expectation(f, pdf, 0, 1, n=1000)
        self.assertAlmostEqual(result, 1 / 3, places=2)


if __name__ == "__main__":
    unittest.main()
