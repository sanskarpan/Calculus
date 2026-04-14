"""
Unit tests for automatic differentiation (autograd) module
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
import math
from Calculus.autograd import (
    Variable, backward, gradient_check, neuron, mse_loss
)


class TestVariableBasics(unittest.TestCase):
    """Test Variable basic operations."""

    def test_addition(self):
        """Test addition with gradient."""
        x = Variable(2.0, name='x')
        y = Variable(3.0, name='y')
        z = x + y

        self.assertEqual(z.value, 5.0)

        backward(z)
        self.assertEqual(x.grad, 1.0)
        self.assertEqual(y.grad, 1.0)

    def test_multiplication(self):
        """Test multiplication with gradient."""
        x = Variable(2.0, name='x')
        y = Variable(3.0, name='y')
        z = x * y

        self.assertEqual(z.value, 6.0)

        backward(z)
        self.assertEqual(x.grad, 3.0)  # dz/dx = y
        self.assertEqual(y.grad, 2.0)  # dz/dy = x

    def test_power(self):
        """Test power operation."""
        x = Variable(3.0, name='x')
        y = x ** 2

        self.assertEqual(y.value, 9.0)

        backward(y)
        self.assertEqual(x.grad, 6.0)  # dy/dx = 2x = 6


class TestCompositeOperations(unittest.TestCase):
    """Test composite operations (chain rule)."""

    def test_chain_rule_simple(self):
        """Test simple chain rule."""
        x = Variable(2.0, name='x')
        y = x ** 2  # y = x²
        z = y + x  # z = x² + x

        self.assertEqual(z.value, 6.0)  # 4 + 2

        backward(z)
        self.assertEqual(x.grad, 5.0)  # dz/dx = 2x + 1 = 5

    def test_chain_rule_complex(self):
        """Test more complex chain rule."""
        x = Variable(2.0, name='x')
        y = x * x * x  # y = x³

        self.assertEqual(y.value, 8.0)

        backward(y)
        self.assertAlmostEqual(x.grad, 12.0, places=4)  # dy/dx = 3x² = 12


class TestActivationFunctions(unittest.TestCase):
    """Test activation functions and their gradients."""

    def test_relu(self):
        """Test ReLU activation."""
        # Positive input
        x = Variable(2.0)
        y = x.relu()
        self.assertEqual(y.value, 2.0)

        backward(y)
        self.assertEqual(x.grad, 1.0)

        # Negative input
        x = Variable(-2.0)
        y = x.relu()
        self.assertEqual(y.value, 0.0)

    def test_sigmoid(self):
        """Test sigmoid activation."""
        x = Variable(0.0)
        y = x.sigmoid()
        self.assertAlmostEqual(y.value, 0.5, places=5)

        backward(y)
        # At x=0: σ'(0) = σ(0)(1-σ(0)) = 0.5 * 0.5 = 0.25
        self.assertAlmostEqual(x.grad, 0.25, places=5)

    def test_tanh(self):
        """Test tanh activation."""
        x = Variable(0.0)
        y = x.tanh()
        self.assertAlmostEqual(y.value, 0.0, places=5)

        backward(y)
        # At x=0: tanh'(0) = 1
        self.assertAlmostEqual(x.grad, 1.0, places=5)


class TestNeuralNetworkComponents(unittest.TestCase):
    """Test neural network building blocks."""

    def test_neuron(self):
        """Test single neuron computation."""
        inputs = [Variable(1.0), Variable(2.0)]
        weights = [Variable(0.5), Variable(-0.3)]
        bias = Variable(0.1)

        output = neuron(inputs, weights, bias, activation='relu')

        # Forward: 0.5*1 + (-0.3)*2 + 0.1 = 0.5 - 0.6 + 0.1 = 0
        # ReLU(0) = 0
        self.assertAlmostEqual(output.value, 0.0, places=5)

    def test_mse_loss(self):
        """Test MSE loss computation."""
        predictions = [Variable(2.5), Variable(3.0)]
        targets = [2.0, 3.5]

        loss = mse_loss(predictions, targets)

        # MSE = ((2.5-2)² + (3.0-3.5)²) / 2 = (0.25 + 0.25) / 2 = 0.25
        self.assertAlmostEqual(loss.value, 0.25, places=5)

        backward(loss)
        # Gradients should be computed
        self.assertNotEqual(predictions[0].grad, 0.0)


class TestGradientChecking(unittest.TestCase):
    """Test gradient verification."""

    def test_gradient_check_simple(self):
        """Test gradient checking for simple function."""
        x = Variable(2.0, name='x')
        y = Variable(3.0, name='y')

        def f():
            return x ** 2 + y ** 2

        self.assertTrue(gradient_check(f, [x, y]))

    def test_gradient_check_product(self):
        """Test gradient checking for product."""
        x = Variable(2.0, name='x')
        y = Variable(3.0, name='y')

        def f():
            return x * y + x

        self.assertTrue(gradient_check(f, [x, y]))


class TestBackpropagation(unittest.TestCase):
    """Test backpropagation through networks."""

    def test_two_layer_network(self):
        """Test backpropagation through two layers."""
        # Input
        x = Variable(1.0, name='x')

        # First layer
        w1 = Variable(2.0, name='w1')
        h = x * w1  # h = 2.0

        # Second layer
        w2 = Variable(3.0, name='w2')
        y = h * w2  # y = 6.0

        self.assertEqual(y.value, 6.0)

        backward(y)

        # dy/dw2 = h = 2.0
        self.assertEqual(w2.grad, 2.0)

        # dy/dw1 = dy/dh * dh/dw1 = w2 * x = 3.0 * 1.0 = 3.0
        self.assertEqual(w1.grad, 3.0)

        # dy/dx = dy/dh * dh/dx = w2 * w1 = 6.0
        self.assertEqual(x.grad, 6.0)


if __name__ == '__main__':
    unittest.main()
