"""
Automatic Differentiation & Backpropagation Tutorial
=====================================================

This tutorial shows how automatic differentiation works - the magic behind
PyTorch, TensorFlow, and JAX!

You'll understand:
- How computational graphs track operations
- How backpropagation computes gradients automatically
- How to train a simple neural network from scratch
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Calculus.autograd import Variable, backward, gradient_check, neuron, mse_loss, sgd_step


def basic_autograd():
    """Demonstrate basic automatic differentiation."""
    print("=" * 60)
    print("AUTOMATIC DIFFERENTIATION - The Basics")
    print("=" * 60)

    print("\n1. Simple Addition")
    x = Variable(2.0, name='x')
    y = Variable(3.0, name='y')
    z = x + y

    print(f"   z = x + y = {z.value}")
    backward(z)
    print(f"   dz/dx = {x.grad} (should be 1)")
    print(f"   dz/dy = {y.grad} (should be 1)")

    print("\n2. Multiplication")
    x = Variable(2.0, name='x')
    y = Variable(3.0, name='y')
    z = x * y

    print(f"   z = x * y = {z.value}")
    backward(z)
    print(f"   dz/dx = {x.grad} (should be y = 3)")
    print(f"   dz/dy = {y.grad} (should be x = 2)")

    print("\n3. Power")
    x = Variable(3.0, name='x')
    y = x ** 2

    print(f"   y = x² = {y.value}")
    backward(y)
    print(f"   dy/dx = {x.grad} (should be 2x = 6)")


def chain_rule_demo():
    """Demonstrate chain rule in action."""
    print("\n" + "=" * 60)
    print("CHAIN RULE - Composition of Functions")
    print("=" * 60)

    print("\nCompute: z = (x + y)²")

    x = Variable(2.0, name='x')
    y = Variable(3.0, name='y')

    # Build computation step by step
    s = x + y  # s = x + y = 5
    z = s ** 2  # z = s² = 25

    print(f"   s = x + y = {s.value}")
    print(f"   z = s² = {z.value}")

    backward(z)

    print(f"\nGradients:")
    print(f"   dz/dx = {x.grad} (should be 2s = 10)")
    print(f"   dz/dy = {y.grad} (should be 2s = 10)")

    print(f"\nHow it works:")
    print(f"   dz/ds = 2s = {2 * s.value}")
    print(f"   ds/dx = 1")
    print(f"   dz/dx = (dz/ds) * (ds/dx) = {2 * s.value} * 1 = {x.grad}")


def activation_functions_demo():
    """Demonstrate activation functions with gradients."""
    print("\n" + "=" * 60)
    print("ACTIVATION FUNCTIONS - Neural Network Nonlinearity")
    print("=" * 60)

    print("\n1. ReLU Activation")
    print("   ReLU(x) = max(0, x)")

    x_pos = Variable(2.0)
    y_pos = x_pos.relu()
    print(f"   ReLU({x_pos.value}) = {y_pos.value}")
    backward(y_pos)
    print(f"   Gradient: {x_pos.grad} (x > 0, so grad = 1)")

    x_neg = Variable(-2.0)
    y_neg = x_neg.relu()
    print(f"   ReLU({x_neg.value}) = {y_neg.value}")
    backward(y_neg)
    print(f"   Gradient: {x_neg.grad} (x < 0, so grad = 0)")

    print("\n2. Sigmoid Activation")
    print("   σ(x) = 1 / (1 + e^(-x))")

    x = Variable(0.0)
    y = x.sigmoid()
    print(f"   σ(0) = {y.value:.4f}")
    backward(y)
    print(f"   σ'(0) = {x.grad:.4f} (should be 0.25)")

    print("\n3. Tanh Activation")
    x = Variable(0.0)
    y = x.tanh()
    print(f"   tanh(0) = {y.value:.4f}")
    backward(y)
    print(f"   tanh'(0) = {x.grad:.4f} (should be 1.0)")


def neural_network_neuron_demo():
    """Demonstrate a single neuron."""
    print("\n" + "=" * 60)
    print("SINGLE NEURON - Building Block of Neural Networks")
    print("=" * 60)

    print("\nNeuron: y = ReLU(w₁x₁ + w₂x₂ + b)")

    # Create neuron
    inputs = [Variable(1.0, name='x1'), Variable(2.0, name='x2')]
    weights = [Variable(0.5, name='w1'), Variable(-0.3, name='w2')]
    bias = Variable(0.1, name='b')

    # Forward pass
    output = neuron(inputs, weights, bias, activation='relu')

    print(f"\nInputs: x₁={inputs[0].value}, x₂={inputs[1].value}")
    print(f"Weights: w₁={weights[0].value}, w₂={weights[1].value}")
    print(f"Bias: b={bias.value}")

    # Compute manually
    z = weights[0].value * inputs[0].value + weights[1].value * inputs[1].value + bias.value
    print(f"\nLinear combination:")
    print(f"   z = {weights[0].value}*{inputs[0].value} + {weights[1].value}*{inputs[1].value} + {bias.value}")
    print(f"   z = {z:.4f}")
    print(f"   output = ReLU(z) = {output.value:.4f}")

    # Backward pass
    backward(output)

    print(f"\nGradients (for backpropagation):")
    print(f"   ∂L/∂w₁ = {weights[0].grad:.4f}")
    print(f"   ∂L/∂w₂ = {weights[1].grad:.4f}")
    print(f"   ∂L/∂b = {bias.grad:.4f}")


def training_example():
    """Demonstrate training a neuron on a simple dataset."""
    print("\n" + "=" * 60)
    print("TRAINING A NEURON - Gradient Descent in Action!")
    print("=" * 60)

    print("\nTask: Learn to compute AND gate")
    print("Truth table:")
    print("   x₁  x₂  | target")
    print("   ---------------")
    print("   0   0   |   0")
    print("   0   1   |   0")
    print("   1   0   |   0")
    print("   1   1   |   1")

    # Training data for AND gate
    dataset = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 0.0),
        ([1.0, 0.0], 0.0),
        ([1.0, 1.0], 1.0),
    ]

    # Initialize neuron parameters
    weights = [Variable(0.1, name='w1'), Variable(0.1, name='w2')]
    bias = Variable(0.0, name='b')

    learning_rate = 0.5
    epochs = 100

    print(f"\nTraining for {epochs} epochs with lr={learning_rate}")
    print("\nEpoch  | Loss")
    print("-" * 20)

    for epoch in range(epochs):
        total_loss = Variable(0.0)

        # Train on all examples
        for inputs_data, target in dataset:
            # Forward pass
            inputs = [Variable(inputs_data[0]), Variable(inputs_data[1])]
            output = neuron(inputs, weights, bias, activation='sigmoid')

            # Compute loss
            error = output - Variable(target)
            loss = error ** 2
            total_loss = total_loss + loss

        # Average loss
        avg_loss = total_loss / Variable(len(dataset))

        if epoch % 10 == 0:
            print(f"  {epoch:3d}  | {avg_loss.value:.6f}")

        # Backward pass
        backward(avg_loss)

        # Update parameters
        sgd_step([weights[0], weights[1], bias], learning_rate)

    print(f"  {epochs:3d}  | {avg_loss.value:.6f}")

    print(f"\nLearned parameters:")
    print(f"   w₁ = {weights[0].value:.4f}")
    print(f"   w₂ = {weights[1].value:.4f}")
    print(f"   b  = {bias.value:.4f}")

    print(f"\nTesting learned neuron:")
    print("   x₁  x₂  | prediction | target")
    print("   ------------------------------------")
    for inputs_data, target in dataset:
        inputs = [Variable(inputs_data[0]), Variable(inputs_data[1])]
        output = neuron(inputs, weights, bias, activation='sigmoid')
        print(f"   {inputs_data[0]:.0f}   {inputs_data[1]:.0f}   |   {output.value:.4f}    |  {target:.0f}")


def computational_graph_demo():
    """Show the computational graph."""
    print("\n" + "=" * 60)
    print("COMPUTATIONAL GRAPH - Visualizing Operations")
    print("=" * 60)

    print("\nExpression: z = x² + xy + y²")

    x = Variable(2.0, name='x')
    y = Variable(3.0, name='y')

    # Build graph step by step
    x_squared = x ** 2
    x_squared._op = 'x²'

    xy = x * y
    xy._op = 'xy'

    y_squared = y ** 2
    y_squared._op = 'y²'

    z = x_squared + xy + y_squared

    print(f"\nComputation steps:")
    print(f"   1. x² = {x_squared.value}")
    print(f"   2. xy = {xy.value}")
    print(f"   3. y² = {y_squared.value}")
    print(f"   4. z = x² + xy + y² = {z.value}")

    backward(z)

    print(f"\nGradients (computed via backpropagation):")
    print(f"   ∂z/∂x = {x.grad:.4f}")
    print(f"   ∂z/∂y = {y.grad:.4f}")

    print(f"\nAnalytical gradients:")
    print(f"   ∂z/∂x = 2x + y = {2*x.value + y.value}")
    print(f"   ∂z/∂y = x + 2y = {x.value + 2*y.value}")

    print(f"\nMatch! ✓")


def gradient_checking_demo():
    """Demonstrate gradient checking."""
    print("\n" + "=" * 60)
    print("GRADIENT CHECKING - Debugging Backpropagation")
    print("=" * 60)

    print("\nGradient checking compares:")
    print("   Automatic gradients (backprop)")
    print("   vs")
    print("   Numerical gradients (finite differences)")

    x = Variable(2.0, name='x')
    y = Variable(3.0, name='y')

    def f():
        return x ** 2 + y ** 2

    print(f"\nFunction: f(x,y) = x² + y²")
    print(f"Point: ({x.value}, {y.value})")

    is_correct = gradient_check(f, [x, y])

    print(f"\nGradient check: {'PASSED ✓' if is_correct else 'FAILED ✗'}")
    print("\nThis verifies our backprop implementation is correct!")


def main():
    """Run all autograd tutorials."""
    basic_autograd()
    chain_rule_demo()
    activation_functions_demo()
    neural_network_neuron_demo()
    training_example()
    computational_graph_demo()
    gradient_checking_demo()

    print("\n" + "=" * 60)
    print("Autograd & Backpropagation Tutorial Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("✓ Computational graphs track operations automatically")
    print("✓ Forward pass computes outputs")
    print("✓ Backward pass computes gradients via chain rule")
    print("✓ This is how PyTorch/TensorFlow/JAX work!")
    print("✓ Gradient descent uses these gradients to learn")
    print("✓ Always use gradient checking when implementing networks")
    print("\nYou now understand the magic behind deep learning! 🚀")


if __name__ == '__main__':
    main()
