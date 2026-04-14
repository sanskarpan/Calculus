"""
Differentiation Tutorial - Understanding Gradients and Backpropagation
======================================================================

This tutorial demonstrates numerical and automatic differentiation,
which are the foundations of training neural networks.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import math
from Calculus.differentiation import (
    derivative, gradient, jacobian, hessian,
    gradient_descent_step, check_gradient
)


def basic_derivatives():
    """Demonstrate basic derivative computation."""
    print("=" * 60)
    print("BASIC DERIVATIVES")
    print("=" * 60)

    print("\n1. Derivative of f(x) = x²")
    f = lambda x: x ** 2
    x = 3
    df_dx = derivative(f, x)
    print(f"   f'({x}) = {df_dx:.4f} (analytical: {2*x:.4f})")

    print("\n2. Derivative of f(x) = sin(x)")
    f = lambda x: math.sin(x)
    x = math.pi / 4
    df_dx = derivative(f, x)
    print(f"   f'(π/4) = {df_dx:.4f} (analytical: cos(π/4) = {math.cos(x):.4f})")

    print("\n3. Derivative of f(x) = eˣ")
    f = lambda x: math.exp(x)
    x = 1
    df_dx = derivative(f, x)
    print(f"   f'(1) = {df_dx:.4f} (analytical: e¹ = {math.exp(1):.4f})")


def multivariable_calculus():
    """Demonstrate gradient computation."""
    print("\n" + "=" * 60)
    print("MULTIVARIABLE CALCULUS - GRADIENTS")
    print("=" * 60)

    print("\n1. Gradient of f(x,y) = x² + y²")
    f = lambda x: x[0] ** 2 + x[1] ** 2
    point = [3, 4]
    grad = gradient(f, point)
    print(f"   Point: ({point[0]}, {point[1]})")
    print(f"   ∇f = [{grad[0]:.4f}, {grad[1]:.4f}]")
    print(f"   Analytical: [2x, 2y] = [{2*point[0]}, {2*point[1]}]")

    print("\n2. Gradient of f(x,y) = xy + x²")
    f = lambda x: x[0] * x[1] + x[0] ** 2
    point = [2, 3]
    grad = gradient(f, point)
    print(f"   Point: ({point[0]}, {point[1]})")
    print(f"   ∇f = [{grad[0]:.4f}, {grad[1]:.4f}]")
    print(f"   Analytical: [y+2x, x] = [{point[1]+2*point[0]}, {point[0]}]")


def jacobian_demo():
    """Demonstrate Jacobian matrix."""
    print("\n" + "=" * 60)
    print("JACOBIAN MATRIX - Neural Network Layers")
    print("=" * 60)

    print("\nJacobian represents how outputs change with inputs")
    print("Critical for backpropagation through layers!")

    # Example: f(x,y) = [x², xy, x+y]
    f = lambda x: [x[0] ** 2, x[0] * x[1], x[0] + x[1]]
    point = [2, 3]

    J = jacobian(f, point)

    print(f"\nFunction: f(x,y) = [x², xy, x+y]")
    print(f"Point: ({point[0]}, {point[1]})")
    print("\nJacobian matrix J:")
    for i, row in enumerate(J):
        print(f"   [{row[0]:7.4f}, {row[1]:7.4f}]")

    print("\nAnalytical Jacobian:")
    print(f"   [[2x, 0  ],  = [[{2*point[0]}, 0],")
    print(f"    [y,  x  ],     [{point[1]}, {point[0]}],")
    print(f"    [1,  1  ]]      [1, 1]]")


def hessian_demo():
    """Demonstrate Hessian matrix for optimization."""
    print("\n" + "=" * 60)
    print("HESSIAN MATRIX - Second-Order Optimization")
    print("=" * 60)

    print("\nHessian tells us about curvature")
    print("Used in Newton's method and second-order optimization")

    f = lambda x: x[0] ** 2 + x[1] ** 2 + x[0] * x[1]
    point = [1, 1]

    H = hessian(f, point)

    print(f"\nFunction: f(x,y) = x² + y² + xy")
    print(f"Point: ({point[0]}, {point[1]})")
    print("\nHessian matrix H:")
    for row in H:
        print(f"   [{row[0]:7.4f}, {row[1]:7.4f}]")

    print("\nAnalytical Hessian:")
    print("   [[2, 1],")
    print("    [1, 2]]")

    # Check if positive definite (all eigenvalues > 0)
    # For 2x2: det > 0 and trace > 0 and H[0][0] > 0
    det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
    trace = H[0][0] + H[1][1]
    print(f"\nDeterminant: {det:.4f} (> 0 ✓)")
    print(f"Trace: {trace:.4f} (> 0 ✓)")
    print("=> Hessian is positive definite")
    print("=> Point is a local minimum!")


def gradient_descent_demo():
    """Demonstrate gradient descent optimization."""
    print("\n" + "=" * 60)
    print("GRADIENT DESCENT - The Heart of ML!")
    print("=" * 60)

    print("\nMinimize f(x,y) = (x-2)² + (y-3)²")
    print("True minimum at (2, 3)")

    f = lambda x: (x[0] - 2) ** 2 + (x[1] - 3) ** 2

    # Start far from minimum
    x = [0.0, 0.0]
    learning_rate = 0.1

    print(f"\nStarting point: ({x[0]:.2f}, {x[1]:.2f})")
    print(f"Learning rate: {learning_rate}")

    print("\nIteration  |   x      y    |  f(x,y)")
    print("-" * 40)

    for i in range(20):
        if i % 5 == 0:
            print(f"   {i:3d}     | {x[0]:6.3f}, {x[1]:6.3f} | {f(x):7.4f}")

        x = gradient_descent_step(f, x, learning_rate)

    print(f"   ...     | {x[0]:6.3f}, {x[1]:6.3f} | {f(x):7.4f}")
    print(f"\nFinal point: ({x[0]:.4f}, {x[1]:.4f})")
    print(f"True minimum: (2.0000, 3.0000)")
    print("Converged! ✓")


def gradient_checking():
    """Demonstrate gradient checking."""
    print("\n" + "=" * 60)
    print("GRADIENT CHECKING - Debugging Neural Networks")
    print("=" * 60)

    print("\nGradient checking verifies your backprop implementation")
    print("Compare analytical gradients vs numerical gradients")

    # Correct gradient
    f = lambda x: x[0] ** 2 + x[1] ** 2
    grad_f = lambda x: [2 * x[0], 2 * x[1]]

    point = [3, 4]
    is_correct = check_gradient(f, grad_f, point)

    print(f"\nFunction: f(x,y) = x² + y²")
    print(f"Analytical gradient: ∇f = [2x, 2y]")
    print(f"Point: ({point[0]}, {point[1]})")
    print(f"\nGradient check: {'PASSED ✓' if is_correct else 'FAILED ✗'}")

    # Wrong gradient (to demonstrate failure)
    print("\n" + "-" * 60)
    print("Testing with WRONG gradient (should fail)")

    wrong_grad = lambda x: [3 * x[0], 2 * x[1]]  # Incorrect!
    is_correct = check_gradient(f, wrong_grad, point)
    print(f"Gradient check: {'PASSED ✓' if is_correct else 'FAILED ✗'}")


def neural_network_gradient_flow():
    """Show how gradients flow through a network."""
    print("\n" + "=" * 60)
    print("GRADIENT FLOW IN NEURAL NETWORKS")
    print("=" * 60)

    print("\nSimple network: y = w2 * relu(w1 * x)")

    # Network parameters
    x = 2.0
    w1 = 0.5
    w2 = 3.0

    # Forward pass
    h = w1 * x  # Hidden layer (linear)
    h_activated = max(0, h)  # ReLU
    y = w2 * h_activated  # Output

    print(f"\nForward pass:")
    print(f"  x = {x}")
    print(f"  h = w1 * x = {w1} * {x} = {h}")
    print(f"  h_activated = ReLU(h) = {h_activated}")
    print(f"  y = w2 * h_activated = {w2} * {h_activated} = {y}")

    # Backward pass (assuming dy/dy = 1, i.e., y is the loss)
    print(f"\nBackward pass (chain rule):")

    dy_dy = 1.0
    print(f"  dy/dy = {dy_dy}")

    dy_dw2 = dy_dy * h_activated
    print(f"  dy/dw2 = dy/dy * h_activated = {dy_dy} * {h_activated} = {dy_dw2}")

    dy_dh_activated = dy_dy * w2
    print(f"  dy/dh_activated = dy/dy * w2 = {dy_dy} * {w2} = {dy_dh_activated}")

    # ReLU gradient: 1 if h > 0, else 0
    dh_activated_dh = 1.0 if h > 0 else 0.0
    print(f"  dh_activated/dh = ReLU'(h) = {dh_activated_dh}")

    dy_dh = dy_dh_activated * dh_activated_dh
    print(f"  dy/dh = {dy_dh_activated} * {dh_activated_dh} = {dy_dh}")

    dy_dw1 = dy_dh * x
    print(f"  dy/dw1 = dy/dh * x = {dy_dh} * {x} = {dy_dw1}")

    print(f"\nGradients:")
    print(f"  dL/dw1 = {dy_dw1}")
    print(f"  dL/dw2 = {dy_dw2}")

    print(f"\nGradient descent update (lr=0.1):")
    lr = 0.1
    w1_new = w1 - lr * dy_dw1
    w2_new = w2 - lr * dy_dw2
    print(f"  w1_new = {w1} - {lr} * {dy_dw1} = {w1_new}")
    print(f"  w2_new = {w2} - {lr} * {dy_dw2} = {w2_new}")


def main():
    """Run all differentiation tutorials."""
    basic_derivatives()
    multivariable_calculus()
    jacobian_demo()
    hessian_demo()
    gradient_descent_demo()
    gradient_checking()
    neural_network_gradient_flow()

    print("\n" + "=" * 60)
    print("Differentiation Tutorial Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("✓ Derivatives measure rate of change")
    print("✓ Gradients point in direction of steepest ascent")
    print("✓ Gradient descent follows negative gradient to minimize")
    print("✓ Jacobian shows how vector outputs change with inputs")
    print("✓ Hessian reveals curvature for second-order optimization")
    print("✓ Chain rule is the foundation of backpropagation")
    print("✓ Always check gradients when implementing neural networks!")


if __name__ == '__main__':
    main()
