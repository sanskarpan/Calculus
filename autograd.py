"""
Automatic Differentiation (Autograd) Module
============================================

Implements computational graphs and automatic differentiation for
understanding backpropagation in deep learning.

This is the CORE of how modern deep learning frameworks work!

Features:
- Forward mode automatic differentiation
- Reverse mode automatic differentiation (backpropagation)
- Computational graph construction
- Gradient computation through chain rule

Essential for understanding:
- PyTorch autograd
- TensorFlow GradientTape
- JAX grad
- Backpropagation in neural networks
"""

from typing import List, Set, Optional, Tuple
import math


class Variable:
    """
    A variable in the computational graph.

    Tracks:
    - Value (forward pass)
    - Gradient (backward pass)
    - Operation that created it
    - Parents in the graph
    """

    def __init__(
        self,
        value: float,
        name: str = "",
        _children: Tuple["Variable", ...] = (),
        _op: str = "",
    ):
        """
        Initialize a Variable.

        Args:
            value: The numerical value
            name: Optional name for debugging
            _children: Parent variables (for internal use)
            _op: Operation that created this variable
        """
        self.value = float(value)
        self.grad = 0.0  # Gradient (dL/dself)
        self.name = name

        # For building computational graph
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Variable(value={self.value:.4f}, grad={self.grad:.4f}, name='{self.name}')"

    def __str__(self) -> str:
        if self.name:
            return f"{self.name}={self.value:.4f}"
        return f"{self.value:.4f}"

    # ============================================================
    # BASIC OPERATIONS (with gradient computation)
    # ============================================================

    def __add__(self, other: "Variable") -> "Variable":
        """
        Addition: z = x + y

        Forward: z = x + y
        Backward: dL/dx = dL/dz * 1, dL/dy = dL/dz * 1
        """
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value + other.value, _children=(self, other), _op="+")

        def _backward():
            # Chain rule: dL/dself = dL/dout * dout/dself
            self.grad += out.grad  # dout/dself = 1
            other.grad += out.grad  # dout/dother = 1

        out._backward = _backward
        return out

    def __mul__(self, other: "Variable") -> "Variable":
        """
        Multiplication: z = x * y

        Forward: z = x * y
        Backward: dL/dx = dL/dz * y, dL/dy = dL/dz * x
        """
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value * other.value, _children=(self, other), _op="*")

        def _backward():
            self.grad += other.value * out.grad  # dout/dself = other.value
            other.grad += self.value * out.grad  # dout/dother = self.value

        out._backward = _backward
        return out

    def __pow__(self, power: float) -> "Variable":
        """
        Power: z = x^n

        Forward: z = x^n
        Backward: dL/dx = dL/dz * n * x^(n-1)
        """
        assert isinstance(power, (int, float)), "Power must be a number"
        out = Variable(self.value**power, _children=(self,), _op=f"**{power}")

        def _backward():
            self.grad += power * (self.value ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Variable":
        """Negation: z = -x"""
        return self * -1

    def __sub__(self, other: "Variable") -> "Variable":
        """Subtraction: z = x - y"""
        return self + (-other)

    def __truediv__(self, other: "Variable") -> "Variable":
        """Division: z = x / y = x * y^(-1)"""
        return self * (other**-1)

    def __radd__(self, other: "Variable") -> "Variable":
        """Right addition: other + self"""
        return self + other

    def __rmul__(self, other: "Variable") -> "Variable":
        """Right multiplication: other * self"""
        return self * other

    def __rsub__(self, other: "Variable") -> "Variable":
        """Right subtraction: other - self"""
        return Variable(other) - self

    def __rtruediv__(self, other: "Variable") -> "Variable":
        """Right division: other / self"""
        return Variable(other) / self

    # ============================================================
    # ACTIVATION FUNCTIONS (with gradients)
    # ============================================================

    def relu(self) -> "Variable":
        """
        ReLU activation: z = max(0, x)

        Forward: z = max(0, x)
        Backward: dL/dx = dL/dz * (1 if x > 0 else 0)
        """
        out = Variable(max(0, self.value), _children=(self,), _op="ReLU")

        def _backward():
            self.grad += (self.value > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Variable":
        """
        Sigmoid activation: z = 1 / (1 + exp(-x))

        Forward: z = σ(x)
        Backward: dL/dx = dL/dz * σ(x) * (1 - σ(x))

        Numerically stable: uses max(0, x) and min(0, x) to avoid overflow.
        """
        # Numerically stable sigmoid
        if self.value >= 0:
            sig = 1 / (1 + math.exp(-self.value))
        else:
            exp_x = math.exp(self.value)
            sig = exp_x / (1 + exp_x)
        out = Variable(sig, _children=(self,), _op="sigmoid")

        def _backward():
            self.grad += sig * (1 - sig) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Variable":
        """
        Tanh activation: z = tanh(x)

        Forward: z = tanh(x)
        Backward: dL/dx = dL/dz * (1 - tanh²(x))
        """
        t = math.tanh(self.value)
        out = Variable(t, _children=(self,), _op="tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Variable":
        """
        Exponential: z = exp(x)

        Forward: z = e^x
        Backward: dL/dx = dL/dz * e^x
        """
        out = Variable(math.exp(self.value), _children=(self,), _op="exp")

        def _backward():
            self.grad += out.value * out.grad  # d/dx(e^x) = e^x

        out._backward = _backward
        return out

    def log(self) -> "Variable":
        """
        Natural logarithm: z = log(x)

        Forward: z = ln(x)
        Backward: dL/dx = dL/dz * (1/x)

        Raises:
            ValueError: If x <= 0 (log is undefined for non-positive values)
        """
        if self.value <= 0:
            raise ValueError(f"log undefined for non-positive value: {self.value}")
        out = Variable(math.log(self.value), _children=(self,), _op="log")

        def _backward():
            self.grad += (1 / self.value) * out.grad

        out._backward = _backward
        return out

    def sin(self) -> "Variable":
        """
        Sine: z = sin(x)

        Forward: z = sin(x)
        Backward: dL/dx = dL/dz * cos(x)
        """
        out = Variable(math.sin(self.value), _children=(self,), _op="sin")

        def _backward():
            self.grad += math.cos(self.value) * out.grad

        out._backward = _backward
        return out

    def cos(self) -> "Variable":
        """
        Cosine: z = cos(x)

        Forward: z = cos(x)
        Backward: dL/dx = dL/dz * (-sin(x))
        """
        out = Variable(math.cos(self.value), _children=(self,), _op="cos")

        def _backward():
            self.grad += -math.sin(self.value) * out.grad

        out._backward = _backward
        return out

    def sqrt(self) -> "Variable":
        """
        Square root: z = sqrt(x)

        Forward: z = x^0.5
        Backward: dL/dx = dL/dz * (1 / (2 * sqrt(x)))

        Raises:
            ValueError: If x < 0 (sqrt undefined for negative values)
        """
        if self.value < 0:
            raise ValueError(f"sqrt undefined for negative value: {self.value}")
        out = Variable(math.sqrt(self.value), _children=(self,), _op="sqrt")

        def _backward():
            if self.value > 0:
                self.grad += (0.5 / math.sqrt(self.value)) * out.grad

        out._backward = _backward
        return out

    def abs(self) -> "Variable":
        """
        Absolute value: z = |x|

        Forward: z = |x|
        Backward: dL/dx = dL/dz * sign(x) where sign(0) = 1 (subgradient)

        Note: Uses subgradient at x=0.
        """
        out = Variable(abs(self.value), _children=(self,), _op="abs")

        def _backward():
            self.grad += (1 if self.value >= 0 else -1) * out.grad

        out._backward = _backward
        return out

    def log1p(self) -> "Variable":
        """
        log(1 + x): z = log(1 + x)

        Numerically stable log for x near 0.
        Forward: z = ln(1 + x)
        Backward: dL/dx = dL/dz * (1 / (1 + x))

        Raises:
            ValueError: If 1 + x <= 0
        """
        if self.value + 1 <= 0:
            raise ValueError(f"log1p undefined for value: {self.value}")
        out = Variable(math.log1p(self.value), _children=(self,), _op="log1p")

        def _backward():
            self.grad += (1 / (1 + self.value)) * out.grad

        out._backward = _backward
        return out

    def softplus(self) -> "Variable":
        """
        Softplus: z = log(1 + exp(x))

        Numerically stable activation function.
        Forward: z = softplus(x)
        Backward: dL/dx = dL/dz * sigmoid(x)
        """
        # Stable softplus:
        # softplus(x) = max(x, 0) + log1p(exp(-|x|))
        x = self.value
        out_value = max(x, 0.0) + math.log1p(math.exp(-abs(x)))
        out = Variable(out_value, _children=(self,), _op="softplus")

        def _backward():
            # d/dx softplus(x) = sigmoid(x), computed stably
            if x >= 0:
                sig = 1 / (1 + math.exp(-x))
            else:
                exp_x = math.exp(x)
                sig = exp_x / (1 + exp_x)
            self.grad += sig * out.grad

        out._backward = _backward
        return out

    # ============================================================
    # BACKPROPAGATION
    # ============================================================

    def backward(self):
        """
        Compute gradients using reverse-mode automatic differentiation.

        This is BACKPROPAGATION!

        Process:
        1. Topologically sort the computational graph
        2. Initialize gradient of output to 1 (dL/dL = 1)
        3. Traverse graph backward, applying chain rule
        """
        # Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Backward pass
        self.grad = 1.0  # dL/dL = 1
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        """Reset gradients to zero (needed between training iterations)."""
        self.grad = 0.0


# ============================================================
# COMPUTATIONAL GRAPH UTILITIES
# ============================================================


class ComputationGraph:
    """
    Manages a computational graph for automatic differentiation.

    Provides utilities for visualizing and analyzing the graph.
    """

    def __init__(self):
        self.nodes = []
        self.edges = []

    @staticmethod
    def trace(root: Variable) -> Tuple[Set[Variable], Set[Tuple[Variable, Variable]]]:
        """
        Build graph from a root variable.

        Args:
            root: Output variable

        Returns:
            Tuple of (nodes, edges)
        """
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        build(root)
        return nodes, edges

    @staticmethod
    def draw_graph(root: Variable, format: str = "text") -> str:
        """
        Create a text representation of the computational graph.

        Args:
            root: Output variable
            format: 'text' for text representation

        Returns:
            String representation of graph
        """
        nodes, edges = ComputationGraph.trace(root)

        # Build text representation
        lines = ["Computational Graph:", "=" * 50]

        # Show nodes
        lines.append("\nNodes:")
        for node in nodes:
            op_str = f" (op: {node._op})" if node._op else ""
            lines.append(f"  {node}{op_str}")

        # Show edges
        lines.append("\nEdges:")
        for src, dst in edges:
            lines.append(f"  {src} -> {dst}")

        return "\n".join(lines)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def backward(output: Variable):
    """
    Compute gradients for all variables in the graph.

    This is the function you call to do backpropagation!

    Args:
        output: The output variable (usually the loss)

    Example:
        >>> x = Variable(2.0, name='x')
        >>> y = Variable(3.0, name='y')
        >>> z = x * y + x  # z = 2*3 + 2 = 8
        >>> backward(z)
        >>> print(f"dz/dx = {x.grad}")  # Should be 3 + 1 = 4
        >>> print(f"dz/dy = {y.grad}")  # Should be 2
    """
    output.backward()


def numerical_gradient(
    f: callable, variables: List[Variable], h: float = 1e-5
) -> List[float]:
    """
    Compute numerical gradients for gradient checking.

    Args:
        f: Function that takes variables and returns scalar
        variables: List of input variables
        h: Finite difference step

    Returns:
        List of numerical gradients

    Example:
        >>> x = Variable(2.0)
        >>> y = Variable(3.0)
        >>> f = lambda: x * y
        >>> num_grads = numerical_gradient(f, [x, y])
    """
    grads = []

    for var in variables:
        original = var.value

        var.value = original + h
        f_plus = f().value

        var.value = original - h
        f_minus = f().value

        var.value = original

        grad = (f_plus - f_minus) / (2 * h)
        grads.append(grad)

    return grads


def gradient_check(
    f: callable, variables: List[Variable], tolerance: float = 1e-5
) -> bool:
    """
    Check if automatic gradients match numerical gradients.

    Critical for debugging neural networks!

    Args:
        f: Function to check
        variables: Input variables
        tolerance: Maximum allowed error

    Returns:
        True if gradients match

    Example:
        >>> x = Variable(2.0, name='x')
        >>> y = Variable(3.0, name='y')
        >>> def f():
        >>>     return x**2 + y**2
        >>> output = f()
        >>> backward(output)
        >>> gradient_check(f, [x, y])
        True
    """
    # Compute automatic gradients
    output = f()
    backward(output)

    auto_grads = [v.grad for v in variables]

    # Reset gradients
    for v in variables:
        v.grad = 0.0

    # Compute numerical gradients
    num_grads = numerical_gradient(f, variables)

    # Compare
    for i, (auto, num) in enumerate(zip(auto_grads, num_grads)):
        diff = abs(auto - num)
        if diff > tolerance:
            print(f"Gradient mismatch for variable {i}:")
            print(f"  Automatic: {auto:.6f}")
            print(f"  Numerical: {num:.6f}")
            print(f"  Difference: {diff:.6f}")
            return False

    return True


# ============================================================
# EXAMPLE: SIMPLE NEURAL NETWORK NEURON
# ============================================================


def neuron(
    inputs: List[Variable],
    weights: List[Variable],
    bias: Variable,
    activation: str = "relu",
) -> Variable:
    """
    Compute output of a single neuron.

    This demonstrates the full forward and backward pass!

    Args:
        inputs: Input variables
        weights: Weight variables
        bias: Bias variable
        activation: 'relu', 'sigmoid', or 'tanh'

    Returns:
        Output variable (with gradient tracking)

    Example:
        >>> # Create inputs and parameters
        >>> x = [Variable(1.0), Variable(2.0)]
        >>> w = [Variable(0.5), Variable(-0.3)]
        >>> b = Variable(0.1)
        >>> # Forward pass
        >>> output = neuron(x, w, b, activation='relu')
        >>> # Backward pass
        >>> backward(output)
        >>> # Now w[0].grad, w[1].grad, b.grad contain gradients!
    """
    # Linear combination: z = w·x + b
    z = bias
    for xi, wi in zip(inputs, weights):
        z = z + wi * xi

    # Activation
    if activation == "relu":
        return z.relu()
    elif activation == "sigmoid":
        return z.sigmoid()
    elif activation == "tanh":
        return z.tanh()
    else:
        return z  # Linear


def mse_loss(predictions: List[Variable], targets: List[float]) -> Variable:
    """
    Mean Squared Error loss function.

    Args:
        predictions: Predicted values (Variables)
        targets: True values (floats)

    Returns:
        MSE loss (Variable)

    Example:
        >>> pred = [Variable(2.5), Variable(3.0)]
        >>> target = [2.0, 3.5]
        >>> loss = mse_loss(pred, target)
        >>> backward(loss)
        >>> # Now predictions have gradients
    """
    n = len(predictions)
    total_loss = Variable(0.0)

    for pred, target in zip(predictions, targets):
        error = pred - Variable(target)
        total_loss = total_loss + error**2

    return total_loss / Variable(n)


def sgd_step(parameters: List[Variable], learning_rate: float = 0.01):
    """
    Perform one step of Stochastic Gradient Descent.

    Updates: θ = θ - α * ∇θ L

    Args:
        parameters: List of parameter variables
        learning_rate: Step size

    Example:
        >>> # After computing loss and backward()
        >>> sgd_step([w1, w2, b], learning_rate=0.01)
        >>> # Parameters are updated
    """
    for param in parameters:
        param.value -= learning_rate * param.grad
        param.grad = 0.0  # Reset gradient


def zero_grad(params: List[Variable]):
    """
    Reset gradients of all parameters to zero.

    Essential for training loops to avoid gradient accumulation.

    Args:
        params: List of parameter Variables

    Example:
        >>> x = Variable(2.0)
        >>> y = x ** 2
        >>> backward(y)
        >>> print(x.grad)  # 4.0
        >>> zero_grad([x])
        >>> print(x.grad)  # 0.0
    """
    for param in params:
        param.grad = 0.0
