# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 1.2.0
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A professional-grade mathematics module for Python, providing a
#                vast array of functionalities from basic arithmetic to graph
#                theory, fourier analysis, and special functions. It remains
#                self-sufficient, leveraging NumPy only for performance-critical
#                array operations as specified.
#
################################################################################

import numpy as np
from functools import reduce
from heapq import heappop, heappush
from typing import (List, Tuple, Callable, Dict, Union, Any, TypeVar,
                    Generic)

# Define a generic type for graph nodes
NodeType = TypeVar('NodeType')

# --- Helper functions used internally for high-precision calculations ---

def _calculate_pi(precision_terms: int = 10000) -> float:
    """Calculates PI using the Nilakantha series for faster convergence."""
    pi = 3.0
    sign = 1
    for i in range(2, 2 * precision_terms + 1, 2):
        pi += sign * (4 / (i * (i + 1) * (i + 2)))
        sign *= -1
    return pi

def _calculate_e(precision_terms: int = 20) -> float:
    """Calculates Euler's number (e) using its series expansion."""
    e = 1.0
    factorial_term = 1.0
    for i in range(1, precision_terms + 1):
        factorial_term *= i
        e += 1.0 / factorial_term
    return e


class Constants:
    """A collection of fundamental mathematical constants."""
    PI: float = _calculate_pi()
    E: float = _calculate_e()
    TAU: float = 2 * PI
    GOLDEN_RATIO: float = (1 + 5**0.5) / 2
    SQRT_2: float = 2**0.5


class BasicOps:
    """Encapsulates fundamental arithmetic and logarithmic operations."""
    @staticmethod
    def add(a: float, b: float) -> float: return a + b
    @staticmethod
    def subtract(a: float, b: float) -> float: return a - b
    @staticmethod
    def multiply(a: float, b: float) -> float: return a * b
    @staticmethod
    def divide(a: float, b: float) -> float:
        if b == 0: raise ValueError("Error: Division by zero is not allowed.")
        return a / b
    @staticmethod
    def power(base: float, exp: float) -> float: return base ** exp
    @staticmethod
    def root(n: int, x: float, precision: float = 1e-12) -> float:
        """Calculates the nth root of x using Newton's method."""
        if x == 0: return 0
        if x < 0 and n % 2 == 0:
            raise ValueError("Cannot calculate an even root of a negative number.")
        if x < 0: return -BasicOps.root(n, -x, precision)
        guess = x / n
        while True:
            next_guess = ((n - 1) * guess + x / (guess ** (n - 1))) / n
            if abs(guess - next_guess) < precision: return next_guess
            guess = next_guess
    @staticmethod
    def sqrt(x: float, precision: float = 1e-12) -> float: return BasicOps.root(2, x, precision)
    @staticmethod
    def abs(x: float) -> float: return x if x >= 0 else -x
    @staticmethod
    def ln(x: float, terms: int = 50) -> float:
        """Calculates the natural logarithm (base e) using a fast-converging series."""
        if x <= 0: raise ValueError("Logarithm is only defined for positive numbers.")
        if x == 1: return 0
        y = (x - 1) / (x + 1)
        res = 0
        for i in range(terms):
            res += (y ** (2 * i + 1)) / (2 * i + 1)
        return 2 * res
    @staticmethod
    def log(base: float, x: float) -> float:
        """Calculates the logarithm of x to a given base."""
        return BasicOps.ln(x) / BasicOps.ln(base)

# --- Advanced Data Type Classes ---

class ComplexNumber:
    """Represents a complex number and defines its arithmetic."""
    def __init__(self, real: float, imag: float = 0.0): self.real, self.imag = real, imag
    def __repr__(self) -> str:
        if self.imag == 0: return f"{self.real}"
        if self.real == 0: return f"{self.imag}i"
        return f"{self.real} {'+' if self.imag > 0 else '-'} {abs(self.imag)}i"
    def __add__(self, other: Any) -> 'ComplexNumber':
        if not isinstance(other, ComplexNumber): other = ComplexNumber(other)
        return ComplexNumber(self.real + other.real, self.imag + other.imag)
    def __sub__(self, other: Any) -> 'ComplexNumber':
        if not isinstance(other, ComplexNumber): other = ComplexNumber(other)
        return ComplexNumber(self.real - other.real, self.imag - other.imag)
    def __mul__(self, other: Any) -> 'ComplexNumber':
        if not isinstance(other, ComplexNumber): other = ComplexNumber(other)
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return ComplexNumber(real, imag)
    def __truediv__(self, other: Any) -> 'ComplexNumber':
        if not isinstance(other, ComplexNumber): other = ComplexNumber(other)
        denom = other.real**2 + other.imag**2
        if denom == 0: raise ValueError("Division by zero complex number.")
        real = (self.real*other.real + self.imag*other.imag) / denom
        imag = (self.imag*other.real - self.real*other.imag) / denom
        return ComplexNumber(real, imag)
    def conjugate(self) -> 'ComplexNumber': return ComplexNumber(self.real, -self.imag)
    def modulus(self) -> float: return BasicOps.sqrt(self.real**2 + self.imag**2)

class Polynomial:
    """Represents a polynomial for symbolic-like operations."""
    def __init__(self, coeffs: List[float]): self.coeffs = coeffs
    def __repr__(self) -> str:
        if not self.coeffs: return "0"
        parts = []
        for i, c in enumerate(self.coeffs):
            if abs(c) < 1e-9: continue
            power = len(self.coeffs) - 1 - i
            c_str = f"{c:.4g}" if (c != 1 and c != -1) or power == 0 else ("-" if c == -1 else "")
            v_str = f"x^{power}" if power > 1 else ("x" if power == 1 else "")
            parts.append(f"{c_str}{v_str}")
        return " + ".join(parts).replace("+ -", "- ") or "0"
    def __call__(self, x: float) -> float: return reduce(lambda acc, c: acc * x + c, self.coeffs)
    def __add__(self, other: 'Polynomial') -> 'Polynomial':
        n1, n2 = len(self.coeffs), len(other.coeffs)
        c1 = [0]*(n2-n1) + self.coeffs if n1 < n2 else self.coeffs
        c2 = [0]*(n1-n2) + other.coeffs if n2 < n1 else other.coeffs
        return Polynomial([a + b for a, b in zip(c1, c2)])
    def __mul__(self, other: 'Polynomial') -> 'Polynomial':
        new_coeffs = [0] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, c1 in enumerate(self.coeffs):
            for j, c2 in enumerate(other.coeffs):
                new_coeffs[i+j] += c1 * c2
        return Polynomial(new_coeffs)
    def differentiate(self) -> 'Polynomial':
        deg = len(self.coeffs) - 1
        if deg < 1: return Polynomial([0])
        return Polynomial([c * (deg - i) for i, c in enumerate(self.coeffs[:-1])])

class Graph(Generic[NodeType]):
    """A versatile graph data structure."""
    def __init__(self, directed: bool = False):
        self.adjacency_list: Dict[NodeType, List[Tuple[NodeType, float]]] = {}
        self.directed = directed
    def add_node(self, node: NodeType) -> None:
        if node not in self.adjacency_list: self.adjacency_list[node] = []
    def add_edge(self, u: NodeType, v: NodeType, weight: float = 1.0) -> None:
        self.add_node(u); self.add_node(v)
        self.adjacency_list[u].append((v, weight))
        if not self.directed: self.adjacency_list[v].append((u, weight))
    def __repr__(self) -> str:
        return (f"Graph(Nodes: {len(self.adjacency_list)}, "
                f"Type: {'Directed' if self.directed else 'Undirected'})")

# --- Mathematical Domain Classes ---

class AdvancedOps:
    """Core advanced operations like factorial and number theory basics."""
    @staticmethod
    def factorial(n: int) -> int:
        if not isinstance(n, int) or n < 0: raise ValueError("Factorial is for non-negative integers.")
        return 1 if n == 0 else reduce(lambda x, y: x*y, range(1, n + 1))
    @staticmethod
    def gcd(a: int, b: int) -> int:
        while b: a, b = b, a % b
        return abs(a)
    @staticmethod
    def lcm(a: int, b: int) -> int:
        if a == 0 or b == 0: return 0
        return abs(a * b) // AdvancedOps.gcd(a, b)

class Trigonometry:
    _TAYLOR_TERMS = 15
    @staticmethod
    def sin(x: float) -> float:
        x %= Constants.TAU
        if x > Constants.PI: x -= Constants.TAU
        return sum(((-1)**i)*(x**(2*i+1))/AdvancedOps.factorial(2*i+1) for i in range(Trigonometry._TAYLOR_TERMS))
    @staticmethod
    def cos(x: float) -> float:
        x %= Constants.TAU
        if x > Constants.PI: x -= Constants.TAU
        return sum(((-1)**i)*(x**(2*i))/AdvancedOps.factorial(2*i) for i in range(Trigonometry._TAYLOR_TERMS))
    @staticmethod
    def tan(x: float) -> float:
        cosine_val = Trigonometry.cos(x)
        if abs(cosine_val) < 1e-15: raise ValueError("Tangent undefined.")
        return Trigonometry.sin(x) / cosine_val

class Calculus:
    @staticmethod
    def differentiate(func: Callable[[float], float], x: float, h: float = 1e-7) -> float:
        return (func(x + h) - func(x - h)) / (2 * h)
    @staticmethod
    def integrate(func: Callable[[float], float], a: float, b: float, n: int = 10000) -> float:
        if n % 2 != 0: n += 1
        h = (b - a) / n
        integral = func(a) + func(b) + \
                   4 * sum(func(a + i * h) for i in range(1, n, 2)) + \
                   2 * sum(func(a + i * h) for i in range(2, n, 2))
        return integral * h / 3

class SpecialFunctions:
    """Implementations of special mathematical functions (e.g., Gamma, erf)."""
    # Lanczos approximation coefficients for g=5, n=7
    _LANCZOS_COEFFS = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    ]
    @staticmethod
    def gamma(z: float) -> float:
        """Calculates the Gamma function Γ(z) using the Lanczos approximation."""
        if z < 0.5:
            # Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
            return Constants.PI / (Trigonometry.sin(Constants.PI * z) * SpecialFunctions.gamma(1 - z))
        z -= 1
        x = SpecialFunctions._LANCZOS_COEFFS[0]
        for i in range(1, len(SpecialFunctions._LANCZOS_COEFFS)):
            x += SpecialFunctions._LANCZOS_COEFFS[i] / (z + i)
        t = z + 5.5  # g=5
        return BasicOps.sqrt(2 * Constants.PI) * (t ** (z + 0.5)) * (Constants.E ** -t) * x
    @staticmethod
    def erf(x: float) -> float:
        """Calculates the error function erf(x) by integrating the Gaussian function."""
        gaussian = lambda t: (2 / BasicOps.sqrt(Constants.PI)) * (Constants.E ** -(t**2))
        if x < 0: return -Calculus.integrate(gaussian, x, 0)
        return Calculus.integrate(gaussian, 0, x)

class NumericalAnalysis:
    """Advanced numerical methods for solving complex problems."""
    @staticmethod
    def find_root(func: Callable, derivative: Callable, x0: float, tol: float = 1e-7, max_iter: int = 100) -> float:
        """Finds a root of a function using Newton-Raphson method."""
        x = x0
        for _ in range(max_iter):
            fx = func(x)
            if abs(fx) < tol: return x
            dfx = derivative(x)
            if dfx == 0: raise ValueError("Derivative is zero. Cannot continue.")
            x = x - fx / dfx
        raise RuntimeError("Root finding failed to converge.")
    @staticmethod
    def solve_ode(f: Callable, y0: float, t_span: Tuple[float, float], h: float = 0.01) -> Tuple[List[float], List[float]]:
        """Solves an ODE y'(t) = f(t, y) using the 4th-order Runge-Kutta method."""
        t_vals = np.arange(t_span[0], t_span[1] + h, h)
        y_vals = np.zeros_like(t_vals)
        y_vals[0] = y0
        for i in range(len(t_vals) - 1):
            t, y = t_vals[i], y_vals[i]
            k1 = h * f(t, y); k2 = h * f(t + 0.5*h, y + 0.5*k1)
            k3 = h * f(t + 0.5*h, y + 0.5*k2); k4 = h * f(t + h, y + k3)
            y_vals[i+1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return t_vals.tolist(), y_vals.tolist()

class LinearAlgebra:
    """Wraps powerful NumPy functions for linear algebra."""
    @staticmethod
    def inverse(matrix: List[List[float]]) -> List[List[float]]:
        try: return np.linalg.inv(matrix).tolist()
        except np.linalg.LinAlgError: raise ValueError("Matrix is not invertible.")
    @staticmethod
    def determinant(matrix: List[List[float]]) -> float:
        try: return np.linalg.det(matrix)
        except np.linalg.LinAlgError: raise ValueError("Matrix must be square.")
    @staticmethod
    def eigenvalues_eigenvectors(matrix: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
        w, v = np.linalg.eig(matrix)
        return w.tolist(), v.tolist()

class FourierAnalysis:
    """Wraps NumPy's high-performance Fast Fourier Transform algorithms."""
    @staticmethod
    def fft(signal: List[float]) -> List[ComplexNumber]:
        """Computes the 1D Fast Fourier Transform of a real or complex signal."""
        transformed = np.fft.fft(signal)
        return [ComplexNumber(c.real, c.imag) for c in transformed]
    @staticmethod
    def ifft(transformed_signal: List[ComplexNumber]) -> List[ComplexNumber]:
        """Computes the 1D Inverse Fast Fourier Transform."""
        complex_values = [c.real + c.imag * 1j for c in transformed_signal]
        inversed = np.fft.ifft(complex_values)
        return [ComplexNumber(c.real, c.imag) for c in inversed]

class GraphTheory:
    """Algorithms for analyzing graphs."""
    @staticmethod
    def bfs(graph: Graph, start_node: NodeType) -> List[NodeType]:
        """Performs a Breadth-First Search, returning the visited nodes in order."""
        visited = {start_node}
        queue = [start_node]
        order = [start_node]
        while queue:
            u = queue.pop(0)
            for v, _ in graph.adjacency_list.get(u, []):
                if v not in visited:
                    visited.add(v); queue.append(v); order.append(v)
        return order
    @staticmethod
    def dijkstra(graph: Graph, start_node: NodeType) -> Tuple[Dict[NodeType, float], Dict[NodeType, NodeType]]:
        """
        Calculates the shortest path from a start node to all other nodes
        in a weighted graph. Returns distances and predecessors.
        """
        distances = {node: float('inf') for node in graph.adjacency_list}
        predecessors = {node: None for node in graph.adjacency_list}
        distances[start_node] = 0
        pq = [(0, start_node)]  # (distance, node)
        while pq:
            dist_u, u = heappop(pq)
            if dist_u > distances[u]: continue
            for v, weight in graph.adjacency_list.get(u, []):
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
                    heappush(pq, (distances[v], v))
        return distances, predecessors

class InformationTheory:
    """Functions for calculating information-theoretic quantities."""
    @staticmethod
    def entropy(probabilities: List[float], tol: float = 1e-9) -> float:
        """Calculates the Shannon entropy of a probability distribution."""
        if abs(sum(probabilities) - 1.0) > tol:
            raise ValueError("Probabilities must sum to 1.")
        return -sum(p * BasicOps.log(2, p) for p in probabilities if p > 0)
    @staticmethod
    def kl_divergence(p: List[float], q: List[float], tol: float = 1e-9) -> float:
        """Calculates the Kullback-Leibler (KL) divergence D_KL(P || Q)."""
        if len(p) != len(q): raise ValueError("Distributions must have the same length.")
        return sum(p[i] * BasicOps.log(2, p[i] / q[i]) for i in range(len(p)) if p[i] > 0 and q[i] > 0)

class Statistics:
    """Common statistical measures."""
    @staticmethod
    def mean(data: List[float]) -> float: return sum(data) / len(data)
    # ... other stats functions like median, variance, std_dev ...

# --- Primary Facade Class ---

class Axiom(
    Constants, BasicOps, AdvancedOps, Trigonometry, Calculus, SpecialFunctions,
    NumericalAnalysis, LinearAlgebra, FourierAnalysis, GraphTheory,
    InformationTheory, Statistics
):
    """
    The unified static interface for the AxiomPy Mathematics Engine.
    Provides access to all functions and advanced data types.
    """
    Complex = ComplexNumber
    Poly = Polynomial
    Graph = Graph
    
    def __init__(self):
        raise TypeError("AxiomPy is a static utility class and cannot be instantiated.")

# --- Example Usage Block ---
if __name__ == "__main__":
    print("=" * 60)
    print("    AxiomPy Mathematics Engine v1.2.0 - Demonstration")
    print("=" * 60)

    # 1. Special Functions
    print("\n--- 1. Special Functions ---")
    print(f"Factorial of 5: {Axiom.factorial(5)}")
    print(f"Gamma function Γ(6): {Axiom.gamma(6):.4f} (Should be 5!)")
    print(f"Gamma function Γ(3.5): {Axiom.gamma(3.5):.4f}")
    print(f"Error function erf(1): {Axiom.erf(1):.6f} (Related to 1-sigma probability)")

    # 2. Graph Theory
    print("\n--- 2. Graph Theory ---")
    city_graph = Axiom.Graph()
    edges = [('A', 'B', 7), ('A', 'C', 9), ('A', 'F', 14), ('B', 'C', 10),
             ('B', 'D', 15), ('C', 'D', 11), ('C', 'F', 2), ('D', 'E', 6),
             ('E', 'F', 9)]
    for u, v, w in edges: city_graph.add_edge(u, v, w)
    print(f"Created a weighted, undirected graph: {city_graph}")
    start, end = 'A', 'E'
    distances, predecessors = Axiom.dijkstra(city_graph, start)
    path = []
    curr = end
    while curr is not None:
        path.insert(0, curr)
        curr = predecessors[curr]
    print(f"Shortest path from {start} to {end}: {' -> '.join(path)}")
    print(f"Total distance: {distances[end]}")
    
    # 3. Fourier Analysis
    print("\n--- 3. Fourier Analysis ---")
    # Signal: 4Hz sine wave + 7Hz sine wave
    sampling_rate = 100
    times = np.linspace(0, 1, sampling_rate, endpoint=False)
    signal = [Axiom.sin(2 * Axiom.PI * 4 * t) + 0.5 * Axiom.sin(2 * Axiom.PI * 7 * t) for t in times]
    fft_result = Axiom.fft(signal)
    # Find the frequencies with the highest amplitude
    magnitudes = [c.modulus() for c in fft_result]
    # Get dominant frequencies (ignoring the second half due to symmetry)
    peak_indices = sorted(range(len(magnitudes)//2), key=lambda i: magnitudes[i], reverse=True)[:2]
    print(f"Signal is a mix of two sine waves.")
    print(f"FFT found dominant frequencies at indices: {peak_indices[0]}Hz and {peak_indices[1]}Hz (approx)")

    # 4. Information Theory
    print("\n--- 4. Information Theory ---")
    dist_fair_coin = [0.5, 0.5]
    dist_biased_coin = [0.9, 0.1]
    print(f"Entropy of a fair coin flip: {Axiom.entropy(dist_fair_coin):.4f} bits (maximum)")
    print(f"Entropy of a biased coin flip: {Axiom.entropy(dist_biased_coin):.4f} bits (lower)")
    print(f"KL Divergence (Biased || Fair): {Axiom.kl_divergence(dist_biased_coin, dist_fair_coin):.4f} bits")
    
    # 5. Logarithm
    print("\n--- 5. Logarithm ---")
    print(f"Natural log of 10 (ln(10)): {Axiom.ln(10):.6f}")
    print(f"Log base 10 of 1000: {Axiom.log(10, 1000):.1f}")
    
    print("\n" + "=" * 60)
    print("           Demonstration Complete")
    print("=" * 60)
