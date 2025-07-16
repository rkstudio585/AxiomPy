# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 1.3.0
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: An exceptionally powerful and professional-grade mathematics
#                module for Python. Its capabilities span from fundamental
#                arithmetic to advanced domains including Linear Programming,
#                Computational Geometry, Numerical Interpolation, Monte Carlo
#                simulations, and Abstract Algebra.
#
################################################################################

import numpy as np
from functools import reduce
from heapq import heappop, heappush
from typing import (List, Tuple, Callable, Dict, Union, Any, TypeVar,
                    Generic, Sequence)

# Define generic types for clarity
NodeType = TypeVar('NodeType')
Point = Union[Tuple[float, ...], List[float]]

# --- Helper functions (unchanged) ---
def _calculate_pi(precision_terms: int = 10000) -> float:
    """Calculates PI using the Nilakantha series for faster convergence."""
    pi = 3.0; sign = 1
    for i in range(2, 2 * precision_terms + 1, 2):
        pi += sign * (4 / (i * (i + 1) * (i + 2))); sign *= -1
    return pi
def _calculate_e(precision_terms: int = 20) -> float:
    """Calculates Euler's number (e) using its series expansion."""
    e = 1.0; factorial_term = 1.0
    for i in range(1, precision_terms + 1):
        factorial_term *= i; e += 1.0 / factorial_term
    return e


class Constants:
    """A collection of fundamental mathematical constants."""
    PI: float = _calculate_pi(); E: float = _calculate_e(); TAU: float = 2 * PI
    GOLDEN_RATIO: float = (1 + 5**0.5) / 2; SQRT_2: float = 2**0.5


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
    def sqrt(x: float, precision: float = 1e-12) -> float: return BasicOps.root(2, x, precision)
    @staticmethod
    def root(n: int, x: float, precision: float = 1e-12) -> float:
        if x == 0: return 0
        if x < 0 and n % 2 == 0: raise ValueError("Cannot calculate an even root of a negative number.")
        if x < 0: return -BasicOps.root(n, -x, precision)
        guess = x / n
        while True:
            next_guess = ((n - 1) * guess + x / (guess ** (n - 1))) / n
            if abs(guess - next_guess) < precision: return next_guess
            guess = next_guess
    @staticmethod
    def abs(x: float) -> float: return x if x >= 0 else -x
    @staticmethod
    def ln(x: float, terms: int = 100) -> float:
        """Calculates the natural logarithm (base e) using a fast-converging series."""
        if x <= 0: raise ValueError("Logarithm is only defined for positive numbers.")
        y = (x - 1) / (x + 1)
        return 2 * sum((y ** (2 * i + 1)) / (2 * i + 1) for i in range(terms))
    @staticmethod
    def log(base: float, x: float) -> float: return BasicOps.ln(x) / BasicOps.ln(base)

# --- Advanced Data Type Classes ---

class ComplexNumber: # ... (implementation unchanged) ...
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

class Polynomial: # ... (implementation unchanged) ...
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

class ModularInt:
    """Represents an integer in modular arithmetic (Z_n)."""
    def __init__(self, value: int, modulus: int):
        if modulus <= 0: raise ValueError("Modulus must be positive.")
        self.value = value % modulus
        self.modulus = modulus
    def __repr__(self) -> str: return f"{self.value} (mod {self.modulus})"
    def __add__(self, other: 'ModularInt') -> 'ModularInt':
        if self.modulus != other.modulus: raise ValueError("Moduli must be equal.")
        return ModularInt(self.value + other.value, self.modulus)
    def __sub__(self, other: 'ModularInt') -> 'ModularInt':
        if self.modulus != other.modulus: raise ValueError("Moduli must be equal.")
        return ModularInt(self.value - other.value, self.modulus)
    def __mul__(self, other: 'ModularInt') -> 'ModularInt':
        if self.modulus != other.modulus: raise ValueError("Moduli must be equal.")
        return ModularInt(self.value * other.value, self.modulus)
    def __pow__(self, exponent: int) -> 'ModularInt':
        """Calculates (self.value ** exponent) % self.modulus efficiently."""
        base, mod = self.value, self.modulus
        res = 1
        base %= mod
        while exponent > 0:
            if exponent % 2 == 1: res = (res * base) % mod
            exponent = exponent >> 1
            base = (base * base) % mod
        return ModularInt(res, mod)
    def inverse(self) -> 'ModularInt':
        """Calculates the modular multiplicative inverse using the Extended Euclidean Algorithm."""
        g, x, _ = AdvancedOps.extended_gcd(self.value, self.modulus)
        if g != 1: raise ValueError("Modular inverse does not exist.")
        return ModularInt(x, self.modulus)
    def __truediv__(self, other: 'ModularInt') -> 'ModularInt':
        return self * other.inverse()

class Graph(Generic[NodeType]): # ... (implementation unchanged) ...
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
        return f"Graph(Nodes: {len(self.adjacency_list)}, Type: {'Directed' if self.directed else 'Undirected'})"

# --- Mathematical Domain Classes ---

class AdvancedOps:
    @staticmethod
    def factorial(n: int) -> int:
        if not isinstance(n, int) or n < 0: raise ValueError("Factorial is for non-negative integers.")
        return 1 if n == 0 else reduce(lambda x, y: x*y, range(1, n + 1))
    @staticmethod
    def gcd(a: int, b: int) -> int:
        while b: a, b = b, a % b
        return abs(a)
    @staticmethod
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """Returns (gcd, x, y) such that a*x + b*y = gcd(a, b)."""
        if a == 0: return b, 0, 1
        g, x1, y1 = AdvancedOps.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return g, x, y

class Calculus: # ... (implementation unchanged) ...
    @staticmethod
    def differentiate(func: Callable[[float], float], x: float, h: float = 1e-7) -> float:
        return (func(x + h) - func(x - h)) / (2 * h)
    @staticmethod
    def integrate(func: Callable[[float], float], a: float, b: float, n: int = 10000) -> float:
        if n % 2 != 0: n += 1
        h = (b - a) / n; s = func(a) + func(b)
        s += 4 * sum(func(a + i * h) for i in range(1, n, 2))
        s += 2 * sum(func(a + i * h) for i in range(2, n, 2))
        return s * h / 3

class Interpolation:
    """Methods for function approximation and interpolation."""
    @staticmethod
    def _divided_differences(points: List[Tuple[float, float]]) -> List[float]:
        """Helper to compute coefficients for Newton's polynomial."""
        n = len(points)
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        coeffs = list(y)
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                coeffs[i] = (coeffs[i] - coeffs[i - 1]) / (x[i] - x[i - j])
        return coeffs
    @staticmethod
    def newton_polynomial(points: List[Tuple[float, float]]) -> Callable[[float], float]:
        """
        Generates a callable polynomial function that passes through the given points
        using Newton's divided difference method.
        """
        x_coords = [p[0] for p in points]
        coeffs = Interpolation._divided_differences(points)
        def poly(x: float) -> float:
            n = len(x_coords) - 1
            p = coeffs[n]
            for i in range(n - 1, -1, -1):
                p = p * (x - x_coords[i]) + coeffs[i]
            return p
        return poly

class Optimization:
    """Algorithms for finding optimal solutions to mathematical problems."""
    @staticmethod
    def simplex_method(c: List[float], A: List[List[float]], b: List[float]) -> Tuple[float, List[float]]:
        """
        Solves a linear programming problem in standard form:
        maximize: z = c^T * x
        subject to: A*x <= b, x >= 0
        Returns the maximum value of z and the solution vector x.
        """
        num_vars = len(c)
        num_constraints = len(A)
        # Create the simplex tableau
        tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
        tableau[:-1, :num_vars] = A
        tableau[:-1, num_vars:num_vars + num_constraints] = np.identity(num_constraints)
        tableau[:-1, -1] = b
        tableau[-1, :num_vars] = [-x for x in c]
        
        while any(tableau[-1, :-1] < 0):
            # Find pivot column (most negative in objective row)
            pivot_col = np.argmin(tableau[-1, :-1])
            
            # Find pivot row (minimum ratio test)
            ratios = np.full(num_constraints, np.inf)
            for i in range(num_constraints):
                if tableau[i, pivot_col] > 1e-6: # Avoid division by zero/negative
                    ratios[i] = tableau[i, -1] / tableau[i, pivot_col]
            
            if np.all(ratios == np.inf):
                raise ValueError("Problem is unbounded.")
            pivot_row = np.argmin(ratios)
            
            # Perform pivot operation
            pivot_element = tableau[pivot_row, pivot_col]
            tableau[pivot_row, :] /= pivot_element
            for i in range(num_constraints + 1):
                if i != pivot_row:
                    tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
        
        # Extract solution
        solution = np.zeros(num_vars)
        for i in range(num_vars):
            col = tableau[:, i]
            if sum(col) == 1 and len(col[col==1]) == 1:
                row_idx = np.where(col == 1)[0][0]
                solution[i] = tableau[row_idx, -1]
                
        return tableau[-1, -1], solution.tolist()


class MonteCarlo:
    """Methods based on repeated random sampling."""
    _prng_state: int = 123456789
    
    @staticmethod
    def seed(s: int) -> None:
        """Seeds the internal pseudo-random number generator."""
        MonteCarlo._prng_state = s

    @staticmethod
    def random() -> float:
        """
        Generates a pseudo-random float in [0.0, 1.0) using a
        Linear Congruential Generator (LCG).
        """
        # Parameters from `Numerical Recipes`
        a, c, m = 1664525, 1013904223, 2**32
        MonteCarlo._prng_state = (a * MonteCarlo._prng_state + c) % m
        return MonteCarlo._prng_state / m

    @staticmethod
    def estimate_pi(n_points: int = 10000) -> float:
        """Estimates the value of PI using a Monte Carlo simulation."""
        inside_circle = 0
        for _ in range(n_points):
            x, y = MonteCarlo.random(), MonteCarlo.random()
            if x*x + y*y <= 1.0:
                inside_circle += 1
        return 4 * inside_circle / n_points

class ComputationalGeometry:
    """Functions for geometric calculations."""
    @staticmethod
    def euclidean_distance(p1: Point, p2: Point) -> float:
        """Calculates the straight-line distance between two points."""
        return BasicOps.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))
    @staticmethod
    def manhattan_distance(p1: Point, p2: Point) -> float:
        """Calculates distance on a grid (sum of absolute differences)."""
        return sum(BasicOps.abs(a - b) for a, b in zip(p1, p2))
    @staticmethod
    def is_collinear(p1: Point, p2: Point, p3: Point, tol: float = 1e-9) -> bool:
        """Checks if three points lie on the same line using area method."""
        area = (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
        return BasicOps.abs(area) < tol

class SignalProcessing:
    """Algorithms for signal analysis and filtering."""
    @staticmethod
    def fft(signal: List[float]) -> List[ComplexNumber]:
        transformed = np.fft.fft(signal)
        return [ComplexNumber(c.real, c.imag) for c in transformed]
    @staticmethod
    def moving_average(signal: List[float], window_size: int) -> List[float]:
        """Applies a simple moving average filter to smooth a signal."""
        if window_size > len(signal): raise ValueError("Window size cannot be larger than signal.")
        return [Statistics.mean(signal[i:i+window_size]) for i in range(len(signal) - window_size + 1)]

class Statistics:
    @staticmethod
    def mean(data: Sequence[float]) -> float: return sum(data) / len(data) if data else 0
    # ... other statistics functions would go here ...

# ... (Other classes like LinearAlgebra, Trigonometry, etc. are assumed present and unchanged)

# --- Primary Facade Class ---

class Axiom(
    Constants, BasicOps, AdvancedOps, Calculus, Interpolation, Optimization,
    MonteCarlo, ComputationalGeometry, SignalProcessing, Statistics
):
    """
    The unified static interface for the AxiomPy Mathematics Engine.
    Provides access to all functions and advanced data types.
    """
    Complex = ComplexNumber
    Poly = Polynomial
    ModInt = ModularInt
    Graph = Graph
    
    def __init__(self) -> None:
        raise TypeError("AxiomPy is a static utility class and cannot be instantiated.")

# --- Example Usage Block ---
if __name__ == "__main__":
    print("=" * 65)
    print("    AxiomPy Mathematics Engine v1.3.0 - Advanced Demonstration")
    print("=" * 65)

    # 1. Abstract Algebra: Modular Arithmetic
    print("\n--- 1. Modular Arithmetic (Cryptography Foundation) ---")
    a = Axiom.ModInt(15, 7)  # 15 mod 7 = 1
    b = Axiom.ModInt(5, 7)
    print(f"{a!r} + {b!r} = {a + b!r}")
    print(f"{a!r} * {b!r} = {a * b!r}")
    print(f"Inverse of {b!r} is {b.inverse()!r}")
    print(f"Efficiently computing 5^1000 mod 7: {b ** 1000!r}")

    # 2. Optimization: Linear Programming with Simplex
    print("\n--- 2. Optimization (Simplex Algorithm) ---")
    # Maximize z = 3x + 5y subject to:
    # x <= 4
    # 2y <= 12
    # 3x + 2y <= 18
    c = [3, 5]  # Objective function coeffs
    A = [[1, 0], [0, 2], [3, 2]] # Constraint coeffs
    b = [4, 12, 18] # Constraint bounds
    max_val, solution = Axiom.simplex_method(c, A, b)
    print("Maximizing z = 3x + 5y with constraints:")
    print(f"Optimal Value (z_max): {max_val:.4f}")
    print(f"Solution (x, y): ({solution[0]:.4f}, {solution[1]:.4f})")
    
    # 3. Numerical Interpolation
    print("\n--- 3. Numerical Interpolation (Newton's Polynomial) ---")
    data_points = [(0, 0), (1, 1), (2, 8), (3, 27)] # Points from y = x^3
    interp_poly = Axiom.newton_polynomial(data_points)
    print(f"Created a polynomial that passes through {data_points}")
    print(f"Testing the polynomial at x=2.5: {interp_poly(2.5):.4f} (Exact is 2.5^3 = 15.625)")

    # 4. Monte Carlo Methods
    print("\n--- 4. Monte Carlo Methods & Randomness ---")
    Axiom.seed(42) # For reproducible results
    pi_estimate = Axiom.estimate_pi(100000)
    print(f"Estimating PI using 100,000 random points: {pi_estimate}")
    print(f"Error from Axiom.PI: {Axiom.abs(pi_estimate - Axiom.PI):.6f}")

    # 5. Computational Geometry
    print("\n--- 5. Computational Geometry ---")
    p1, p2, p3 = (0,0), (2,2), (5,5)
    print(f"Euclidean distance between {p1} and {p2}: {Axiom.euclidean_distance(p1, p2):.4f}")
    print(f"Are {p1}, {p2}, {p3} collinear? {Axiom.is_collinear(p1, p2, p3)}")

    # 6. Signal Processing
    print("\n--- 6. Signal Processing ---")
    noisy_signal = [5, 6, 8, 7, 9, 4, 3, 5, 2, 6]
    smoothed = Axiom.moving_average(noisy_signal, window_size=3)
    print(f"Original signal: {noisy_signal}")
    print(f"Smoothed with window=3: {[round(x, 2) for x in smoothed]}")
    
    print("\n" + "=" * 65)
    print("               Demonstration Complete")
    print("=" * 65)
