# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 1.5.0
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A professional-grade mathematics module for Python. This version
#                introduces advanced capabilities in Machine Learning (K-Means),
#                Non-Linear Optimization (Gradient Descent), PDE Solvers (Heat
#                Equation), and Exact Arithmetic (Rational Numbers).
#
################################################################################

import numpy as np
from functools import reduce
from heapq import heapify, heappop, heappush
from collections import deque
from typing import (List, Tuple, Callable, Dict, Union, Any, TypeVar,
                    Generic, Sequence)
import random

# --- Type Aliases for Enhanced Clarity ---
NodeType = TypeVar('NodeType')
Vector = List[float]
Matrix = List[List[float]]
Point = Union[Tuple[float, ...], Vector]

# --- Core Data Type and Utility Classes ---

class Constants:
    PI: float = 3.141592653589793; E: float = 2.718281828459045
    TAU: float = 2 * PI; GOLDEN_RATIO: float = (1 + 5**0.5)/2

class AdvancedOps:
    @staticmethod
    def gcd(a: int, b: int) -> int:
        while b: a, b = b, a % b
        return abs(a)

class RationalNumber:
    """Represents a rational number with perfect precision."""
    def __init__(self, numerator: int, denominator: int = 1):
        if denominator == 0: raise ValueError("Denominator cannot be zero.")
        common = AdvancedOps.gcd(numerator, denominator)
        self.num = numerator // common
        self.den = denominator // common
        if self.den < 0: self.num, self.den = -self.num, -self.den
    def __repr__(self) -> str:
        return f"{self.num}/{self.den}" if self.den != 1 else f"{self.num}"
    def __add__(self, other: 'RationalNumber') -> 'RationalNumber':
        new_num = self.num * other.den + other.num * self.den
        new_den = self.den * other.den
        return RationalNumber(new_num, new_den)
    def __sub__(self, other: 'RationalNumber') -> 'RationalNumber':
        new_num = self.num * other.den - other.num * self.den
        new_den = self.den * other.den
        return RationalNumber(new_num, new_den)
    def __mul__(self, other: 'RationalNumber') -> 'RationalNumber':
        return RationalNumber(self.num * other.num, self.den * other.den)
    def __truediv__(self, other: 'RationalNumber') -> 'RationalNumber':
        return RationalNumber(self.num * other.den, self.den * other.num)
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RationalNumber): return NotImplemented
        return self.num == other.num and self.den == other.den
    def to_float(self) -> float: return self.num / self.den

# (Other data types like ComplexNumber, Polynomial, Graph assumed present)
class Graph(Generic[NodeType]): #... (full implementation)
    def __init__(self, directed: bool = False):
        self.adj: Dict[NodeType, List[Tuple[NodeType, float]]] = {}
        self.directed=directed
    def add_edge(self,u,v,w=1.0):
        if u not in self.adj: self.adj[u] = []
        if v not in self.adj: self.adj[v] = []
        self.adj[u].append((v,w))
        if not self.directed: self.adj[v].append((u,w))

# --- Mathematical Domain Classes ---

class LinearAlgebra:
    """Core and advanced linear algebra operations."""
    @staticmethod
    def transpose(matrix: Matrix) -> Matrix: return np.transpose(matrix).tolist()
    @staticmethod
    def inverse(matrix: Matrix) -> Matrix: return np.linalg.inv(matrix).tolist()
    @staticmethod
    def eigenvalues_eigenvectors(matrix: Matrix) -> Tuple[Vector, Matrix]:
        w, v = np.linalg.eig(matrix); return w.tolist(), v.tolist()
    @staticmethod
    def dot(v1: Vector, v2: Vector) -> float: return np.dot(v1, v2)
    @staticmethod
    def norm(v: Vector, ord: int = 2) -> float: return np.linalg.norm(v, ord=ord)
    @staticmethod
    def project_vector(v: Vector, u: Vector) -> Vector:
        """Projects vector v onto vector u."""
        u_np = np.array(u)
        v_np = np.array(v)
        scalar_projection = np.dot(v_np, u_np) / np.dot(u_np, u_np)
        return (scalar_projection * u_np).tolist()

class Statistics: # ... (full implementation assumed present)
    @staticmethod
    def mean(data: Sequence[float]) -> float: return sum(data) / len(data) if data else 0
    @staticmethod
    def chi_squared_test(observed, expected): return {} # Assumed present

class Optimization:
    """Algorithms for finding optimal solutions to mathematical problems."""
    @staticmethod
    def gradient_descent(
        grad: Callable[[Vector], Vector],
        start: Vector,
        learn_rate: float = 0.01,
        n_iter: int = 1000,
        tolerance: float = 1e-6
    ) -> Vector:
        """
        Finds a local minimum of a multivariate function using gradient descent.
        :param grad: A function that computes the gradient of the function to minimize.
        :param start: The initial starting point (vector).
        :return: The point (vector) at which a local minimum is found.
        """
        vec = np.array(start, dtype=float)
        for i in range(n_iter):
            gradient = np.array(grad(vec.tolist()))
            if LinearAlgebra.norm(gradient) < tolerance:
                print(f"Converged at iteration {i}.")
                return vec.tolist()
            vec -= learn_rate * gradient
        print("Warning: Did not converge within the specified number of iterations.")
        return vec.tolist()

class MachineLearning:
    """A collection of fundamental machine learning algorithms."""
    @staticmethod
    def _euclidean_dist_sq(p1: Vector, p2: Vector) -> float:
        return sum([(a - b)**2 for a, b in zip(p1, p2)])

    @staticmethod
    def kmeans(data: Matrix, k: int, max_iter: int = 100) -> Tuple[Vector, List[int]]:
        """
        Partitions a dataset into k clusters using the K-Means algorithm.
        :return: A tuple of (centroids, labels) where labels[i] is the
                 cluster index for data[i].
        """
        data_np = np.array(data)
        # 1. Initialize centroids randomly from data points
        centroids = data_np[random.sample(range(len(data_np)), k)]
        
        for _ in range(max_iter):
            # 2. Assignment step
            labels = [np.argmin([MachineLearning._euclidean_dist_sq(p, c) for c in centroids])
                      for p in data_np]
            
            # 3. Update step
            new_centroids = np.array([data_np[np.array(labels) == i].mean(axis=0)
                                     for i in range(k)])
            
            # 4. Check for convergence
            if np.allclose(centroids, new_centroids): break
            centroids = new_centroids
            
        return centroids.tolist(), labels

class NumericalAnalysis:
    """Advanced numerical methods for solvers and simulations."""
    @staticmethod
    def pde_solve_heat_1d(
        L: float, T: float, alpha: float,
        nx: int, nt: int,
        initial_condition: Callable[[float], float],
        boundary_conditions: Tuple[float, float]
    ) -> Matrix:
        """
        Solves the 1D Heat Equation u_t = alpha * u_xx using Finite Differences.
        :return: A matrix where rows are time steps and columns are spatial points.
        """
        dx = L / (nx - 1)
        dt = T / (nt - 1)
        r = alpha * dt / (dx**2)
        if r > 0.5:
            print(f"Warning: Stability condition not met (r = {r:.4f} > 0.5). "
                  "Solution may be unstable.")

        u = np.zeros((nt, nx))
        # Set initial condition
        u[0, :] = [initial_condition(i * dx) for i in range(nx)]
        # Set boundary conditions
        bc1, bc2 = boundary_conditions
        u[:, 0] = bc1
        u[:, -1] = bc2

        # Solve using FTCS scheme
        for t in range(0, nt - 1):
            for x in range(1, nx - 1):
                u[t+1, x] = u[t, x] + r * (u[t, x+1] - 2*u[t, x] + u[t, x-1])
        
        return u.tolist()
    #... other numerical analysis methods (solvers, etc.)


# --- Primary Facade Class: Composition-based Design ---

class AxiomPy:
    def __init__(self):
        self._linalg = LinearAlgebra()
        self._stats = Statistics()
        self._optim = Optimization()
        self._ml = MachineLearning()
        self._numan = NumericalAnalysis()
        
        # Expose data types directly
        self.Rational = RationalNumber
        self.Graph = Graph

    @property
    def linalg(self) -> LinearAlgebra: return self._linalg
    @property
    def stats(self) -> Statistics: return self._stats
    @property
    def optim(self) -> Optimization: return self._optim
    @property
    def ml(self) -> MachineLearning: return self._ml
    @property
    def numerical_analysis(self) -> NumericalAnalysis: return self._numan
    @property
    def constants(self) -> Constants: return self._constants

# Create a single, ready-to-use instance
Axiom = AxiomPy()


# --- Example Usage Block ---
if __name__ == "__main__":
    print("=" * 70)
    print("    AxiomPy Mathematics Engine v1.5.0 - Advanced Demonstration")
    print("=" * 70)

    # 1. Machine Learning: K-Means Clustering
    print("\n--- 1. Machine Learning (K-Means Clustering) ---")
    # Create three distinct clusters of data
    cluster1 = np.random.randn(20, 2) + np.array([5, 5])
    cluster2 = np.random.randn(20, 2) + np.array([-5, -5])
    cluster3 = np.random.randn(20, 2) + np.array([5, -5])
    ml_data = np.vstack([cluster1, cluster2, cluster3]).tolist()
    centroids, _ = Axiom.ml.kmeans(ml_data, k=3)
    print("Found centroids for 3 data clusters:")
    for i, c in enumerate(centroids):
        print(f"  Centroid {i+1}: ({c[0]:.2f}, {c[1]:.2f})")

    # 2. Non-Linear Optimization: Multivariate Gradient Descent
    print("\n--- 2. Non-Linear Optimization (Gradient Descent) ---")
    # Minimize the paraboloid function f(x,y) = x^2 + y^2
    # The gradient is grad(f) = [2x, 2y]
    grad_f = lambda v: [2 * v[0], 2 * v[1]]
    start_point = [10.0, -8.0]
    minimum = Axiom.optim.gradient_descent(grad_f, start_point, learn_rate=0.1)
    print(f"Minimizing f(x,y) = x^2 + y^2, starting from {start_point}:")
    print(f"  Found minimum near: ({minimum[0]:.4f}, {minimum[1]:.4f}) (Exact is [0,0])")

    # 3. Numerical Analysis: PDE Solver for the Heat Equation
    print("\n--- 3. PDE Solver (1D Heat Equation) ---")
    # A rod of length 1, held at 0 degrees at both ends.
    # Initial condition: A temperature spike of 100 degrees in the middle.
    ic = lambda x: 100 * np.sin(Constants.PI * x)
    heat_simulation = Axiom.numerical_analysis.pde_solve_heat_1d(
        L=1.0, T=0.1, alpha=0.01, nx=21, nt=101,
        initial_condition=ic, boundary_conditions=(0, 0)
    )
    print("Simulating heat diffusion in a 1D rod:")
    print(f"  Temp at rod center (x=0.5) at t=0.00: {heat_simulation[0][10]:.2f}")
    print(f"  Temp at rod center (x=0.5) at t=0.05: {heat_simulation[50][10]:.2f}")
    print(f"  Temp at rod center (x=0.5) at t=0.10: {heat_simulation[-1][10]:.2f}")

    # 4. Exact Arithmetic: Rational Numbers
    print("\n--- 4. Exact Arithmetic (Rational Numbers) ---")
    r1 = Axiom.Rational(1, 3)
    r2 = Axiom.Rational(1, 6)
    r_sum = r1 + r2
    print(f"Demonstrating perfect precision (no float errors):")
    print(f"  {r1} + {r2} = {r_sum} (Simplified from 9/18)")
    r_div = r1 / Axiom.Rational(2)
    print(f"  ({r1}) / 2 = {r_div}")
    # Compare to floating point
    float_sum = 1/3 + 1/3 + 1/3
    rational_sum = Axiom.Rational(1,3) + Axiom.Rational(1,3) + Axiom.Rational(1,3)
    print(f"  Floating point: 1/3 + 1/3 + 1/3 = {float_sum}")
    print(f"  Rational: 1/3 + 1/3 + 1/3 = {rational_sum}")

    # 5. Advanced Linear Algebra: Vector Projection
    print("\n--- 5. Linear Algebra (Vector Projection) ---")
    v = [2, 3]
    u = [4, 0]
    proj_v_on_u = Axiom.linalg.project_vector(v, u)
    print(f"The projection of vector {v} onto vector {u} is {proj_v_on_u}")
    
    print("\n" + "=" * 70)
    print("               Demonstration Complete")
    print("=" * 70)
