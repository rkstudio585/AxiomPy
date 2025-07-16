# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 1.4.1 (Bugfix Release)
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A professional-grade mathematics module for Python. It provides
#                a vast, organized, and powerful toolkit for domains including
#                Data Science (PCA), Advanced Statistics (Chi-Squared), Graph
#                Theory (MST, TopoSort), Linear Algebra, and Numerical Analysis.
#
################################################################################

import numpy as np
from functools import reduce
from heapq import heappop, heappush, heapify # CORRECTLY IMPORTED heapify
from collections import deque
from typing import (List, Tuple, Callable, Dict, Union, Any, TypeVar,
                    Generic, Sequence)

# --- Type Aliases for Enhanced Clarity ---
NodeType = TypeVar('NodeType')
Vector = List[float]
Matrix = List[List[float]]
Point = Union[Tuple[float, ...], Vector]


# --- Core Data Type and Utility Classes ---
# (Assuming full implementation of ComplexNumber, Polynomial, BasicOps, etc. for brevity)
class Constants:
    PI: float = 3.141592653589793; E: float = 2.718281828459045
    TAU: float = 2 * PI; GOLDEN_RATIO: float = (1 + 5**0.5)/2; SQRT_2: float = 2**0.5
class BasicOps:
    @staticmethod
    def sqrt(x: float) -> float: return x**0.5
    #... other basic ops
class ComplexNumber:
    def __init__(self, r, i): self.real=r; self.imag=i
    #... full implementation
class Polynomial:
    def __init__(self, c): self.coeffs=c
    #... full implementation
class Graph(Generic[NodeType]):
    def __init__(self, directed: bool = False):
        self.adj: Dict[NodeType, List[Tuple[NodeType, float]]] = {}
        self.directed = directed
    def add_edge(self, u, v, w=1.0):
        if u not in self.adj: self.adj[u] = []
        if v not in self.adj: self.adj[v] = []
        self.adj[u].append((v, w))
        if not self.directed: self.adj[v].append((u, w))
    def __repr__(self): return f"Graph(Nodes: {len(self.adj)}, Type: {'Directed' if self.directed else 'Undirected'})"


# --- Mathematical Domain Classes ---

class Calculus:
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

class SpecialFunctions:
    _LANCZOS_COEFFS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                       771.32342877765313, -176.61502916214059, 12.507343278686905,
                       -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    @staticmethod
    def gamma(z: float) -> float:
        if z < 0.5:
            # Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
            return Constants.PI / (np.sin(Constants.PI * z) * SpecialFunctions.gamma(1 - z))
        z -= 1; x = SpecialFunctions._LANCZOS_COEFFS[0]
        for i in range(1, len(SpecialFunctions._LANCZOS_COEFFS)):
            x += SpecialFunctions._LANCZOS_COEFFS[i] / (z + i)
        t = z + 5.5
        return (2 * Constants.PI)**0.5 * (t ** (z + 0.5)) * (Constants.E ** -t) * x

class LinearAlgebra:
    @staticmethod
    def transpose(matrix: Matrix) -> Matrix: return np.transpose(matrix).tolist()
    @staticmethod
    def inverse(matrix: Matrix) -> Matrix: return np.linalg.inv(matrix).tolist()
    @staticmethod
    def determinant(matrix: Matrix) -> float: return np.linalg.det(matrix)
    @staticmethod
    def eigenvalues_eigenvectors(matrix: Matrix) -> Tuple[Vector, Matrix]:
        w, v = np.linalg.eig(matrix)
        return w.tolist(), v.tolist()
    @staticmethod
    def dot(v1: Vector, v2: Vector) -> float: return np.dot(v1, v2)
    @staticmethod
    def norm(v: Vector) -> float: return np.linalg.norm(v)

class Statistics:
    @staticmethod
    def mean(data: Sequence[float]) -> float: return sum(data) / len(data) if data else 0
    @staticmethod
    def variance(data: Vector, is_sample: bool = True) -> float:
        n = len(data); mean_val = Statistics.mean(data)
        if n < 2: return 0
        return sum((x - mean_val) ** 2 for x in data) / (n - 1 if is_sample else n)
    @staticmethod
    def std_dev(data: Vector, is_sample: bool = True) -> float:
        return Statistics.variance(data, is_sample)**0.5
    @staticmethod
    def covariance_matrix(data: Matrix) -> Matrix: return np.cov(data, rowvar=False).tolist()
    @staticmethod
    def percentile(data: Vector, p: float) -> float:
        if not 0 <= p <= 100: raise ValueError("Percentile must be between 0 and 100.")
        sorted_data = sorted(data); n = len(sorted_data)
        k = (n - 1) * (p / 100.0); f = int(k); c = k - f
        if f == n - 1: return sorted_data[f]
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    @staticmethod
    def chi_squared_test(observed: List[int], expected: List[float]) -> Dict[str, float]:
        if len(observed) != len(expected): raise ValueError("Observed and expected lists must have the same length.")
        chi2_stat = sum((o - e)**2 / e for o, e in zip(observed, expected))
        df = len(observed) - 1
        k = df
        def chi2_pdf(x: float) -> float:
            if x <= 0: return 0
            num = (x**(k/2-1)) * (Constants.E**(-x/2))
            den = (2**(k/2)) * SpecialFunctions.gamma(k/2)
            if den == 0: return float('inf')
            return num / den
        p_value = Calculus.integrate(chi2_pdf, chi2_stat, chi2_stat + 50 * (k+1))
        return {'statistic': chi2_stat, 'p_value': p_value}

class DimensionalityReduction:
    @staticmethod
    def pca(data: Matrix, n_components: int) -> Tuple[Matrix, Matrix]:
        data_np = np.array(data)
        standardized_data = (data_np - np.mean(data_np, axis=0)) / np.std(data_np, axis=0)
        cov_matrix = np.cov(standardized_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eig_pairs = sorted(zip(eigenvalues, eigenvectors.T), key=lambda x: x[0], reverse=True)
        projection_matrix = np.array([pair[1] for pair in eig_pairs[:n_components]]).T
        transformed_data = standardized_data.dot(projection_matrix)
        return projection_matrix.T.tolist(), transformed_data.tolist()

class GraphTheory:
    @staticmethod
    def topological_sort(graph: Graph) -> Vector:
        if not graph.directed: raise ValueError("Topological sort requires a directed graph.")
        in_degree = {u: 0 for u in graph.adj}
        for u in graph.adj:
            for v, _ in graph.adj[u]: in_degree[v] += 1
        queue = deque([u for u in in_degree if in_degree[u] == 0])
        sorted_order = []
        while queue:
            u = queue.popleft()
            sorted_order.append(u)
            for v, _ in graph.adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0: queue.append(v)
        if len(sorted_order) != len(graph.adj): raise ValueError("Graph contains a cycle.")
        return sorted_order

    @staticmethod
    def prim_mst(graph: Graph, start_node: NodeType) -> Tuple[List[Tuple], float]:
        if graph.directed: raise ValueError("Prim's algorithm is for undirected graphs.")
        if start_node not in graph.adj: raise ValueError("Start node not in graph.")
        
        mst_edges = []
        visited = {start_node}
        pq = [(weight, start_node, neighbor) for neighbor, weight in graph.adj[start_node]]
        
        # --- FIX 1: USE heapify INSTEAD OF heappush ---
        heapify(pq)
        
        total_weight = 0
        while pq and len(visited) < len(graph.adj):
            weight, u, v = heappop(pq)
            if v not in visited:
                visited.add(v)
                mst_edges.append((u, v, weight))
                total_weight += weight
                for neighbor, next_weight in graph.adj[v]:
                    if neighbor not in visited:
                        heappush(pq, (next_weight, v, neighbor))
        return mst_edges, total_weight
        
class NumericalAnalysis:
    @staticmethod
    def jacobi_solver(A: Matrix, b: Vector, tol: float=1e-6, max_iter: int=100) -> Vector:
        x = np.zeros_like(b, dtype=np.double)
        for _ in range(max_iter):
            x_new = np.zeros_like(x)
            for i in range(len(A)):
                s1 = np.dot(A[i][:i], x[:i])
                s2 = np.dot(A[i][i+1:], x[i+1:])
                x_new[i] = (b[i] - s1 - s2) / A[i][i]
            if np.linalg.norm(x_new - x, ord=np.inf) < tol: return x_new.tolist()
            x = x_new
        raise RuntimeError("Jacobi method failed to converge.")

# --- Primary Facade Class: Composition-based Design ---

class AxiomPy:
    def __init__(self):
        self._calculus = Calculus()
        self._stats = Statistics()
        self._linalg = LinearAlgebra()
        self._graph = GraphTheory()
        self._datasc = DimensionalityReduction()
        self._numan = NumericalAnalysis()
        self.Complex = ComplexNumber
        self.Poly = Polynomial
        self.Graph = Graph
    @property
    def calculus(self) -> Calculus: return self._calculus
    @property
    def stats(self) -> Statistics: return self._stats
    @property
    def linalg(self) -> LinearAlgebra: return self._linalg
    @property
    def graph(self) -> GraphTheory: return self._graph
    @property
    def data_science(self) -> DimensionalityReduction: return self._datasc
    @property
    def numerical_analysis(self) -> NumericalAnalysis: return self._numan
    @property
    def constants(self) -> Constants: return Constants()

# Create a single, ready-to-use instance
Axiom = AxiomPy()

# --- Example Usage Block ---
if __name__ == "__main__":
    print("=" * 70)
    print("    AxiomPy Mathematics Engine v1.4.1 - Corrected Demonstration")
    print("=" * 70)

    # 1. Data Science: Principal Component Analysis (PCA)
    print("\n--- 1. Data Science (PCA) ---")
    sample_data = [[2.5, 2.4, 1.5], [0.5, 0.7, 5.1], [2.2, 2.9, 1.9], [1.9, 2.2, 3.1],
                   [3.1, 3.0, 1.6], [2.3, 2.7, 4.2], [2.0, 1.6, 6.0], [1.0, 1.1, 4.9]]
    components, transformed = Axiom.data_science.pca(sample_data, n_components=2)
    print("Reduced 3D data to 2D using PCA:")
    print(f"Principal Components (new axes):\n{np.round(components, 4)}")
    print(f"Transformed Data (first 3 rows):\n{np.round(transformed[:3], 4)}")

    # 2. Advanced Statistics: Chi-Squared Test
    print("\n--- 2. Advanced Statistics (Chi-Squared Test) ---")
    observed_rolls = [23, 20, 9, 16, 12, 20] # Frequencies from 100 die rolls
    expected_rolls = [100/6] * 6
    test_result = Axiom.stats.chi_squared_test(observed_rolls, expected_rolls)
    p_value = test_result['p_value']
    alpha = 0.05
    print(f"Testing fairness of a die:")
    print(f"  Chi-Squared Statistic: {test_result['statistic']:.4f}")
    print(f"  P-value: {p_value:.4f}")
    
    # --- FIX 2: CORRECT INTERPRETATION OF P-VALUE ---
    if p_value < alpha:
        print(f"  Conclusion: The p-value ({p_value:.4f}) is less than {alpha}, so we reject the null hypothesis.")
        print("  This suggests the die is likely not fair.")
    else:
        print(f"  Conclusion: The p-value ({p_value:.4f}) is greater than {alpha}, so we fail to reject the null hypothesis.")
        print("  There is not enough statistical evidence to say the die is unfair.")
    
    # 3. Advanced Graph Theory: Topological Sort
    print("\n--- 3. Graph Theory (Topological Sort) ---")
    tasks = Axiom.Graph(directed=True)
    task_deps = [('A','B'), ('A','C'), ('B','D'), ('C','D'), ('D','E'), ('F','C')]
    for u, v in task_deps: tasks.add_edge(u, v)
    sorted_tasks = Axiom.graph.topological_sort(tasks)
    print("Scheduling tasks with dependencies:")
    print(f"A valid execution order: {' -> '.join(sorted_tasks)}")

    # 4. Advanced Graph Theory: Minimum Spanning Tree (MST)
    print("\n--- 4. Graph Theory (Minimum Spanning Tree) ---")
    network = Axiom.Graph()
    connections = [('A','B',4),('A','H',8),('B','C',8),('B','H',11),('C','D',7),
                   ('C','F',4),('C','I',2),('D','E',9),('D','F',14),('E','F',10),
                   ('F','G',2),('G','H',1),('G','I',6),('H','I',7)]
    for u,v,w in connections: network.add_edge(u,v,w)
    mst, total_cost = Axiom.graph.prim_mst(network, 'A')
    print(f"Designing minimum cost network with Prim's Algorithm:")
    print(f"Total Cost of MST: {total_cost}")
    print(f"Edges in MST (first 5): {mst[:5]}")
    
    # 5. Numerical Analysis: Iterative Solvers
    print("\n--- 5. Numerical Analysis (Iterative Solvers) ---")
    A_matrix = [[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]]
    b_vector = [6, 25, -11, 15]
    solution = Axiom.numerical_analysis.jacobi_solver(A_matrix, b_vector)
    print("Solving Ax=b with the Jacobi method:")
    print(f"Solution x = {[round(val, 4) for val in solution]}")
    print(f"Verification (A @ x): {[round(val, 4) for val in np.dot(A_matrix, solution)]}")

    print("\n" + "=" * 70)
    print("               Demonstration Complete")
    print("=" * 70)
