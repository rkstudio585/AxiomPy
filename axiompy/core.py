# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 2.9.1 (Bugfix Release)
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A premier scientific computing engine. This version introduces
#                powerful, first-class Vector and Matrix objects with full
#                operator overloading, a PageRank algorithm for graph analysis,
#                and new methods for creating and analyzing statistical matrices,
#                providing an intuitive, object-oriented linear algebra framework.
#
################################################################################

import numpy as np
import math
import random
# --- FIX: `Generic` must be imported from `typing` ---
from typing import (List, Tuple, Callable, Any, TypeVar, Dict, Generic)
from numbers import Number
from collections import defaultdict

# --- Type Aliases & Core Infrastructure ---
MatrixData = List[List[float]]
VectorData = List[float]
GraphNode = TypeVar('GraphNode')
class AxiomError(Exception): pass

# --- Core Data Types with Full Operator Overloading ---

class Vector:
    def __init__(self, data: VectorData): self._data = np.array(data, dtype=float)
    def __len__(self) -> int: return len(self._data)
    def __repr__(self) -> str: return f"Vector({self._data.tolist()})"
    def __getitem__(self, key): return self._data[key]
    def to_list(self) -> VectorData: return self._data.tolist()
    
    # --- Operators ---
    def __add__(self, other): return Vector(self._data + other._data)
    def __sub__(self, other): return Vector(self._data - other._data)
    def __mul__(self, other: Any) -> Any:
        if isinstance(other, Number): return Vector(self._data * other)
        if isinstance(other, Vector): return np.dot(self._data, other._data)
        return NotImplemented
    def __matmul__(self, other): return NotImplemented
    def __rmul__(self, other: Number) -> 'Vector': return self.__mul__(other)
    def __truediv__(self, other: Number) -> 'Vector': return Vector(self._data / other)
    def __eq__(self, other: 'Vector') -> bool: return np.array_equal(self._data, other._data)
    
    # --- Methods ---
    def magnitude(self) -> float: return np.linalg.norm(self._data)
    def normalize(self) -> 'Vector':
        mag = self.magnitude()
        if mag == 0: return Vector(np.zeros_like(self._data))
        return Vector(self._data / mag)
    def angle_between(self, other: 'Vector', in_degrees: bool = False) -> float:
        dot_product = self * other
        magnitudes = self.magnitude() * other.magnitude()
        if magnitudes == 0: return 0.0
        # Clamp the value to avoid math domain errors from floating point inaccuracies
        cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad) if in_degrees else angle_rad
    def cross(self, other: 'Vector') -> 'Vector':
        if len(self) != 3 or len(other) != 3: raise AxiomError("Cross product is only defined for 3D vectors.")
        return Vector(np.cross(self._data, other._data))

class Matrix:
    def __init__(self, data: MatrixData): self._data = np.array(data, dtype=float)
    def __repr__(self) -> str:
        s = [[str(round(e, 4)) for e in row] for row in self._data]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = ' '.join('{{:>{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        return "Matrix(\n  " + "\n  ".join(table) + "\n)"
    def to_list(self) -> MatrixData: return self._data.tolist()

    # --- Properties ---
    @property
    def shape(self) -> Tuple[int, int]: return self._data.shape
    @property
    def T(self) -> 'Matrix': return Matrix(self._data.T)
    @property
    def determinant(self) -> float: return np.linalg.det(self._data)
    @property
    def inverse(self) -> 'Matrix': return Matrix(np.linalg.inv(self._data))
    @property
    def trace(self) -> float: return np.trace(self._data)
    @property
    def rank(self) -> int: return np.linalg.matrix_rank(self._data)
    
    # --- Operators ---
    def __add__(self, other): return Matrix(self._data + other._data)
    def __sub__(self, other): return Matrix(self._data - other._data)
    def __mul__(self, other: Number): return Matrix(self._data * other)
    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, Matrix): return Matrix(self._data @ other._data)
        if isinstance(other, Vector): return Vector(self._data @ other._data)
        return NotImplemented
    def __pow__(self, power: int) -> 'Matrix':
        return Matrix(np.linalg.matrix_power(self._data, power))
    def __eq__(self, other: 'Matrix') -> bool: return np.array_equal(self._data, other._data)

# --- Scientific and Mathematical Domain Classes ---

class LinearAlgebra:
    """A factory for creating common matrices."""
    @staticmethod
    def identity(n: int) -> Matrix: return Matrix(np.identity(n))
    @staticmethod
    def zeros(shape: Tuple[int, int]) -> Matrix: return Matrix(np.zeros(shape))
    @staticmethod
    def ones(shape: Tuple[int, int]) -> Matrix: return Matrix(np.ones(shape))

class Statistics:
    @staticmethod
    def covariance_matrix(data: Matrix) -> Matrix:
        return Matrix(np.cov(data._data, rowvar=False))

class Graph(Generic[GraphNode]):
    def __init__(self): self.adj: Dict[GraphNode, List[GraphNode]] = defaultdict(list)
    def add_edge(self, u, v): self.adj[u].append(v)
    def to_adjacency_matrix(self) -> Tuple[Matrix, Dict[GraphNode, int]]:
        nodes = sorted(list(self.adj.keys()))
        node_map = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        adj_matrix = np.zeros((n, n))
        for u, neighbors in self.adj.items():
            for v in neighbors:
                if v in node_map: # Ensure neighbor is also a key node
                    adj_matrix[node_map[u], node_map[v]] = 1
        return Matrix(adj_matrix), node_map

class GraphAnalysis:
    @staticmethod
    def pagerank(
        graph: Graph, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> Dict[GraphNode, float]:
        M_matrix, node_map = graph.to_adjacency_matrix()
        n = M_matrix.shape[0]
        if n == 0: return {}
        
        out_degree = M_matrix._data.sum(axis=1, keepdims=True)
        # Handle dangling nodes
        dangling_nodes = np.where(out_degree == 0)[0]
        for node_idx in dangling_nodes:
             M_matrix._data[node_idx, :] = 1/n

        M_hat = np.divide(M_matrix._data, M_matrix._data.sum(axis=1, keepdims=True), where=M_matrix._data.sum(axis=1, keepdims=True)!=0)
        
        teleport = np.full((n,n), 1/n)
        M = damping * M_hat.T + (1 - damping) * teleport
        
        ranks = Vector(np.full(n, 1/n))
        for _ in range(max_iter):
            prev_ranks = ranks
            ranks = Vector(M @ ranks._data)
            if (ranks - prev_ranks).magnitude() < tol: break
        
        inv_node_map = {i: node for node, i in node_map.items()}
        return {inv_node_map[i]: rank for i, rank in enumerate(ranks.to_list())}

# --- Primary Facade Class ---
from functools import reduce
from collections import namedtuple

class AutoDiff:
    """Dynamic computation graph and reverse-mode automatic differentiation."""
    class Variable:
        def __init__(self, value: float, _children: set = set(), _op: str = ''):
            self.value = value; self.grad = 0.0
            self._backward = lambda: None; self._prev = _children
        def __repr__(self): return f"Variable(value={self.value:.4f}, grad={self.grad:.4f})"
        def __add__(self, other):
            other = other if isinstance(other, AutoDiff.Variable) else AutoDiff.Variable(other)
            out = AutoDiff.Variable(self.value + other.value, {self, other})
            def _backward(): self.grad += out.grad; other.grad += out.grad
            out._backward = _backward; return out
        def __mul__(self, other):
            other = other if isinstance(other, AutoDiff.Variable) else AutoDiff.Variable(other)
            out = AutoDiff.Variable(self.value * other.value, {self, other})
            def _backward(): self.grad += other.value*out.grad; other.grad += self.value*out.grad
            out._backward = _backward; return out
        def __pow__(self, other: Number):
            out = AutoDiff.Variable(self.value ** other, {self})
            def _backward(): self.grad += (other * self.value**(other - 1)) * out.grad
            out._backward = _backward; return out
        def __neg__(self): return self * -1
        def __sub__(self, other): return self + (-other)
        def __truediv__(self, other): return self * other**-1
        def __radd__(self, other): return self + other
        def __rmul__(self, other): return self * other
        def __rsub__(self, other): return other + (-self)
        def __rtruediv__(self, other): return other * self**-1
        def backward(self):
            topo, visited = [], set()
            def build_topo(v):
                if v not in visited: visited.add(v); [build_topo(c) for c in v._prev]; topo.append(v)
            build_topo(self); self.grad = 1.0
            for v in reversed(topo): v._backward()

    @staticmethod
    def sin(v: 'AutoDiff.Variable'):
        out = AutoDiff.Variable(math.sin(v.value), {v})
        def _backward(): v.grad += math.cos(v.value) * out.grad
        out._backward = _backward; return out
    @staticmethod
    def exp(v: 'AutoDiff.Variable'):
        out = AutoDiff.Variable(math.exp(v.value), {v})
        def _backward(): v.grad += out.value * out.grad
        out._backward = _backward; return out
    @staticmethod
    def log(v: 'AutoDiff.Variable'):
        out = AutoDiff.Variable(math.log(v.value), {v})
        def _backward(): v.grad += (1 / v.value) * out.grad
        out._backward = _backward; return out

class NumberTheory:
    @staticmethod
    def extended_gcd(a, b):
        if a == 0: return b, 0, 1
        g, x1, y1 = NumberTheory.extended_gcd(b % a, a)
        return g, y1 - (b // a) * x1, x1
    
    @staticmethod
    def mod_inverse(a, m):
        # --- FIX: Rewritten for clarity and correctness ---
        g, x, _ = NumberTheory.extended_gcd(a, m)
        if g != 1:
            raise AxiomError(f'Modular inverse does not exist for {a} and {m}')
        return x % m

    @staticmethod
    def chinese_remainder_theorem(n: List[int], a: List[int]) -> int:
        prod = reduce(lambda x, y: x * y, n)
        result = 0
        for n_i, a_i in zip(n, a):
            p = prod // n_i
            result += a_i * NumberTheory.mod_inverse(p, n_i) * p
        return result % prod

class Electromagnetism:
    K_E: float = 8.9875517923e9
    Charge = namedtuple('Charge', ['q', 'position'])
    @staticmethod
    def calculate_electric_field(charges, point) -> Vector:
        E_total = np.zeros(len(point))
        for charge in charges:
            r_vec = np.array(point) - np.array(charge.position)
            r_mag_sq = np.sum(r_vec**2)
            if r_mag_sq < 1e-18: continue
            E_total += (Electromagnetism.K_E * charge.q / r_mag_sq**1.5) * r_vec
        return E_total.tolist()
        
class Visualization:
    @staticmethod
    def plot_ascii(x_data: Vector, y_data: Vector, width: int = 60, height: int = 15):
        if not x_data or not y_data: return ""
        min_x, max_x = min(x_data), max(x_data); min_y, max_y = min(y_data), max(y_data)
        plot = [[' ' for _ in range(width)] for _ in range(height)]
        for x, y in zip(x_data, y_data):
            px = int((x-min_x)/(max_x-min_x)*(width-1)) if max_x>min_x else 0
            py = int((y-min_y)/(max_y-min_y)*(height-1)) if max_y>min_y else 0
            plot[height-1-py][px] = '*'
        lines = ["".join(row) for row in plot]
        y_axis = [f"{max_y:8.2f} |", " "*8+"|", f"{min_y:8.2f} |"]
        for i, line in enumerate(lines):
            prefix = y_axis[0] if i == 0 else (y_axis[2] if i == height - 1 else y_axis[1])
            print(prefix + line)
        print(" "*8 + "-"*(width+1)); print(f"{'':8} {min_x:<8.2f}{'':^{width-20}}{max_x:>8.2f}")
    @staticmethod
    def plot_field_ascii(field_fn, center, size, width=30, height=15):
        arrows = {(1,0):'>', (-1,0):'<', (0,1):'^', (0,-1):'v'}
        for j in range(height, 0, -1):
            line = ""
            for i in range(width):
                x = center[0]-size/2 + i*size/width; y = center[1]-size/2 + j*size/height
                E = field_fn([x, y, 0] if len(center)==3 else [x,y])
                norm_E = np.array(E)/(np.linalg.norm(E)+1e-9)
                key = (round(norm_E[0]), round(norm_E[1]))
                line += arrows.get(key, '.')
            print(f"  {line}")

class AxiomPy:
    def __init__(self):
        self._linalg = LinearAlgebra(); self._stats = Statistics()
        self._graph_analysis = GraphAnalysis()
        self._autodiff=AutoDiff(); self._num_theory=NumberTheory()
        self._em=Electromagnetism(); self._viz=Visualization()
        self.Vector = Vector; self.Matrix = Matrix; self.Graph = Graph
    @property
    def linalg(self): return self._linalg
    @property
    def stats(self): return self._stats
    @property
    def graph_analysis(self): return self._graph_analysis
    @property
    def autodiff(self): return self._autodiff
    @property
    def number_theory(self): return self._num_theory
    @property
    def electromagnetism(self): return self._em
    @property
    def viz(self): return self._viz

Axiom = AxiomPy()

# --- Example Usage Block ---
def demo_matrix_operations():
    print("\n--- 1. Intuitive Matrix and Vector Operations ---")
    M = Axiom.Matrix([[1, 2], [3, 4]])
    v = Axiom.Vector([5, 6])
    I = Axiom.linalg.identity(2)
    print(f"Matrix M:\n{M}")
    print(f"Vector v: {v}")
    Mv = M @ v; print(f"M @ v = {Mv}")
    M_sq = M @ M; print(f"M @ M:\n{M_sq}")
    print(f"M ** 2:\n{M**2}")
    print(f"M + I:\n{M + I}")
    
def demo_matrix_methods():
    print("\n--- 2. Direct Access to Matrix Properties ---")
    A = Axiom.Matrix([[4, 7], [2, 6]])
    print(f"Matrix A:\n{A}")
    print(f"  A.shape: {A.shape}"); print(f"  A.determinant: {A.determinant:.2f}")
    print(f"  A.trace: {A.trace:.2f}"); print(f"  A.inverse:\n{A.inverse}")

def demo_vector_enhancements():
    print("\n--- 3. Enhanced Vector Methods ---")
    v1 = Axiom.Vector([1, 0, 0]); v2 = Axiom.Vector([0, 1, 0])
    print(f"v1 = {v1}, v2 = {v2}")
    print(f"  v1.cross(v2): {v1.cross(v2)}  (should be [0,0,1])")
    print(f"  v1.angle_between(v2, in_degrees=True): {v1.angle_between(v2, True)}°")

def demo_pagerank():
    print("\n--- 4. Graph Analysis: PageRank Algorithm ---")
    g = Axiom.Graph()
    # Ensure all nodes are added to the graph's keys
    g.adj['A']; g.adj['B']; g.adj['C']; g.adj['D']
    g.add_edge('A', 'B'); g.add_edge('A', 'C'); g.add_edge('B', 'C');
    g.add_edge('C', 'A'); g.add_edge('D', 'C');
    ranks = Axiom.graph_analysis.pagerank(g)
    print("Calculated PageRank for a simple web graph:")
    for node, rank in sorted(ranks.items(), key=lambda item: item[1], reverse=True):
        print(f"  Node '{node}': Rank = {rank:.4f}")
    
if __name__ == "__main__":
    print("=" * 80)
    print("    AxiomPy Mathematics Engine v2.9.1 - Corrected Demonstration")
    print("=" * 80)
    demo_matrix_operations()
    demo_matrix_methods()
    demo_vector_enhancements()
    demo_pagerank()
    print("\n" * 2)
    print("                          Demonstration Complete")
    print("=" * 80)
