# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 2.7.0
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A premier scientific computing engine. This version introduces
#                the Conjugate Gradient method for solving large linear systems,
#                a from-scratch Decision Tree classifier for machine learning, an
#                Electromagnetism engine for field calculations, and a Topology
#                module for finding connected components in graphs.
#
################################################################################

import numpy as np
import random
import math
from collections import Counter, namedtuple
from typing import (List, Tuple, Callable, Any, TypeVar, Set, Generic, Dict)

# --- Type Aliases & Core Infrastructure ---
Vector = List[float]; Matrix = List[List[float]]
State = TypeVar('State'); Symbol = TypeVar('Symbol')
GraphNode = TypeVar('GraphNode')
# (Other core data types assumed present for brevity)

class AxiomError(Exception): pass
class ConvergenceError(AxiomError): pass

# --- Scientific and Mathematical Domain Classes ---

class LinearAlgebra:
    """Core and advanced linear algebra operations."""
    @staticmethod
    def conjugate_gradient(
        A: Matrix, b: Vector, x0: Vector = None,
        max_iter: int = 1000, tol: float = 1e-6
    ) -> Vector:
        """
        Solves a system of linear equations Ax=b for a symmetric,
        positive-definite matrix A using the Conjugate Gradient method.
        """
        A = np.array(A); b = np.array(b)
        n = len(b)
        x = np.zeros(n) if x0 is None else np.array(x0)
        
        r = b - A @ x
        p = r.copy()
        rs_old = r.T @ r
        
        for i in range(max_iter):
            Ap = A @ p
            alpha = rs_old / (p.T @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = r.T @ r
            
            if math.sqrt(rs_new) < tol: break
            
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        else:
            raise ConvergenceError("Conjugate Gradient method did not converge.")
            
        return x.tolist()
    #... Other linalg functions assumed present

class MachineLearning:
    """A collection of fundamental and advanced machine learning algorithms."""
    class DecisionTreeClassifier:
        def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
            self.max_depth, self.min_samples_split = max_depth, min_samples_split
            self.tree = None
        
        def _gini_impurity(self, y: np.ndarray) -> float:
            if y.size == 0: return 0
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / y.size
            return 1 - np.sum(probabilities**2)

        def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
            best_gini, best_feature, best_threshold = 1.0, None, None
            n_features = X.shape[1]
            
            for feature_idx in range(n_features):
                thresholds = np.unique(X[:, feature_idx])
                for threshold in thresholds:
                    y_left = y[X[:, feature_idx] < threshold]
                    y_right = y[X[:, feature_idx] >= threshold]
                    if len(y_left) == 0 or len(y_right) == 0: continue
                    
                    gini = (len(y_left)/len(y)) * self._gini_impurity(y_left) + \
                           (len(y_right)/len(y)) * self._gini_impurity(y_right)
                    
                    if gini < best_gini:
                        best_gini, best_feature, best_threshold = gini, feature_idx, threshold
                        
            return best_feature, best_threshold

        def _grow_tree(self, X, y, depth=0):
            if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
                leaf_value = Counter(y).most_common(1)[0][0]
                return {'leaf_value': leaf_value}
            
            feature, threshold = self._find_best_split(X, y)
            if feature is None:
                leaf_value = Counter(y).most_common(1)[0][0]
                return {'leaf_value': leaf_value}
                
            left_idxs = X[:, feature] < threshold
            right_idxs = X[:, feature] >= threshold
            
            left_tree = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
            right_tree = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
            
            return {'feature': feature, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

        def fit(self, X: Matrix, y: Vector):
            self.tree = self._grow_tree(np.array(X), np.array(y))

        def _predict_single(self, x, node):
            if 'leaf_value' in node: return node['leaf_value']
            if x[node['feature']] < node['threshold']:
                return self._predict_single(x, node['left'])
            return self._predict_single(x, node['right'])
            
        def predict(self, X: Matrix) -> Vector:
            return [self._predict_single(x, self.tree) for x in X]

class Electromagnetism:
    """Functions for simulating electromagnetic phenomena."""
    K_E: float = 8.9875517923e9 # Coulomb's constant
    Charge = namedtuple('Charge', ['q', 'position'])

    @staticmethod
    def calculate_electric_field(charges: List[Charge], point: Vector) -> Vector:
        """Calculates the resultant electric field from a list of point charges."""
        E_total = np.zeros(len(point))
        p = np.array(point)
        for charge in charges:
            r_vec = p - np.array(charge.position)
            r_mag = np.linalg.norm(r_vec)
            if r_mag < 1e-9: continue # Avoid singularity at charge location
            E_mag = Electromagnetism.K_E * charge.q / r_mag**2
            E_vec = E_mag * (r_vec / r_mag)
            E_total += E_vec
        return E_total.tolist()

class Graph(Generic[GraphNode]):
    def __init__(self): self.adj: Dict[GraphNode, List[GraphNode]] = {}
    def add_node(self, node: GraphNode):
        if node not in self.adj: self.adj[node] = []
    def add_edge(self, u: GraphNode, v: GraphNode):
        self.add_node(u); self.add_node(v)
        self.adj[u].append(v); self.adj[v].append(u)

class Topology:
    """Algorithms for analyzing the structural properties of spaces, like graphs."""
    @staticmethod
    def find_connected_components(graph: Graph) -> List[Set[GraphNode]]:
        """Finds all connected components (subgraphs) in a graph."""
        visited = set()
        components = []
        for node in graph.adj:
            if node not in visited:
                component = set()
                q = [node]
                visited.add(node)
                while q:
                    u = q.pop(0)
                    component.add(u)
                    for v in graph.adj[u]:
                        if v not in visited:
                            visited.add(v); q.append(v)
                components.append(component)
        return components

# --- Primary Facade Class: Composition-based Design ---

class AxiomPy:
    def __init__(self):
        self._linalg = LinearAlgebra(); self._ml = MachineLearning()
        self._em = Electromagnetism(); self._topo = Topology()
        self.Graph = Graph
    @property
    def linalg(self): return self._linalg
    @property
    def ml(self): return self._ml
    @property
    def electromagnetism(self): return self._em
    @property
    def topology(self): return self._topo

Axiom = AxiomPy()

# --- Example Usage Block ---
def demo_conjugate_gradient():
    print("\n--- 1. Numerical Linear Algebra: Conjugate Gradient Method ---")
    A = [[4, 1], [1, 3]]
    b = [1, 2]
    x = Axiom.linalg.conjugate_gradient(A, b)
    print(f"Solving Ax=b for A={A}, b={b}")
    print(f"  Calculated solution x = {[f'{val:.4f}' for val in x]} (Exact: [0.0909, 0.6364])")
    print(f"  Verification (A @ x): {[f'{val:.4f}' for val in (np.array(A) @ np.array(x))]}")

def demo_decision_tree():
    print("\n--- 2. Machine Learning: Decision Tree Classifier ---")
    # A simple, non-linearly separable dataset (a circle inside a square)
    X = [[2,2], [2,-2], [-2,2], [-2,-2], [0,1], [1,0], [-1,0], [0,-1]]
    y = [0, 0, 0, 0, 1, 1, 1, 1] # 0 = outer, 1 = inner
    tree = Axiom.ml.DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)
    print("Trained a decision tree on a 'circle-in-a-square' dataset.")
    test_points = [[3,3], [0,0]] # One outer, one inner
    predictions = tree.predict(test_points)
    print(f"  Prediction for {test_points[0]} (outer): Class {predictions[0]}")
    print(f"  Prediction for {test_points[1]} (inner): Class {predictions[1]}")

def demo_electromagnetism():
    print("\n--- 3. Electromagnetism: Calculating an Electric Field ---")
    # A simple electric dipole
    charges = [
        Axiom.electromagnetism.Charge(q=1e-9, position=[-0.1, 0, 0]), # +1 nC
        Axiom.electromagnetism.Charge(q=-1e-9, position=[0.1, 0, 0])  # -1 nC
    ]
    point_of_interest = [0, 0.1, 0]
    E_field = Axiom.electromagnetism.calculate_electric_field(charges, point_of_interest)
    print("Calculated the E-field for a dipole.")
    print(f"  E-field vector at {point_of_interest}: {[f'{val:.2f}' for val in E_field]} N/C")

def demo_topology():
    print("\n--- 4. Computational Topology: Finding Connected Components ---")
    graph = Axiom.Graph()
    graph.add_edge('A', 'B'); graph.add_edge('B', 'C') # Component 1
    graph.add_edge('D', 'E')                         # Component 2
    graph.add_node('F')                              # Component 3
    
    components = Axiom.topology.find_connected_components(graph)
    print("Identified connected components in a disconnected graph.")
    for i, comp in enumerate(components):
        print(f"  Component {i+1}: {sorted(list(comp))}")
    
if __name__ == "__main__":
    print("=" * 80)
    print("    AxiomPy Mathematics Engine v2.7.0 - Premier Scientific Demonstration")
    print("=" * 80)
    demo_conjugate_gradient()
    demo_decision_tree()
    demo_electromagnetism()
    demo_topology()
    print("\n" * 2)
    print("                          Demonstration Complete")
    print("=" * 80)
