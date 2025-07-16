# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 1.7.0
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A premier scientific computing module for Python. This version
#                introduces a from-scratch Neural Network engine, the RSA
#                public-key cryptography algorithm, a KD-Tree for spatial
#                searches, and advanced numerical/linear algebra methods like
#                Gaussian Quadrature and Singular Value Decomposition (SVD).
#
################################################################################

import numpy as np
from functools import reduce
from typing import (List, Tuple, Callable, Dict, Union, TypeVar, Optional)
import random
import math

# --- Type Aliases & Core Data Types ---
Vector = List[float]; Matrix = List[List[float]]
# (Assuming full implementation of RationalNumber, Graph, etc. for brevity)

class NumberTheory:
    @staticmethod
    def is_prime(n: int, k: int = 128) -> bool:
        """ Miller-Rabin primality test. """
        if n < 2: return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p: return True
            if n % p == 0: return False
        d, s = n - 1, 0
        while d % 2 == 0: d //= 2; s += 1
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            if x == 1 or x == n - 1: continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1: break
            else: return False
        return True
    @staticmethod
    def generate_prime(bits: int) -> int:
        while True:
            p = random.getrandbits(bits)
            if p % 2 != 0 and NumberTheory.is_prime(p):
                return p

class AdvancedOps:
    @staticmethod
    def extended_gcd(a,b): #...
        if a == 0: return b, 0, 1
        g, x1, y1 = AdvancedOps.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1; y = x1
        return g, x, y
    @staticmethod
    def mod_inverse(a, m):
        g, x, _ = AdvancedOps.extended_gcd(a, m)
        if g != 1: raise Exception('Modular inverse does not exist')
        return x % m

# --- Scientific and Mathematical Domain Classes ---

class LinearAlgebra:
    @staticmethod
    def eigenvalues_eigenvectors(matrix: Matrix) -> Tuple[Vector, Matrix]: #...
        w,v=np.linalg.eig(matrix); idx=w.argsort(); return w[idx].tolist(), v[:,idx].T.tolist()
    @staticmethod
    def svd(matrix: Matrix) -> Tuple[Matrix, Vector, Matrix]:
        """Performs Singular Value Decomposition (SVD): A = U * Sigma * V^T."""
        U, s, Vh = np.linalg.svd(matrix, full_matrices=True)
        return U.tolist(), s.tolist(), Vh.tolist()

class NumericalAnalysis:
    # 5-point weights and abscissae for Gaussian Quadrature
    _GAUSS_POINTS = {
        'abscissae': [-0.90617985, -0.53846931, 0.0, 0.53846931, 0.90617985],
        'weights': [0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689]
    }
    @staticmethod
    def gaussian_quadrature(func: Callable[[float], float], a: float, b: float) -> float:
        """Approximates the integral of func from a to b using 5-point Gaussian Quadrature."""
        c1, c2 = (b - a) / 2.0, (b + a) / 2.0
        integral = 0.0
        for i in range(len(NumericalAnalysis._GAUSS_POINTS['abscissae'])):
            x_i = NumericalAnalysis._GAUSS_POINTS['abscissae'][i]
            w_i = NumericalAnalysis._GAUSS_POINTS['weights'][i]
            integral += w_i * func(c1 * x_i + c2)
        return c1 * integral

class Cryptography:
    """Implementations of cryptographic algorithms."""
    @staticmethod
    def rsa_generate_keys(bits: int = 128) -> Dict:
        p = NumberTheory.generate_prime(bits)
        q = NumberTheory.generate_prime(bits)
        while p == q: q = NumberTheory.generate_prime(bits)
        n = p * q
        phi_n = (p - 1) * (q - 1)
        e = 65537  # Common choice for e
        d = AdvancedOps.mod_inverse(e, phi_n)
        return {'public': (e, n), 'private': (d, n)}
    @staticmethod
    def rsa_encrypt(public_key: Tuple[int, int], message: int) -> int:
        e, n = public_key
        return pow(message, e, n)
    @staticmethod
    def rsa_decrypt(private_key: Tuple[int, int], ciphertext: int) -> int:
        d, n = private_key
        return pow(ciphertext, d, n)

class Spatial:
    """Data structures and algorithms for spatial problems."""
    class _KDNode:
        def __init__(self, point, axis, left=None, right=None):
            self.point = point; self.axis = axis; self.left = left; self.right = right

    class KDTree:
        def __init__(self, data: Matrix):
            def build(points, depth=0):
                if not points: return None
                k = len(points[0])
                axis = depth % k
                points.sort(key=lambda x: x[axis])
                median = len(points) // 2
                return Spatial._KDNode(points[median], axis,
                                       build(points[:median], depth+1),
                                       build(points[median+1:], depth+1))
            self.root = build(list(data))
        
        def find_nearest(self, query_point: Vector) -> Tuple[Vector, float]:
            best = None
            def search(node):
                nonlocal best
                if node is None: return
                
                dist_sq = sum((a-b)**2 for a,b in zip(node.point, query_point))
                if best is None or dist_sq < best[1]:
                    best = (node.point, dist_sq)

                axis = node.axis
                diff = query_point[axis] - node.point[axis]
                
                close, away = (node.left, node.right) if diff < 0 else (node.right, node.left)
                
                search(close)
                if diff**2 < best[1]:
                    search(away)

            search(self.root)
            return (best[0], math.sqrt(best[1])) if best else (None, float('inf'))

class MachineLearning:
    class NeuralNetwork:
        def __init__(self, layer_sizes: List[int], learn_rate: float = 0.1):
            self.learn_rate = learn_rate
            self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
            self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        @staticmethod
        def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
        @staticmethod
        def _sigmoid_derivative(z): s = MachineLearning.NeuralNetwork._sigmoid(z); return s * (1 - s)
        def feedforward(self, a: np.ndarray) -> np.ndarray:
            for w, b in zip(self.weights, self.biases):
                a = self._sigmoid(np.dot(w, a) + b)
            return a
        def backpropagate(self, x, y):
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            
            activation = x
            activations = [x]
            zs = []
            
            for w, b in zip(self.weights, self.biases):
                z = np.dot(w, activation) + b
                zs.append(z)
                activation = self._sigmoid(z)
                activations.append(activation)
            
            delta = (activations[-1] - y) * self._sigmoid_derivative(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].T)

            for l in range(2, len(self.weights) + 1):
                z = zs[-l]
                sp = self._sigmoid_derivative(z)
                delta = np.dot(self.weights[-l+1].T, delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].T)
            
            return nabla_w, nabla_b
        def train(self, training_data: List[Tuple[Vector, Vector]], epochs: int):
            for j in range(epochs):
                for x_vec, y_vec in training_data:
                    x = np.array(x_vec).reshape(-1, 1)
                    y = np.array(y_vec).reshape(-1, 1)
                    delta_nabla_w, delta_nabla_b = self.backpropagate(x, y)
                    self.weights = [w - self.learn_rate*nw for w, nw in zip(self.weights, delta_nabla_w)]
                    self.biases = [b - self.learn_rate*nb for b, nb in zip(self.biases, delta_nabla_b)]
        def predict(self, x_vec: Vector) -> Vector:
            return self.feedforward(np.array(x_vec).reshape(-1, 1)).flatten().tolist()

# --- Primary Facade Class: Composition-based Design ---

class AxiomPy:
    def __init__(self):
        self._linalg = LinearAlgebra()
        self._numan = NumericalAnalysis()
        self._crypto = Cryptography()
        self._spatial = Spatial()
        self._ml = MachineLearning()
    @property
    def linalg(self): return self._linalg
    @property
    def numerical_analysis(self): return self._numan
    @property
    def crypto(self): return self._crypto
    @property
    def spatial(self): return self._spatial
    @property
    def ml(self): return self._ml

Axiom = AxiomPy()

# --- Example Usage Block ---
if __name__ == "__main__":
    print("=" * 75)
    print("    AxiomPy Mathematics Engine v1.7.0 - PhD-Level Demonstration")
    print("=" * 75)

    # 1. Artificial Intelligence: Solving the XOR Problem with a Neural Network
    print("\n--- 1. AI: From-Scratch Neural Network solving XOR ---")
    xor_data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
    net = Axiom.ml.NeuralNetwork([2, 3, 1], learn_rate=0.5)
    net.train(xor_data, epochs=10000)
    print("Trained a neural network to solve the non-linear XOR problem.")
    for inp, _ in xor_data:
        print(f"  Input: {inp} -> Output: {net.predict(inp)[0]:.4f}")

    # 2. Cryptography: RSA Public-Key Encryption
    print("\n--- 2. Cryptography: RSA Algorithm from First Principles ---")
    keys = Axiom.crypto.rsa_generate_keys(bits=64)
    message = 123456789
    print(f"Original message: {message}")
    ciphertext = Axiom.crypto.rsa_encrypt(keys['public'], message)
    print(f"Encrypted ciphertext: {ciphertext}")
    decrypted_message = Axiom.crypto.rsa_decrypt(keys['private'], ciphertext)
    print(f"Decrypted message: {decrypted_message}")
    
    # 3. Spatial Data Structures: KD-Tree for Nearest Neighbor Search
    print("\n--- 3. Spatial Data Structures: KD-Tree Search ---")
    points = [[2,3], [5,4], [9,6], [4,7], [8,1], [7,2]]
    kdtree = Axiom.spatial.KDTree(points)
    query = [9, 2]
    nearest_point, distance = kdtree.find_nearest(query)
    print(f"Built a KD-Tree with points: {points}")
    print(f"The nearest point to {query} is {nearest_point} at a distance of {distance:.4f}")

    # 4. High-Precision Numerical Integration: Gaussian Quadrature
    print("\n--- 4. Numerical Analysis: Gaussian Quadrature ---")
    # Integrate f(x) = x^3 * sin(x) from 0 to pi
    func_to_integrate = lambda x: x**3 * math.sin(x)
    analytical_result = math.pi**3 - 6*math.pi
    numerical_result = Axiom.numerical_analysis.gaussian_quadrature(func_to_integrate, 0, math.pi)
    print("Integrating f(x) = x^3 * sin(x) from 0 to pi:")
    print(f"  Numerical Result (5-point Gaussian): {numerical_result:.8f}")
    print(f"  Analytical Result:                   {analytical_result:.8f}")

    # 5. Linear Algebra: Singular Value Decomposition (SVD)
    print("\n--- 5. Linear Algebra: Singular Value Decomposition ---")
    A = [[1, 2, 3], [4, 5, 6]]
    U, s, Vh = Axiom.linalg.svd(A)
    print("Decomposing a 2x3 matrix A using SVD (A = U * Sigma * V^T):")
    print(f"  Singular values (Sigma): {[f'{val:.4f}' for val in s]}")
    print(f"  Left singular vectors (U): \n{np.round(U, 4)}")

    print("\n" + "=" * 75)
    print("                     Demonstration Complete")
    print("=" * 75)
