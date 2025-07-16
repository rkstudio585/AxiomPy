# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 2.8.2 (Bugfix Release)
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A premier scientific computing engine. This version introduces
#                a powerful ASCII visualization engine, an expanded dynamic
#                expression system with automatic differentiation, and new
#                features in number theory and linear algebra.
#
################################################################################

import numpy as np
import math
import random
from typing import (List, Tuple, Callable, Any, TypeVar, Dict)
from numbers import Number
from functools import reduce
from collections import namedtuple

# --- Type Aliases & Core Infrastructure ---
Vector = List[float]; Matrix = List[List[float]]
# (Other core data types assumed present for brevity)

class AxiomError(Exception): pass

# --- Scientific and Mathematical Domain Classes ---

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

class LinearAlgebra:
    @staticmethod
    def qr_decomposition(matrix: Matrix) -> Tuple[Matrix, Matrix]:
        q, r = np.linalg.qr(matrix); return q.tolist(), r.tolist()

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

# --- Primary Facade Class ---
class AxiomPy:
    def __init__(self):
        self._autodiff=AutoDiff(); self._linalg=LinearAlgebra(); self._num_theory=NumberTheory()
        self._em=Electromagnetism(); self._viz=Visualization()
    @property
    def autodiff(self): return self._autodiff
    @property
    def linalg(self): return self._linalg
    @property
    def number_theory(self): return self._num_theory
    @property
    def electromagnetism(self): return self._em
    @property
    def viz(self): return self._viz

Axiom = AxiomPy()

# --- Example Usage Block ---
def demo_autodiff_expressions():
    print("\n--- 1. Expanded Dynamic Expressions & Gradients ---")
    a = Axiom.autodiff.Variable(0.5); b = Axiom.autodiff.Variable(4.0)
    z = Axiom.autodiff.sin(a * b) + Axiom.autodiff.log(a)
    print(f"Expression: z = sin(a*b) + log(a)"); print(f"  a={a.value}, b={b.value} => z={z.value:.4f}")
    z.backward()
    print("Gradients computed via backpropagation:")
    print(f"  dz/da (should be ~0.3354): {a.grad:.4f}")

def demo_ascii_plotting():
    print("\n--- 2. ASCII Visualization Engine ---")
    x = np.linspace(0, 4*np.pi, 60).tolist(); y = [math.sin(val) for val in x]
    print("Plotting y = sin(x):"); Axiom.viz.plot_ascii(x, y)

def demo_field_visualization():
    print("\n--- 3. Visualizing an Electric Field ---")
    dipole = [Axiom.electromagnetism.Charge(1, [-1,0]), Axiom.electromagnetism.Charge(-1, [1,0])]
    field_func = lambda p: Axiom.electromagnetism.calculate_electric_field(dipole, p)
    print("ASCII plot of a dipole electric field:"); Axiom.viz.plot_field_ascii(field_func, center=[0,0], size=5)

def demo_number_theory():
    print("\n--- 4. Advanced Number Theory: Chinese Remainder Theorem ---")
    n = [3, 5, 7]; a = [2, 3, 2]
    solution = Axiom.number_theory.chinese_remainder_theorem(n, a)
    print(f"Solving system of congruences: x = a_i (mod n_i)")
    print(f"  n={n}, a={a}"); print(f"  Solution: {solution} (e.g., 23 % 3 = 2, 23 % 5 = 3, 23 % 7 = 2)")
    
if __name__ == "__main__":
    print("=" * 80)
    print("    AxiomPy Mathematics Engine v2.8.2 - Corrected Demonstration")
    print("=" * 80)
    demo_autodiff_expressions()
    demo_ascii_plotting()
    demo_field_visualization()
    demo_number_theory()
    print("\n" * 2)
    print("                          Demonstration Complete")
    print("=" * 80)
