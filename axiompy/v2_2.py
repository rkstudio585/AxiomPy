# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 2.2.0
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A premier scientific computing engine. This version introduces
#                full operator overloading for a new, powerful Vector class,
#                a PID Controller for industrial control systems, ANOVA for
#                advanced statistical testing, and Elliptic Curve algebra, the
#                foundation of modern public-key cryptography.
#
################################################################################

import numpy as np
import random
import math
from typing import (List, Tuple, Callable, Any, TypeVar, Set, Generic, Dict)
from numbers import Number

# --- Type Aliases & Core Infrastructure ---
Matrix = List[List[float]]

class AxiomError(Exception): pass
class ConvergenceError(AxiomError): pass

# --- Core Data Types with Full Operator Overloading ---

class Vector:
    """A robust Vector class with full operator support."""
    def __init__(self, data: List[float]):
        self._data = np.array(data, dtype=float)
    def __len__(self) -> int: return len(self._data)
    def __repr__(self) -> str: return f"Vector({self._data.tolist()})"
    def __getitem__(self, key): return self._data[key]
    
    # 1. Arithmetic Operators
    def __add__(self, other: 'Vector') -> 'Vector':
        if len(self) != len(other): raise AxiomError("Vectors must have the same dimension for addition.")
        return Vector(self._data + other._data)
    def __sub__(self, other: 'Vector') -> 'Vector':
        if len(self) != len(other): raise AxiomError("Vectors must have the same dimension for subtraction.")
        return Vector(self._data - other._data)
    def __mul__(self, other: Any) -> Any:
        if isinstance(other, Number): # Scalar multiplication
            return Vector(self._data * other)
        if isinstance(other, Vector): # Dot product
            if len(self) != len(other): raise AxiomError("Vectors must have the same dimension for dot product.")
            return np.dot(self._data, other._data)
        return NotImplemented
    def __rmul__(self, other: Number) -> 'Vector': return self.__mul__(other)
    def __truediv__(self, other: Number) -> 'Vector':
        if not isinstance(other, Number): return NotImplemented
        return Vector(self._data / other)
    
    # 2. Comparison Operators
    def __eq__(self, other: 'Vector') -> bool: return np.array_equal(self._data, other._data)
    def __ne__(self, other: 'Vector') -> bool: return not self.__eq__(other)
    def magnitude(self) -> float: return np.linalg.norm(self._data)
    def __lt__(self, other: 'Vector') -> bool: return self.magnitude() < other.magnitude()
    def __le__(self, other: 'Vector') -> bool: return self.magnitude() <= other.magnitude()
    def __gt__(self, other: 'Vector') -> bool: return self.magnitude() > other.magnitude()
    def __ge__(self, other: 'Vector') -> bool: return self.magnitude() >= other.magnitude()

    # 3. Assignment Operators (In-place)
    def __iadd__(self, other: 'Vector') -> 'Vector':
        self._data += other._data; return self
    def __isub__(self, other: 'Vector') -> 'Vector':
        self._data -= other._data; return self
    def __imul__(self, other: Number) -> 'Vector':
        self._data *= other; return self
        
    def to_list(self) -> List[float]: return self._data.tolist()

# (Quaternion, RationalNumber, Polynomial etc. can also have these overloads)

# --- Scientific and Mathematical Domain Classes ---

class Statistics:
    @staticmethod
    def mean(data: List[float]) -> float: return sum(data) / len(data) if data else 0
    @staticmethod
    def variance(data: List[float]) -> float:
        n = len(data); mean = Statistics.mean(data)
        return sum((x - mean) ** 2 for x in data) / (n - 1)
    
    @staticmethod
    def one_way_anova(*groups: List[float]) -> Dict[str, float]:
        """
        Performs a One-Way Analysis of Variance (ANOVA).
        Returns a dictionary with F-statistic and p-value.
        """
        all_data = [item for group in groups for item in group]
        overall_mean = Statistics.mean(all_data)
        
        n_total = len(all_data)
        k_groups = len(groups)
        
        # Sum of Squares Between groups (SSB)
        ssb = sum(len(g) * (Statistics.mean(g) - overall_mean)**2 for g in groups)
        df_between = k_groups - 1
        msb = ssb / df_between
        
        # Sum of Squares Within groups (SSW)
        ssw = sum(sum((x - Statistics.mean(g))**2 for x in g) for g in groups)
        df_within = n_total - k_groups
        msw = ssw / df_within
        
        if msw == 0: return {'f_stat': float('inf'), 'p_value': 0.0}
        
        f_stat = msb / msw
        # p-value would typically come from an F-distribution table or scipy.stats.f.sf
        # For this from-scratch library, we return the F-statistic as the key result.
        return {'f_stat': f_stat, 'df': (df_between, df_within)}

class ControlSystems:
    """Algorithms for modeling and simulating dynamical systems."""
    class PIDController:
        def __init__(self, Kp: float, Ki: float, Kd: float, setpoint: float):
            self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
            self.setpoint = setpoint
            self._integral = 0
            self._previous_error = 0
        def update(self, current_value: float, dt: float) -> float:
            error = self.setpoint - current_value
            self._integral += error * dt
            derivative = (error - self._previous_error) / dt
            self._previous_error = error
            return self.Kp * error + self.Ki * self._integral + self.Kd * derivative

class Cryptography:
    """Implementations of cryptographic algorithms."""
    class EllipticCurve:
        def __init__(self, a: int, b: int, p: int):
            self.a, self.b, self.p = a, b, p
            # Point at infinity
            self.infinity = self.Point(None, None)
        class Point:
            def __init__(self, x, y): self.x, self.y = x, y
            def __eq__(self, other): return self.x == other.x and self.y == other.y
            def __repr__(self): return f"Point({self.x}, {self.y})" if self.x is not None else "Point(Infinity)"
        
        def _mod_inverse(self, n): return pow(n, self.p - 2, self.p)

        def add(self, p1: Point, p2: Point) -> Point:
            if p1 == self.infinity: return p2
            if p2 == self.infinity: return p1
            if p1.x == p2.x and p1.y != p2.y: return self.infinity

            if p1.x == p2.x: # Point doubling
                m = (3 * p1.x**2 + self.a) * self._mod_inverse(2 * p1.y)
            else: # Point addition
                m = (p2.y - p1.y) * self._mod_inverse(p2.x - p1.x)
            
            m %= self.p
            x3 = (m**2 - p1.x - p2.x) % self.p
            y3 = (m * (p1.x - x3) - p1.y) % self.p
            return self.Point(x3, y3)

# --- Primary Facade Class: Composition-based Design ---

class AxiomPy:
    def __init__(self):
        self._stats = Statistics()
        self._control = ControlSystems()
        self._crypto = Cryptography()
        # Expose data types directly
        self.Vector = Vector
    @property
    def stats(self) -> Statistics: return self._stats
    @property
    def control(self) -> ControlSystems: return self._control
    @property
    def crypto(self) -> Cryptography: return self._crypto

Axiom = AxiomPy()

# --- Example Usage Block ---
def demo_operators():
    print("\n--- 1. Intuitive Operator Overloading for a new Vector class ---")
    v1 = Axiom.Vector([1, 2, 3])
    v2 = Axiom.Vector([4, 5, 6])
    print(f"v1 = {v1}, v2 = {v2}")
    # Arithmetic
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 * 3  = {v1 * 3}")
    # Dot product
    print(f"v1 * v2 (Dot Product) = {v1 * v2}")
    # Comparison
    v3 = Axiom.Vector([1, 1, 1])
    print(f"Magnitude of v1: {v1.magnitude():.2f}")
    print(f"Magnitude of v3: {v3.magnitude():.2f}")
    print(f"Is v1 > v3? {v1 > v3}")

def demo_pid_controller():
    print("\n--- 2. Control Systems: PID Controller simulation ---")
    # Simulate heating a room
    pid = Axiom.control.PIDController(Kp=0.5, Ki=0.1, Kd=0.05, setpoint=20.0)
    temp = 15.0
    print(f"Target temperature: {pid.setpoint}°C. Starting at {temp}°C.")
    for i in range(5):
        control_output = pid.update(temp, dt=1.0)
        # Simulate simple physics: temp changes based on heater output and heat loss
        temp += control_output * 0.1 - 0.2
        print(f"  Time {i+1}: Temp={temp:.2f}°C, Heater Output={control_output:.2f}")

def demo_anova():
    print("\n--- 3. Statistics: One-Way ANOVA Test ---")
    # Effectiveness of three different fertilizers
    group_a = [22, 24, 21, 25, 20]
    group_b = [28, 30, 29, 27, 26]
    group_c = [20, 19, 22, 21, 18]
    print("Testing if there is a significant difference between fertilizer groups:")
    print(f"  Group A: {group_a}")
    print(f"  Group B: {group_b}")
    print(f"  Group C: {group_c}")
    result = Axiom.stats.one_way_anova(group_a, group_b, group_c)
    print(f"  ANOVA Result: F-statistic = {result['f_stat']:.4f}, df = {result['df']}")
    print("  (A high F-statistic suggests a significant difference between groups)")

def demo_elliptic_curve():
    print("\n--- 4. Cryptography: Elliptic Curve Point Addition ---")
    # Using the curve secp256k1 (used by Bitcoin) over a small prime field for demo
    # y^2 = x^3 + 7 over F_p
    curve = Axiom.crypto.EllipticCurve(a=0, b=7, p=229)
    P = curve.Point(10, 150)
    Q = curve.Point(55, 98)
    print(f"Curve: y^2 = x^3 + 7 (mod {curve.p})")
    print(f"  P = {P}")
    print(f"  Q = {Q}")
    R = curve.add(P, Q)
    print(f"  P + Q = {R}")
    
if __name__ == "__main__":
    print("=" * 80)
    print("    AxiomPy Mathematics Engine v2.2.0 - Premier Scientific Demonstration")
    print("=" * 80)
    demo_operators()
    demo_pid_controller()
    demo_anova()
    demo_elliptic_curve()
    print("\n" * 2)
    print("                          Demonstration Complete")
    print("=" * 80)
