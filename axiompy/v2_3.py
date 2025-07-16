# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 2.3.1 (Bugfix Release)
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A premier scientific computing engine. This version introduces
#                a Dynamic Computation Graph with Automatic Differentiation,
#                allowing for expressive, dynamic formula creation and gradient
#                computation. Also adds new modules for Time Series Analysis
#                and Classical Mechanics simulations.
#
################################################################################

import numpy as np
import math
# --- FIX: `Dict` and other types must be imported from `typing` ---
from typing import (List, Tuple, Callable, Dict, Any, TypeVar, Set, Generic)
from numbers import Number

# --- Type Aliases & Core Infrastructure ---
Vector = List[float]; Matrix = List[List[float]]

class AxiomError(Exception): pass
# (Other core data types assumed present for brevity)

# --- Scientific and Mathematical Domain Classes ---

class NumericalAnalysis:
    @staticmethod
    def solve_ode(f: Callable, y0: Vector, t_span: Tuple[float, float], h: float=0.01) -> Tuple[Vector, Matrix]:
        """Solves a system of ODEs y'(t) = f(t, y) using RK4."""
        t_vals = np.arange(t_span[0], t_span[1] + h, h)
        y_vals = np.zeros((len(t_vals), len(y0)))
        y_vals[0, :] = y0
        for i in range(len(t_vals) - 1):
            t, y = t_vals[i], y_vals[i, :]
            k1 = h * np.array(f(t, y)); k2 = h * np.array(f(t + 0.5*h, y + 0.5*k1))
            k3 = h * np.array(f(t + 0.5*h, y + 0.5*k2)); k4 = h * np.array(f(t + h, y + k3))
            y_vals[i+1, :] = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return t_vals.tolist(), y_vals.tolist()

class AutoDiff:
    """Dynamic computation graph and reverse-mode automatic differentiation."""
    class Variable:
        def __init__(self, value: float, _children: set = set(), _op: str = ''):
            self.value = value
            self.grad = 0.0
            self._backward = lambda: None
            self._prev = _children
            self._op = _op

        def __repr__(self) -> str: return f"Variable(value={self.value:.4f}, grad={self.grad:.4f})"
        
        def __add__(self, other) -> 'AutoDiff.Variable':
            other = other if isinstance(other, AutoDiff.Variable) else AutoDiff.Variable(other)
            out = AutoDiff.Variable(self.value + other.value, {self, other}, '+')
            def _backward():
                self.grad += out.grad
                other.grad += out.grad
            out._backward = _backward
            return out
        
        def __mul__(self, other) -> 'AutoDiff.Variable':
            other = other if isinstance(other, AutoDiff.Variable) else AutoDiff.Variable(other)
            out = AutoDiff.Variable(self.value * other.value, {self, other}, '*')
            def _backward():
                self.grad += other.value * out.grad
                other.grad += self.value * out.grad
            out._backward = _backward
            return out
        
        def __pow__(self, other: Number) -> 'AutoDiff.Variable':
            out = AutoDiff.Variable(self.value ** other, {self}, f'**{other}')
            def _backward():
                self.grad += (other * self.value**(other - 1)) * out.grad
            out._backward = _backward
            return out

        def backward(self):
            topo, visited = [], set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev: build_topo(child)
                    topo.append(v)
            build_topo(self)
            
            self.grad = 1.0
            for v in reversed(topo): v._backward()

        def __radd__(self, other): return self + other
        def __rmul__(self, other): return self * other

class ClassicalMechanics:
    """Solvers for problems in classical mechanics."""
    @staticmethod
    def simulate_pendulum(
        L: float, theta0_deg: float, omega0: float = 0.0,
        T: float = 10.0, dt: float = 0.01,
        g: float = 9.81, damping: float = 0.1
    ) -> Dict[str, Vector]: # This line now works because Dict is imported
        """Solves the non-linear, damped pendulum ODE system."""
        theta0_rad = math.radians(theta0_deg)
        def pendulum_system(t, Y):
            theta, omega = Y
            dtheta_dt = omega
            domega_dt = -(g / L) * math.sin(theta) - damping * omega
            return [dtheta_dt, domega_dt]
        
        t, y_matrix_list = NumericalAnalysis.solve_ode(pendulum_system, [theta0_rad, omega0], (0, T), h=dt)
        y_matrix = np.array(y_matrix_list) # Convert for easy slicing
        return {
            'time': t,
            'angle_rad': y_matrix[:, 0].tolist(),
            'angular_velocity': y_matrix[:, 1].tolist()
        }

class TimeSeries:
    """Functions for time series analysis."""
    @staticmethod
    def autocorrelation(signal: Vector, max_lags: int) -> Vector:
        """Calculates the Autocorrelation Function (ACF) for a signal."""
        n = len(signal)
        mean = sum(signal) / n
        variance = sum((x - mean) ** 2 for x in signal)
        if variance == 0: return [1.0] * (max_lags + 1) # Avoid division by zero for constant signal
        
        acf = []
        for lag in range(max_lags + 1):
            covariance = sum((signal[i]-mean)*(signal[i-lag]-mean) for i in range(lag, n))
            acf.append(covariance / variance)
        return acf

# --- Primary Facade Class: Composition-based Design ---

class AxiomPy:
    def __init__(self):
        self._autodiff = AutoDiff()
        self._mechanics = ClassicalMechanics()
        self._time_series = TimeSeries()
    @property
    def autodiff(self) -> AutoDiff: return self._autodiff
    @property
    def mechanics(self) -> ClassicalMechanics: return self._mechanics
    @property
    def time_series(self) -> TimeSeries: return self._time_series

Axiom = AxiomPy()

# --- Example Usage Block ---
def demo_autodiff():
    print("\n--- 1. Dynamic Expressions with Automatic Differentiation ---")
    a = Axiom.autodiff.Variable(2.0)
    b = Axiom.autodiff.Variable(5.0)
    c = Axiom.autodiff.Variable(10.0)
    y = a * b + c**2
    print(f"Expression: y = a*b + c^2")
    print(f"  a={a.value}, b={b.value}, c={c.value}")
    print(f"  Result y = {y.value}")
    y.backward()
    print("Gradients computed via backpropagation:")
    print(f"  dy/da (should be b=5): {a.grad:.4f}")
    print(f"  dy/db (should be a=2): {b.grad:.4f}")
    print(f"  dy/dc (should be 2c=20): {c.grad:.4f}")

def demo_pendulum():
    print("\n--- 2. Classical Mechanics: Simulating a Non-Linear Pendulum ---")
    sim = Axiom.mechanics.simulate_pendulum(L=1.0, theta0_deg=90, T=5.0)
    print("Simulated a pendulum released from 90 degrees.")
    time_pts = sim['time']
    angles_deg = [math.degrees(a) for a in sim['angle_rad']]
    print(f"  Angle at t=0.0s: {angles_deg[0]:.2f}°")
    t1_idx = int(1.0 / (time_pts[1]-time_pts[0]))
    print(f"  Angle at t=1.0s: {angles_deg[t1_idx]:.2f}° (has swung past the bottom)")
    print(f"  Angle at t=5.0s: {angles_deg[-1]:.2f}° (damped)")
    
def demo_time_series():
    print("\n--- 3. Time Series Analysis: Autocorrelation Function (ACF) ---")
    period = 10
    signal = [math.sin(2 * math.pi * i / period) for i in range(100)]
    acf_values = Axiom.time_series.autocorrelation(signal, max_lags=25)
    print("Calculated ACF for a sine wave with period=10:")
    print(f"  ACF at lag 0 (perfect correlation): {acf_values[0]:.4f}")
    print(f"  ACF at lag 5 (anti-correlation):  {acf_values[5]:.4f}")
    print(f"  ACF at lag 10 (high correlation): {acf_values[10]:.4f}")
    print(f"  ACF at lag 20 (high correlation): {acf_values[20]:.4f}")

if __name__ == "__main__":
    print("=" * 80)
    print("    AxiomPy Mathematics Engine v2.3.1 - Corrected Demonstration")
    print("=" * 80)
    demo_autodiff()
    demo_pendulum()
    demo_time_series()
    print("\n" * 2)
    print("                          Demonstration Complete")
    print("=" * 80)
