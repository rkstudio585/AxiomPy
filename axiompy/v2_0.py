# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 2.0.0
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A paradigm-shifting scientific computing engine. This v2.0
#                release introduces a Kalman Filter for optimal state estimation,
#                a Support Vector Machine for machine learning, a 2D Wave
#                Equation solver for physics simulations, and a Finite Group
#                analyzer for abstract algebra.
#
################################################################################

import numpy as np
import random
from typing import (List, Tuple, Callable, Dict, Any, TypeVar, Set, Generic)

# --- Type Aliases & Core Infrastructure ---
Vector = List[float]; Matrix = List[List[float]]
Element = TypeVar('Element')

class AxiomError(Exception):
    """Base exception class for all errors raised by the AxiomPy module."""
    pass

# (Other core data types like Quaternion, RationalNumber, etc. assumed present for brevity)

# --- Scientific and Mathematical Domain Classes ---

class LinearAlgebra:
    """Core linear algebra operations, powered by NumPy for performance."""
    @staticmethod
    def inverse(matrix: Matrix) -> Matrix:
        try: return np.linalg.inv(matrix).tolist()
        except np.linalg.LinAlgError as e: raise AxiomError(f"Matrix inversion failed: {e}")
    #... Other linalg functions assumed present

class NumericalAnalysis:
    """Advanced numerical methods for solvers and simulations."""
    @staticmethod
    def solve_wave_equation_2d(
        grid_size: Tuple[int, int], L: float, T: float, c: float,
        initial_condition: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> List[Matrix]:
        """
        Solves the 2D Wave Equation u_tt = c^2 * (u_xx + u_yy)
        using the Finite Difference Method.
        :return: A list of grids representing snapshots of the system over time.
        """
        nx, ny = grid_size
        dx = L / (nx - 1)
        dt = 0.5 * dx / c  # Ensure stability (CFL condition)
        nt = int(T / dt)
        
        x = np.linspace(0, L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y)
        
        u = np.zeros((nx, ny))
        u_prev = initial_condition(X, Y)
        u_curr = np.copy(u_prev)
        
        snapshots = [u_curr.tolist()]
        r_sq = (c * dt / dx)**2
        
        for _ in range(nt):
            # Use array slicing for performance
            u[1:-1, 1:-1] = (2*u_curr[1:-1, 1:-1] - u_prev[1:-1, 1:-1] +
                             r_sq * (u_curr[2:, 1:-1] + u_curr[:-2, 1:-1] +
                                     u_curr[1:-1, 2:] + u_curr[1:-1, :-2] -
                                     4*u_curr[1:-1, 1:-1]))
            u_prev, u_curr = u_curr, u
            if _ % (nt // 10) == 0: snapshots.append(u_curr.tolist())
            
        return snapshots

class StateEstimation:
    """Algorithms for estimating the state of dynamic systems."""
    class KalmanFilter:
        def __init__(self, F: Matrix, B: Matrix, H: Matrix, Q: Matrix, R: Matrix, x0: Vector, P0: Matrix):
            self.F, self.B, self.H = np.array(F), np.array(B), np.array(H)
            self.Q, self.R = np.array(Q), np.array(R)
            self.x, self.P = np.array(x0), np.array(P0)
        def predict(self, u: Vector = None) -> Vector:
            """Prediction (time update) step."""
            u = np.zeros(self.B.shape[1]) if u is None else np.array(u)
            self.x = self.F @ self.x + self.B @ u
            self.P = self.F @ self.P @ self.F.T + self.Q
            return self.x.tolist()
        def update(self, z: Vector) -> Vector:
            """Update (measurement) step."""
            z = np.array(z)
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
            return self.x.tolist()

class MachineLearning:
    """A collection of fundamental machine learning algorithms."""
    class LinearSVM:
        def __init__(self, learn_rate: float = 0.001, lambda_param: float = 0.01, epochs: int = 1000):
            self.lr, self.lambda_param, self.epochs = learn_rate, lambda_param, epochs
            self.w, self.b = None, None
        def fit(self, X: Matrix, y: Vector):
            y_np = np.where(np.array(y) <= 0, -1, 1)
            X_np = np.array(X)
            n_samples, n_features = X_np.shape
            self.w = np.zeros(n_features)
            self.b = 0
            for _ in range(self.epochs):
                for idx, x_i in enumerate(X_np):
                    condition = y_np[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                    if condition:
                        self.w -= self.lr * (2 * self.lambda_param * self.w)
                    else:
                        self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_np[idx]))
                        self.b -= self.lr * y_np[idx]
        def predict(self, X: Matrix) -> Vector:
            approx = np.dot(np.array(X), self.w) - self.b
            return np.sign(approx).tolist()

class AbstractAlgebra:
    """Classes for exploring abstract algebraic structures."""
    class FiniteGroup(Generic[Element]):
        def __init__(self, elements: Set[Element], operation: Callable[[Element, Element], Element]):
            self.elements = elements
            self.op = operation
        def verify_axioms(self) -> Dict[str, bool]:
            results = {'closure': True, 'identity': False, 'inverse': False, 'associativity': True}
            # Check Closure
            for a in self.elements:
                for b in self.elements:
                    if self.op(a, b) not in self.elements:
                        results['closure'] = False; break
                if not results['closure']: break
            if not results['closure']: return results
            
            # Find Identity
            identity = None
            for e in self.elements:
                if all(self.op(e, a) == a and self.op(a, e) == a for a in self.elements):
                    identity = e; results['identity'] = True; break
            if not results['identity']: return results
            
            # Check for Inverses
            has_inverses = True
            for a in self.elements:
                if not any(self.op(a, b) == identity and self.op(b, a) == identity for b in self.elements):
                    has_inverses = False; break
            results['inverse'] = has_inverses
            
            # Check Associativity
            for a in self.elements:
                for b in self.elements:
                    for c in self.elements:
                        if self.op(self.op(a, b), c) != self.op(a, self.op(b, c)):
                            results['associativity'] = False; break
                    if not results['associativity']: break
                if not results['associativity']: break
            return results

# --- Primary Facade Class: Composition-based Design ---

class AxiomPy:
    def __init__(self):
        self._numan = NumericalAnalysis()
        self._state_est = StateEstimation()
        self._ml = MachineLearning()
        self._algebra = AbstractAlgebra()
    @property
    def numerical_analysis(self): return self._numan
    @property
    def state_estimation(self): return self._state_est
    @property
    def ml(self): return self._ml
    @property
    def abstract_algebra(self): return self._algebra

Axiom = AxiomPy()

# --- Example Usage Block ---
def demo_svm():
    print("\n--- 1. AI: Training a Linear Support Vector Machine ---")
    X = [[-2,4], [4,1], [-1,6], [2,4], [6,2]] # Linearly separable data
    y = [-1, -1, -1, 1, 1]
    svm = Axiom.ml.LinearSVM()
    svm.fit(X, y)
    print("Trained a Linear SVM on separable data.")
    print(f"  Resulting Weights (w): {[f'{w:.4f}' for w in svm.w]}")
    print(f"  Resulting Bias (b): {svm.b:.4f}")
    test_point = [0, 5]
    prediction = "Class +1" if svm.predict([test_point])[0] > 0 else "Class -1"
    print(f"  Prediction for {test_point}: {prediction}")

def demo_kalman_filter():
    print("\n--- 2. State Estimation: Kalman Filter tracking a moving object ---")
    dt = 1.0
    kf = Axiom.state_estimation.KalmanFilter(
        F=[[1, dt], [0, 1]], B=[[0.5*dt**2], [dt]], H=[[1, 0]],
        Q=[[0.1, 0], [0, 0.1]], R=[[5]],
        x0=[0, 0], P0=[[1, 0], [0, 1]]
    )
    measurements = [2.1, 3.9, 6.2, 8.1, 10.3, 11.8, 14.1, 15.9]
    print(f"Noisy Measurements: {measurements}")
    estimates = []
    for z in measurements:
        kf.predict(u=[0]) # No control input
        estimate = kf.update([z])
        estimates.append(estimate[0])
    print(f"Kalman Estimates:   {[f'{p:.2f}' for p in estimates]}")

def demo_wave_equation():
    print("\n--- 3. Numerical Physics: Simulating a 2D vibrating membrane ---")
    ic = lambda X, Y: np.exp(-100 * ((X - 0.5)**2 + (Y - 0.5)**2))
    snapshots = Axiom.numerical_analysis.solve_wave_equation_2d(
        grid_size=(30, 30), L=1.0, T=2.0, c=1.0, initial_condition=ic
    )
    print("Simulated a 2D wave equation with a central 'pluck'.")
    print(f"  Generated {len(snapshots)} snapshots of the membrane's state.")
    center_idx = len(snapshots[0]) // 2
    print(f"  Amplitude at center at t=0:   {snapshots[0][center_idx][center_idx]:.4f}")
    print(f"  Amplitude at center at t~0.6: {snapshots[3][center_idx][center_idx]:.4f} (wave has moved outwards)")
    print(f"  Amplitude at center at t~1.2: {snapshots[6][center_idx][center_idx]:.4f} (wave reflecting back)")

def demo_group_theory():
    print("\n--- 4. Abstract Algebra: Verifying a Finite Group (Z_4) ---")
    Z4_elements = {0, 1, 2, 3}
    add_mod_4 = lambda a, b: (a + b) % 4
    Z4_group = Axiom.abstract_algebra.FiniteGroup(Z4_elements, add_mod_4)
    axioms_check = Z4_group.verify_axioms()
    print("Testing the set {0, 1, 2, 3} with addition modulo 4...")
    for axiom, result in axioms_check.items():
        print(f"  Axiom '{axiom.capitalize()}': {'Passed' if result else 'Failed'}")

if __name__ == "__main__":
    print("=" * 80)
    print("    AxiomPy Mathematics Engine v2.0.0 - Premier Scientific Demonstration")
    print("=" * 80)
    demo_svm()
    demo_kalman_filter()
    demo_wave_equation()
    demo_group_theory()
    print("\n" + "=" * 80)
    print("                          Demonstration Complete")
    print("=" * 80)
