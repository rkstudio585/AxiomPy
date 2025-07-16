# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 2.4.1 (Bugfix Release)
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A premier scientific computing engine. This version introduces
#                the Nelder-Mead method for gradient-free optimization, a Lorenz
#                Attractor simulator for chaos theory, Independent Component
#                Analysis (ICA) for blind signal separation, and a
#                Propositional Logic solver.
#
################################################################################

import numpy as np
import random
import math
from typing import (List, Tuple, Callable, Any, TypeVar, Set, Generic, Dict)
from collections import namedtuple

# --- Type Aliases & Core Infrastructure ---
Vector = List[float]; Matrix = List[List[float]]
# (Other core data types assumed present for brevity)

class AxiomError(Exception): pass
class ConvergenceError(AxiomError): pass

# --- Scientific and Mathematical Domain Classes ---

class NumericalAnalysis:
    @staticmethod
    def solve_ode(f, y0, t_span, h=0.01) -> Tuple[List[float], Matrix]:
        """Solves a system of ODEs y'(t) = f(t, y) using RK4."""
        t_vals = np.arange(t_span[0], t_span[1] + h, h)
        y_vals = np.zeros((len(t_vals), len(y0)))
        y_vals[0, :] = y0
        for i in range(len(t_vals) - 1):
            t, y = t_vals[i], y_vals[i, :]
            k1 = h*np.array(f(t,y)); k2 = h*np.array(f(t+0.5*h, y+0.5*k1))
            k3 = h*np.array(f(t+0.5*h, y+0.5*k2)); k4 = h*np.array(f(t+h, y+k3))
            y_vals[i+1, :] = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return t_vals.tolist(), y_vals.tolist()

class Optimization:
    """Advanced optimization algorithms."""
    @staticmethod
    def nelder_mead(
        f: Callable[[Vector], float], x_start: Vector,
        step: float = 0.1, max_iter: int = 1000, tol: float = 1e-6
    ) -> Tuple[Vector, float]:
        """Minimizes a function using the Nelder-Mead simplex algorithm."""
        n = len(x_start)
        simplex = [np.array(x_start)]
        for i in range(n):
            p = np.array(x_start); p[i] += step
            simplex.append(p)

        for i in range(max_iter):
            scores = [(s, f(s.tolist())) for s in simplex]
            scores.sort(key=lambda x: x[1])
            simplex = [s[0] for s in scores]
            
            if scores[-1][1] - scores[0][1] < tol: break

            centroid = np.mean(simplex[:-1], axis=0)
            
            x_reflected = centroid + (centroid - simplex[-1])
            score_reflected = f(x_reflected.tolist())
            
            if scores[0][1] <= score_reflected < scores[-2][1]:
                simplex[-1] = x_reflected; continue

            if score_reflected < scores[0][1]:
                x_expanded = centroid + 2 * (x_reflected - centroid)
                if f(x_expanded.tolist()) < score_reflected:
                    simplex[-1] = x_expanded
                else:
                    simplex[-1] = x_reflected
                continue
            
            x_contracted = centroid + 0.5 * (simplex[-1] - centroid)
            if f(x_contracted.tolist()) < scores[-1][1]:
                simplex[-1] = x_contracted; continue

            x0 = simplex[0]
            for j in range(1, n + 1): simplex[j] = x0 + 0.5 * (simplex[j] - x0)
        
        best_point = simplex[0]
        return best_point.tolist(), f(best_point.tolist())

class DynamicalSystems:
    """Simulators for complex, non-linear dynamical systems."""
    @staticmethod
    def simulate_lorenz_attractor(
        initial_state: Vector = [0.0, 1.0, 1.05], T: float = 40.0, dt: float = 0.01,
        sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0
    ) -> Dict[str, Any]:
        """Solves the Lorenz system of equations, a model for atmospheric convection."""
        def lorenz_system(t, state):
            x, y, z = state
            dx_dt = sigma * (y - x); dy_dt = x * (rho - z) - y; dz_dt = x * y - beta * z
            return [dx_dt, dy_dt, dz_dt]
        
        t, states_list = NumericalAnalysis.solve_ode(lorenz_system, initial_state, (0, T), h=dt)
        
        # --- FIX: Convert the returned list to a NumPy array for slicing ---
        states_np = np.array(states_list)
        
        return {
            'time': t,
            'x': states_np[:, 0].tolist(),
            'y': states_np[:, 1].tolist(),
            'z': states_np[:, 2].tolist()
        }

class MachineLearning:
    @staticmethod
    def independent_component_analysis(X: Matrix, n_components: int, max_iter=200, tol=1e-5) -> Tuple[Matrix, Matrix]:
        X = np.array(X).T
        X -= X.mean(axis=1, keepdims=True)
        cov = np.cov(X); d, E = np.linalg.eigh(cov)
        D = np.diag(1. / np.sqrt(d + 1e-5)); W_pre = D @ E.T
        X_w = W_pre @ X
        W = np.random.randn(n_components, n_components)
        for i in range(n_components):
            w = W[i, :].copy(); w /= np.linalg.norm(w)
            for _ in range(max_iter):
                w_prev = w.copy()
                g = np.tanh(w @ X_w); g_prime = 1 - g**2
                w = (X_w * g).mean(axis=1) - g_prime.mean() * w
                w -= (w @ W[:i, :].T) @ W[:i, :]; w /= np.linalg.norm(w)
                if np.abs(np.abs(w @ w_prev) - 1) < tol: break
            W[i, :] = w
        return W.tolist(), (W @ X_w).T.tolist()

class Logic:
    @staticmethod
    def solve_propositional_formula(formula: str, variables: List[str]) -> Dict:
        n = len(variables); truth_table = []; is_true_list = []
        safe_formula = formula.replace('and', '&').replace('or', '|').replace('not', '~')
        for i in range(2**n):
            row = {}; temp = i
            for var in reversed(variables):
                row[var] = (temp % 2 == 1); temp //= 2
            result = eval(safe_formula, {"__builtins__": None}, row)
            is_true_list.append(result); row['result'] = result; truth_table.append(row)
        is_tautology = all(is_true_list); is_contradiction = not any(is_true_list)
        return {'truth_table': truth_table, 'is_tautology': is_tautology,
                'is_contradiction': is_contradiction, 'is_satisfiable': not is_contradiction}
        
# --- Primary Facade Class ---

class AxiomPy:
    def __init__(self):
        self._optim = Optimization(); self._dynamics = DynamicalSystems()
        self._ml = MachineLearning(); self._logic = Logic()
    @property
    def optim(self): return self._optim
    @property
    def dynamics(self): return self._dynamics
    @property
    def ml(self): return self._ml
    @property
    def logic(self): return self._logic

Axiom = AxiomPy()

# --- Example Usage Block ---
def demo_nelder_mead():
    print("\n--- 1. Optimization: Nelder-Mead on Rosenbrock's Banana Function ---")
    rosenbrock = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    start_point = [-1.0, 2.0]
    min_point, min_val = Axiom.optim.nelder_mead(rosenbrock, start_point)
    print("Minimized Rosenbrock function, which is difficult for gradient methods.")
    print(f"  Found minimum near: {[f'{p:.4f}' for p in min_point]} (Exact is [1,1])")
    print(f"  Value at minimum: {min_val:.6f}")

def demo_lorenz_attractor():
    print("\n--- 2. Dynamical Systems: Simulating the Lorenz Attractor (Chaos) ---")
    sim = Axiom.dynamics.simulate_lorenz_attractor(T=5)
    print("Simulated the Lorenz system, a classic model of chaos.")
    print(f"  State at t=0:   x={sim['x'][0]:.2f}, y={sim['y'][0]:.2f}, z={sim['z'][0]:.2f}")
    t1_idx = len(sim['time']) // 2
    t2_idx = -1
    print(f"  State at t=2.5: x={sim['x'][t1_idx]:.2f}, y={sim['y'][t1_idx]:.2f}, z={sim['z'][t1_idx]:.2f}")
    print(f"  State at t=5.0: x={sim['x'][t2_idx]:.2f}, y={sim['y'][t2_idx]:.2f}, z={sim['z'][t2_idx]:.2f}")

def demo_ica():
    print("\n--- 3. Machine Learning: Independent Component Analysis (ICA) ---")
    t = np.linspace(0, 1, 500); s1 = np.sin(2*np.pi*5*t); s2 = np.sign(np.sin(2*np.pi*8*t))
    S = np.c_[s1, s2]; A = np.array([[1, 1], [0.5, 2]]); X = S @ A.T
    _, S_est = Axiom.ml.independent_component_analysis(X.tolist(), n_components=2)
    print("Separated two mixed signals (sine and square wave).")
    corr_with_s1 = np.corrcoef(np.array(S_est)[:, 0], s1)[0, 1]
    print(f"  Correlation of estimated component 1 with original sine: {abs(corr_with_s1):.4f}")
    
def demo_logic():
    print("\n--- 4. Formal Logic: Propositional Logic Solver ---")
    formula = "(P or Q) and ((not P) or (not Q))"
    variables = ['P', 'Q']
    result = Axiom.logic.solve_propositional_formula(formula, variables)
    print(f"Analyzing formula: {formula}")
    print("  Truth Table:")
    for row in result['truth_table']: print(f"    P={row['P']}, Q={row['Q']}  =>  Result={row['result']}")
    print(f"  Is Tautology? {result['is_tautology']}")
    print(f"  Is Contradiction? {result['is_contradiction']}")
    print(f"  Is Satisfiable? {result['is_satisfiable']}")

if __name__ == "__main__":
    print("=" * 80)
    print("    AxiomPy Mathematics Engine v2.4.1 - Corrected Demonstration")
    print("=" * 80)
    demo_nelder_mead()
    demo_lorenz_attractor()
    demo_ica()
    demo_logic()
    print("\n" * 2)
    print("                          Demonstration Complete")
    print("=" * 80)
