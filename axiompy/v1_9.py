# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 1.9.1 (Bugfix Release)
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A premier scientific computing module. This version introduces
#                a Game Theory engine for solving Nash Equilibria, a Markov
#                Chain simulator for stochastic processes, the Haar Wavelet
#                Transform for advanced signal processing, and a Boundary Value
#                Problem solver for advanced physics and engineering models.
#
################################################################################

import numpy as np
import random
import math
from typing import (List, Tuple, Callable, Dict, Any, TypeVar, Set, Generic)

# --- Type Aliases & Core Data Types ---
Vector = List[float]; Matrix = List[List[float]]
# (Other core data types assumed present for brevity)

class LinearAlgebra:
    @staticmethod
    def transpose(matrix: Matrix) -> Matrix: return np.transpose(matrix).tolist()
    @staticmethod
    def eigenvalues_eigenvectors(matrix: Matrix) -> Tuple[Vector, Matrix]:
        w, v = np.linalg.eig(matrix)
        idx = w.argsort()
        eigenvalues = w[idx]; eigenvectors = v[:, idx]
        return eigenvalues.tolist(), eigenvectors.T.tolist()
    #... Other linalg functions assumed present

class NumericalAnalysis:
    @staticmethod
    def solve_ode(f: Callable, y0: Vector, t_span: Tuple[float, float], h: float=0.01) -> Tuple[Vector, Matrix]:
        """Solves a system of ODEs y'(t) = f(t, y) using RK4."""
        t_vals = np.arange(t_span[0], t_span[1] + h, h)
        y_vals = np.zeros((len(t_vals), len(y0)))
        y_vals[0, :] = y0
        for i in range(len(t_vals) - 1):
            t, y = t_vals[i], y_vals[i, :]
            k1 = h * np.array(f(t, y))
            k2 = h * np.array(f(t + 0.5*h, y + 0.5*k1))
            k3 = h * np.array(f(t + 0.5*h, y + 0.5*k2))
            k4 = h * np.array(f(t + h, y + k3))
            y_vals[i+1, :] = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return t_vals.tolist(), y_vals.tolist()

    @staticmethod
    def solve_bvp_shooting(
        f: Callable[[float, float, float], float], # f(x, y, y')
        x_span: Tuple[float, float], y_bounds: Tuple[float, float],
        s_guesses: Tuple[float, float] = (1.0, -1.0), tol: float = 1e-6, max_iter: int = 20
    ) -> Tuple[Vector, Vector]:
        """
        Solves a 2nd-order Boundary Value Problem y''=f with y(a),y(b) known
        using the Shooting Method.
        """
        a, b = x_span; ya, yb = y_bounds
        s1, s2 = s_guesses
        
        def ode_system(t, Y): return [Y[1], f(t, Y[0], Y[1])]

        _, sol1_list = NumericalAnalysis.solve_ode(ode_system, [ya, s1], x_span)
        
        # --- FIX: Convert list output to NumPy array for correct indexing ---
        sol1 = np.array(sol1_list)
        phi1 = sol1[-1, 0] # Now this works

        for _ in range(max_iter):
            _, sol2_list = NumericalAnalysis.solve_ode(ode_system, [ya, s2], x_span)
            
            # --- FIX: Convert list output to NumPy array ---
            sol2 = np.array(sol2_list)
            phi2 = sol2[-1, 0] # Now this works
            
            if abs(phi2 - yb) < tol: break
            
            # Secant method to find better guess for y'(a)
            if abs(phi2 - phi1) < 1e-12: # Avoid division by zero
                 print("Warning: Shooting method failed, guesses are not distinct.")
                 break
            s_next = s2 - (phi2 - yb) * (s2 - s1) / (phi2 - phi1)
            s1, phi1 = s2, phi2
            s2 = s_next
        else:
            print("Warning: Shooting method did not converge within max iterations.")

        x_vals, y_vals_list = NumericalAnalysis.solve_ode(ode_system, [ya, s2], x_span)
        
        # --- FIX: Convert final solution matrix for correct data extraction ---
        y_vals_matrix = np.array(y_vals_list)
        return x_vals, y_vals_matrix[:, 0].tolist()

class GameTheory:
    @staticmethod
    def solve_2x2_zero_sum(payoff_matrix: Matrix) -> Dict:
        a, b = payoff_matrix[0]; c, d = payoff_matrix[1]
        saddle_check = max(min(a,b), min(c,d)) == min(max(a,c), max(b,d))
        if saddle_check:
             return {'p1_strategy': [1.0, 0.0], 'p2_strategy': [1.0, 0.0], 'value': max(min(a,b), min(c,d)), 'type': 'Pure'}
        denom = (a - c) - (b - d)
        if abs(denom) < 1e-9: raise ValueError("Game has no unique mixed strategy solution.")
        p1 = (d - c) / denom; q1 = (d - b) / denom; value = (a*d - b*c) / denom
        return {'p1_strategy': [p1, 1-p1], 'p2_strategy': [q1, 1-q1], 'value': value, 'type': 'Mixed'}

class StochasticProcesses:
    class MarkovChain:
        def __init__(self, states: List[Any], transition_matrix: Matrix):
            self.states = states; self.state_map = {s: i for i, s in enumerate(states)}
            self.transition_matrix = np.array(transition_matrix)
            self.current_state = random.choice(states)
        def step(self) -> Any:
            idx = self.state_map[self.current_state]
            self.current_state = random.choices(self.states, weights=self.transition_matrix[idx, :], k=1)[0]
            return self.current_state
        def run(self, n_steps: int) -> List[Any]: return [self.step() for _ in range(n_steps)]
        def steady_state(self) -> Dict[Any, float]:
            eigenvalues, eigenvectors = LinearAlgebra.eigenvalues_eigenvectors(self.transition_matrix.T.tolist())
            idx = np.argmin(np.abs(np.array(eigenvalues) - 1.0))
            ss_vector = np.real(eigenvectors[idx]); ss_vector /= ss_vector.sum()
            return {s: p for s, p in zip(self.states, ss_vector)}

class SignalProcessing:
    @staticmethod
    def _pad_to_power_of_two(signal: Vector) -> Vector:
        next_pow2 = 1 << (len(signal) - 1).bit_length(); return signal + [0] * (next_pow2 - len(signal))
    @staticmethod
    def dwt_haar(signal: Vector) -> Vector:
        data = SignalProcessing._pad_to_power_of_two(signal); output = [0.0]*len(data); L = len(data)
        while L >= 2:
            half = L // 2
            for i in range(half): output[i] = (data[2*i] + data[2*i+1]) / math.sqrt(2)
            for i in range(half): output[i+half] = (data[2*i] - data[2*i+1]) / math.sqrt(2)
            data[:L] = output[:L]; L = half
        return output
    @staticmethod
    def idwt_haar(coeffs: Vector) -> Vector:
        L = 2; data = list(coeffs)
        while L <= len(data):
            half = L // 2; approx, detail = data[:half], data[half:L]
            for i in range(half):
                data[2*i] = (approx[i] + detail[i]) / math.sqrt(2)
                data[2*i+1] = (approx[i] - detail[i]) / math.sqrt(2)
            L *= 2
        return data

# --- Primary Facade Class ---

class AxiomPy:
    def __init__(self):
        self._numan = NumericalAnalysis(); self._game = GameTheory()
        self._stochastic = StochasticProcesses(); self._signal = SignalProcessing()
    @property
    def numerical_analysis(self): return self._numan
    @property
    def game_theory(self): return self._game
    @property
    def stochastic(self): return self._stochastic
    @property
    def signal(self): return self._signal

Axiom = AxiomPy()

# --- Example Usage Block ---
def demo_game_theory():
    print("\n--- 1. Game Theory: Solving 'Matching Pennies' Nash Equilibrium ---")
    payoffs = [[1, -1], [-1, 1]]
    equilibrium = Axiom.game_theory.solve_2x2_zero_sum(payoffs)
    print("Payoff Matrix for Player 1:", payoffs)
    print(f"  Equilibrium Type: {equilibrium['type']}")
    print(f"  Player 1 Strategy (Heads/Tails): {[f'{p:.2f}' for p in equilibrium['p1_strategy']]}")
    print(f"  Player 2 Strategy (Heads/Tails): {[f'{p:.2f}' for p in equilibrium['p2_strategy']]}")
    print(f"  Value of the game: {equilibrium['value']:.2f}")

def demo_stochastic_processes():
    print("\n--- 2. Stochastic Processes: Simulating a Weather Markov Chain ---")
    states = ['Sunny', 'Cloudy', 'Rainy']
    transitions = [[0.8, 0.2, 0.0], [0.4, 0.4, 0.2], [0.2, 0.6, 0.2]]
    mc = Axiom.stochastic.MarkovChain(states, transitions)
    print(f"Simulating 10 days of weather, starting from '{mc.current_state}':")
    print(f"  Path: {mc.run(10)}")
    steady_state = mc.steady_state()
    print("Long-term steady-state probabilities:")
    for state, prob in steady_state.items(): print(f"  P({state}) = {prob:.4f}")

def demo_wavelet_transform():
    print("\n--- 3. Advanced Signal Processing: Haar Wavelet Transform ---")
    signal = [4, 6, 10, 12, 8, 6, 6, 4]
    coeffs = Axiom.signal.dwt_haar(signal)
    reconstructed = Axiom.signal.idwt_haar(coeffs)
    print(f"Original signal:      {signal}")
    print(f"Wavelet Coefficients: {[f'{c:.2f}' for c in coeffs]}")
    print(f"Reconstructed signal: {[f'{x:.1f}' for x in reconstructed]}")

def demo_bvp_solver():
    print("\n--- 4. Numerical Methods: Solving a Boundary Value Problem ---")
    f = lambda x, y, yp: -y
    x_vals, y_vals = Axiom.numerical_analysis.solve_bvp_shooting(
        f, x_span=(0, math.pi/2), y_bounds=(0, 1)
    )
    print("Solved y''=-y with y(0)=0, y(pi/2)=1 using Shooting Method.")
    idx_mid = len(x_vals) // 2
    x_mid, y_mid = x_vals[idx_mid], y_vals[idx_mid]
    exact_mid = math.sin(x_mid)
    print(f"  Solution at x={x_mid:.4f}: y={y_mid:.6f}")
    print(f"  Exact solution (sin(x)): {exact_mid:.6f}")

if __name__ == "__main__":
    print("=" * 80)
    print("    AxiomPy Mathematics Engine v1.9.1 - Corrected Demonstration")
    print("=" * 80)
    demo_game_theory()
    demo_stochastic_processes()
    demo_wavelet_transform()
    demo_bvp_solver()
    print("\n" + "=" * 80)
    print("                          Demonstration Complete")
    print("=" * 80)
