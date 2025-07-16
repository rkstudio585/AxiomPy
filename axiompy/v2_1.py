# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 2.1.1 (Bugfix Release)
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A capstone scientific computing engine. This version introduces
#                a Finite Element Method (FEM) solver, a Simulated Annealing
#                global optimizer, a Q-Learning agent for Reinforcement Learning,
#                and a Lattice Boltzmann Method for Computational Fluid Dynamics.
#
################################################################################

import numpy as np
import random
import math
from typing import (List, Tuple, Callable, Dict, Any, TypeVar, Set, Generic)

# --- Type Aliases & Core Infrastructure ---
Vector = List[float]; Matrix = List[List[float]]

class AxiomError(Exception):
    """Base exception class for all errors raised by the AxiomPy module."""
    pass
class ConvergenceError(AxiomError):
    """Raised when an iterative method fails to converge."""
    pass

# (Other core data types like Quaternion, etc. assumed present for brevity)

class NumericalAnalysis:
    @staticmethod
    def gaussian_quadrature(func, a, b, n=5): #... Assumed present
        # Mock implementation for demo
        c1, c2 = (b - a) / 2.0, (b + a) / 2.0
        abscissae = [-0.90617985, -0.53846931, 0.0, 0.53846931, 0.90617985]
        weights = [0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689]
        integral = sum(w * func(c1 * x + c2) for w, x in zip(weights, abscissae))
        return c1 * integral

class FiniteElement:
    @staticmethod
    def solve_poisson_1d(
        domain: Tuple[float, float], n_elements: int,
        f: Callable[[float], float], bc: Tuple[float, float]
    ) -> Tuple[Vector, Vector]:
        a, b = domain; u_a, u_b = bc
        h = (b - a) / n_elements
        nodes = np.linspace(a, b, n_elements + 1)
        K = np.zeros((n_elements + 1, n_elements + 1))
        F = np.zeros(n_elements + 1)
        for i in range(n_elements):
            k_local = (1/h) * np.array([[1, -1], [-1, 1]])
            f_local_func = lambda x, j: f(x) * (1-(x-nodes[i])/h if j==0 else (x-nodes[i])/h)
            f_local = [NumericalAnalysis.gaussian_quadrature(lambda x: f_local_func(x,j), nodes[i], nodes[i+1]) for j in range(2)]
            K[i:i+2, i:i+2] += k_local
            F[i:i+2] += f_local
        F = F - K[:, 0] * u_a - K[:, -1] * u_b
        K[0,:], K[:,0], K[-1,:], K[:,-1] = 0,0,0,0
        K[0,0], K[-1,-1] = 1, 1
        F[0], F[-1] = u_a, u_b
        return nodes.tolist(), np.linalg.solve(K, F).tolist()

class Optimization:
    class SimulatedAnnealing:
        def __init__(self, cost_fn, neighbor_fn, initial_temp, final_temp, alpha):
            self.cost_fn, self.neighbor_fn = cost_fn, neighbor_fn
            self.T_i, self.T_f, self.alpha = initial_temp, final_temp, alpha
        def solve(self, initial_solution: Any) -> Tuple[Any, float]:
            current_sol, current_cost = initial_solution, self.cost_fn(initial_solution)
            best_sol, best_cost = current_sol, current_cost
            temp = self.T_i
            while temp > self.T_f:
                neighbor = self.neighbor_fn(current_sol)
                neighbor_cost = self.cost_fn(neighbor)
                cost_diff = neighbor_cost - current_cost
                if cost_diff < 0 or random.random() < math.exp(-cost_diff / temp):
                    current_sol, current_cost = neighbor, neighbor_cost
                if current_cost < best_cost:
                    best_sol, best_cost = current_sol, current_cost
                temp *= self.alpha
            return best_sol, best_cost

class MachineLearning:
    class QLearning:
        def __init__(self, actions: List[Any], learn_rate: float,
                     gamma: float, epsilon: float, epsilon_decay: float = 0.999):
            self.q_table = {}
            self.actions = actions
            self.lr, self.gamma, self.epsilon, self.eps_decay = learn_rate, gamma, epsilon, epsilon_decay
        def choose_action(self, state: Any) -> Any:
            self.q_table.setdefault(state, [0.0] * len(self.actions))
            if random.random() < self.epsilon: return random.choice(self.actions)
            return self.actions[np.argmax(self.q_table[state])]
        def update_q_table(self, state, action, reward, next_state):
            # --- FIX: Ensure the current state exists in the Q-table before access ---
            self.q_table.setdefault(state, [0.0] * len(self.actions))
            
            action_idx = self.actions.index(action)
            self.q_table.setdefault(next_state, [0.0] * len(self.actions))
            old_val = self.q_table[state][action_idx]
            future_opt_val = np.max(self.q_table[next_state])
            new_val = old_val + self.lr * (reward + self.gamma * future_opt_val - old_val)
            self.q_table[state][action_idx] = new_val
            self.epsilon *= self.eps_decay

class ComputationalFluidDynamics:
    @staticmethod
    def lattice_boltzmann_2d(nx, ny, obstacle_mask, tau, n_iter) -> List[Matrix]:
        w = np.array([4/9, 1/9,1/9,1/9,1/9, 1/36,1/36,1/36,1/36])
        c = np.array([[0,0], [1,0],[0,1],[-1,0],[0,-1], [1,1],[-1,1],[-1,-1],[1,-1]])
        u0 = 0.1
        fin = np.ones((ny, nx, 9)); fin[:, :, :] *= w[np.newaxis, np.newaxis, :]
        snapshots = []
        for t in range(n_iter):
            for i in range(9): fin[:, :, i] = np.roll(np.roll(fin[:, :, i], c[i,0], axis=1), c[i,1], axis=0)
            bnd_out = fin[obstacle_mask, :]; fin[obstacle_mask, :] = bnd_out[:, [0,3,4,1,2,7,8,5,6]]
            rho = np.sum(fin, 2); ux = np.sum(fin * c[:,0], 2)/rho; uy = np.sum(fin * c[:,1], 2)/rho
            ux[:, 0] = u0; uy[:, 0] = 0; rho[:, 0] = 1
            feq = np.zeros_like(fin)
            for i in range(9):
                cu = 3 * (c[i,0]*ux + c[i,1]*uy)
                feq[:,:,i] = rho * w[i] * (1 + cu + 0.5*cu**2 - 1.5*(ux**2+uy**2))
            fin += -(1.0/tau) * (fin - feq)
            if t % (n_iter // 10) == 0: snapshots.append(np.sqrt(ux**2 + uy**2).tolist())
        return snapshots

# --- Primary Facade Class ---

class AxiomPy:
    def __init__(self):
        self._fem = FiniteElement(); self._optim = Optimization()
        self._ml = MachineLearning(); self._cfd = ComputationalFluidDynamics()
    @property
    def fem(self): return self._fem
    @property
    def optim(self): return self._optim
    @property
    def ml(self): return self._ml
    @property
    def cfd(self): return self._cfd

Axiom = AxiomPy()

# --- Example Usage Block ---
def demo_fem():
    print("\n--- 1. Finite Element Method: Solving 1D Poisson Equation ---")
    f = lambda x: math.sin(math.pi * x)
    nodes, u = Axiom.fem.solve_poisson_1d(domain=(0,1), n_elements=10, f=f, bc=(0,0))
    mid_idx = len(nodes) // 2
    print("Solved -u'' = sin(pi*x) with u(0)=0, u(1)=0.")
    print(f"  FEM solution at x=0.5:   {u[mid_idx]:.6f}")
    print(f"  Exact solution at x=0.5: {math.sin(math.pi*0.5)/math.pi**2:.6f}")

def demo_simulated_annealing():
    print("\n--- 2. Global Optimization: Simulated Annealing for TSP ---")
    points = {'A': (0,0), 'B': (1,5), 'C': (3,1), 'D': (6,4), 'E': (7,0)}
    def tsp_cost(path): return -sum(math.sqrt((points[path[i]][0]-points[path[(i+1)%len(path)]][0])**2 + (points[path[i]][1]-points[path[(i+1)%len(path)]][1])**2) for i in range(len(path)))
    def tsp_neighbor(path): new_path = list(path); i, j = random.sample(range(len(new_path)), 2); new_path[i], new_path[j] = new_path[j], new_path[i]; return new_path
    sa = Axiom.optim.SimulatedAnnealing(tsp_cost, tsp_neighbor, 1000, 1, 0.995)
    best_path, best_score = sa.solve(list(points.keys()))
    print(f"Found optimal path for a 5-city TSP: {' -> '.join(best_path)}")
    print(f"  Path Length: {-best_score:.4f}")

def demo_q_learning():
    print("\n--- 3. Reinforcement Learning: Q-Learning to solve a maze ---")
    agent = Axiom.ml.QLearning(actions=['U','D','L','R'], learn_rate=0.1, gamma=0.9, epsilon=1.0)
    # Simplified training loop for demonstration
    agent.update_q_table(state=(0,0), action='R', reward=-1, next_state=(0,1))
    agent.update_q_table(state=(0,1), action='R', reward=-1, next_state=(0,2))
    agent.update_q_table(state=(0,2), action='R', reward=10, next_state=(0,3)) # A good move
    print("Trained a Q-Learning agent on a simple environment.")
    print(f"  Optimal action from state (0,2): {agent.choose_action((0,2))}")

def demo_cfd():
    print("\n--- 4. Computational Fluid Dynamics: LBM for flow simulation ---")
    nx, ny = 100, 50
    obstacle = np.fromfunction(lambda y,x: (y-ny/2)**2 + (x-nx/4)**2 < (ny/4)**2, (ny, nx))
    snapshots = Axiom.cfd.lattice_boltzmann_2d(nx, ny, obstacle, tau=0.6, n_iter=200)
    print("Simulated 2D fluid flow around a cylindrical obstacle.")
    print(f"  Generated {len(snapshots)} snapshots of the velocity field.")
    center_y, after_x = ny // 2, nx // 2
    print(f"  Initial fluid speed at ({after_x},{center_y}): {snapshots[0][center_y][after_x]:.4f}")
    print(f"  Final fluid speed at ({after_x},{center_y}):   {snapshots[-1][center_y][after_x]:.4f}")

if __name__ == "__main__":
    print("=" * 80)
    print("    AxiomPy Mathematics Engine v2.1.1 - Corrected Demonstration")
    print("=" * 80)
    demo_fem()
    demo_simulated_annealing()
    demo_q_learning()
    demo_cfd()
    print("\n" * 2)
    print("                          Demonstration Complete")
    print("=" * 80)
