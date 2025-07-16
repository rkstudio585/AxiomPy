# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 1.8.1 (Bugfix Release)
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A premier scientific computing module for Python. This version
#                introduces a Genetic Algorithm for complex optimization,
#                Quaternions for 3D rotations, a Mandelbrot Set generator for
#                exploring complex dynamics, and a Deterministic Finite Automaton
#                for theoretical computer science applications.
#
################################################################################

import numpy as np
import random
import math
from typing import (List, Tuple, Callable, Dict, Any, TypeVar, Set, Generic)

# --- Type Aliases & Core Data Types ---
Vector = List[float]; Matrix = List[List[float]]
State = TypeVar('State'); Symbol = TypeVar('Symbol')

class Quaternion:
    """Represents a Quaternion for 3D rotations and algebra."""
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w, self.x, self.y, self.z = w, x, y, z
    def __repr__(self) -> str:
        return f"({self.w:.4f} + {self.x:.4f}i + {self.y:.4f}j + {self.z:.4f}k)"
    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(self.w+other.w, self.x+other.x, self.y+other.y, self.z+other.z)
    def __mul__(self, other: Any) -> 'Quaternion':
        if isinstance(other, Quaternion):
            w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
            x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
            y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
            z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float)):
            return Quaternion(self.w*other, self.x*other, self.y*other, self.z*other)
        return NotImplemented
    def conjugate(self) -> 'Quaternion': return Quaternion(self.w, -self.x, -self.y, -self.z)
    def norm(self) -> float: return math.sqrt(self.w**2+self.x**2+self.y**2+self.z**2)
    def normalize(self) -> 'Quaternion':
        n = self.norm()
        if n == 0: return Quaternion(0, 0, 0, 0)
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
    @staticmethod
    def from_axis_angle(axis: Vector, angle_rad: float) -> 'Quaternion':
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0: return Quaternion(1, 0, 0, 0) # Return identity for zero axis
        axis = np.array(axis) / axis_norm
        w = math.cos(angle_rad / 2.0)
        x, y, z = axis * math.sin(angle_rad / 2.0)
        return Quaternion(w, x, y, z)
    def rotate_vector(self, v: Vector) -> Vector:
        p = Quaternion(0, v[0], v[1], v[2])
        q_prime = self * p * self.conjugate()
        return [q_prime.x, q_prime.y, q_prime.z]

# (Other data types like ComplexNumber, RationalNumber, etc. assumed present)

# --- Scientific and Mathematical Domain Classes ---

class Optimization:
    """Advanced optimization algorithms."""
    class GeneticAlgorithm:
        def __init__(self, fitness_fn: Callable[[List[Any]], float], gene_pool: List[Any],
                     individual_len: int, pop_size: int, mutation_rate: float,
                     crossover_rate: float, elitism: bool = True):
            # --- FIX: individual_len is now an explicit, required parameter ---
            self.fitness_fn = fitness_fn
            self.gene_pool = gene_pool
            self.individual_len = individual_len
            self.pop_size = pop_size
            self.mutation_rate = mutation_rate
            self.crossover_rate = crossover_rate
            self.elitism = elitism
            self.population = self._init_population()

        def _init_population(self):
            return [[random.choice(self.gene_pool) for _ in range(self.individual_len)] for _ in range(self.pop_size)]
        def _crossover(self, parent1, parent2):
            if random.random() < self.crossover_rate:
                if self.individual_len <= 1: return parent1, parent2 # Cannot crossover len 1
                point = random.randint(1, self.individual_len - 1)
                return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
            return parent1, parent2
        def _mutate(self, individual):
            return [gene if random.random() > self.mutation_rate else random.choice(self.gene_pool) for gene in individual]
        def evolve(self, generations: int) -> Tuple[List[Any], float]:
            for _ in range(generations):
                fitness_scores = [self.fitness_fn(ind) for ind in self.population]
                
                # Selection (Tournament)
                new_population = []
                if self.elitism:
                    elite_idx = np.argmax(fitness_scores)
                    new_population.append(self.population[elite_idx])

                while len(new_population) < self.pop_size:
                    # Select parent 1
                    p1_idx, p2_idx = random.sample(range(self.pop_size), 2)
                    parent1 = self.population[p1_idx] if fitness_scores[p1_idx] > fitness_scores[p2_idx] else self.population[p2_idx]
                    
                    # Select parent 2
                    p3_idx, p4_idx = random.sample(range(self.pop_size), 2)
                    parent2 = self.population[p3_idx] if fitness_scores[p3_idx] > fitness_scores[p4_idx] else self.population[p4_idx]
                    
                    child1, child2 = self._crossover(parent1, parent2)
                    new_population.append(self._mutate(child1))
                    if len(new_population) < self.pop_size:
                        new_population.append(self._mutate(child2))
                
                self.population = new_population

            final_scores = [self.fitness_fn(ind) for ind in self.population]
            best_idx = np.argmax(final_scores)
            return self.population[best_idx], final_scores[best_idx]

class Fractals:
    @staticmethod
    def generate_mandelbrot(width: int, height: int, x_min: float, x_max: float,
                            y_min: float, y_max: float, max_iter: int) -> Matrix:
        result = np.zeros((height, width))
        for row in range(height):
            for col in range(width):
                c = complex(x_min + (col/width)*(x_max-x_min),
                            y_min + (row/height)*(y_max-y_min))
                z = 0
                n = 0
                while abs(z) <= 2 and n < max_iter:
                    z = z*z + c
                    n += 1
                result[row, col] = n
        return result.tolist()

class Automata(Generic[State, Symbol]):
    class DFA:
        def __init__(self, states: Set[State], alphabet: Set[Symbol],
                     transitions: Dict[Tuple[State, Symbol], State],
                     start_state: State, accept_states: Set[State]):
            self.states, self.alphabet, self.transitions = states, alphabet, transitions
            self.start_state, self.accept_states = start_state, accept_states
        def accepts(self, input_string: List[Symbol]) -> bool:
            current_state = self.start_state
            for symbol in input_string:
                if symbol not in self.alphabet: return False
                current_state = self.transitions.get((current_state, symbol))
                if current_state is None: return False
            return current_state in self.accept_states

# --- Primary Facade Class: Composition-based Design ---

class AxiomPy:
    def __init__(self):
        self._optim = Optimization()
        self._fractals = Fractals()
        self._automata = Automata()
        self.Quaternion = Quaternion
    @property
    def optim(self) -> Optimization: return self._optim
    @property
    def fractals(self) -> Fractals: return self._fractals
    @property
    def automata(self) -> Automata: return self._automata

Axiom = AxiomPy()

# --- Example Usage Block ---
if __name__ == "__main__":
    print("=" * 80)
    print("    AxiomPy Mathematics Engine v1.8.1 - Corrected Demonstration")
    print("=" * 80)

    # 1. Metaheuristic Optimization: Genetic Algorithm
    print("\n--- 1. Optimization: Genetic Algorithm solving a 'guess the password' problem ---")
    TARGET = "AXIOM"
    GENE_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    def password_fitness(individual: List[str]) -> float:
        return sum(1 for expected, actual in zip(TARGET, individual) if expected == actual)
    
    # --- FIX: `individual_len` is now passed explicitly ---
    ga = Axiom.optim.GeneticAlgorithm(
        fitness_fn=password_fitness, gene_pool=list(GENE_POOL),
        individual_len=len(TARGET), pop_size=100,
        mutation_rate=0.1, crossover_rate=0.8
    )
    best_solution, best_fitness = ga.evolve(generations=50)
    print(f"Target password: {TARGET}")
    print(f"Best solution found by GA: {''.join(best_solution)}")
    print(f"Fitness score: {best_fitness}/{len(TARGET)}")
    
    # 2. 3D Rotational Algebra: Quaternions
    print("\n--- 2. Rotational Algebra: Using Quaternions to rotate a vector ---")
    v = [1, 0, 0]
    axis = [0, 0, 1]
    angle = math.pi / 2
    q_rot = Axiom.Quaternion.from_axis_angle(axis, angle)
    v_rotated = q_rot.rotate_vector(v)
    print(f"Rotating vector {v} by 90 degrees around the z-axis {axis}")
    print(f"Rotation Quaternion: {q_rot}")
    print(f"Resulting vector: {[f'{x:.4f}' for x in v_rotated]} (should be approx [0, 1, 0])")
    
    # 3. Complex Dynamics: Mandelbrot Set
    print("\n--- 3. Fractal Geometry: Generating a small ASCII Mandelbrot set ---")
    mandelbrot_matrix = Axiom.fractals.generate_mandelbrot(
        width=40, height=20, x_min=-2.0, x_max=1.0, y_min=-1.0, y_max=1.0, max_iter=30
    )
    chars = ".,-~:;=!*#$@"
    for row in mandelbrot_matrix:
        line = ""
        for val in row:
            if val == 30: line += " "
            else: line += chars[int(val / 30 * len(chars))]
        print(f"  {line}")

    # 4. Automata Theory: Deterministic Finite Automaton (DFA)
    print("\n--- 4. Theoretical CS: DFA for recognizing strings ending in '01' ---")
    dfa = Axiom.automata.DFA(
        states={'q0', 'q1', 'q2'},
        alphabet={'0', '1'},
        transitions={('q0', '0'): 'q1', ('q0', '1'): 'q0',
                     ('q1', '0'): 'q1', ('q1', '1'): 'q2',
                     ('q2', '0'): 'q1', ('q2', '1'): 'q0'},
        start_state='q0',
        accept_states={'q2'}
    )
    test_strings = ["10101", "0001", "10110", "1", "01"]
    for s in test_strings:
        result = "Accepted" if dfa.accepts(list(s)) else "Rejected"
        print(f"String '{s}': {result}")

    print("\n" + "=" * 80)
    print("                          Demonstration Complete")
    print("=" * 80)
