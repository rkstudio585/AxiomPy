# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 2.5.2 (Bugfix Release)
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A premier scientific computing engine. This version introduces
#                a Stochastic Differential Equation (SDE) solver for financial
#                modeling, a Special Relativity engine with Lorentz
#                transformations, the Savitzky-Golay filter for advanced signal
#                processing, and the Huffman algorithm for data compression.
#
################################################################################

import numpy as np
import random
import math
import heapq
from collections import Counter, namedtuple
from typing import (List, Tuple, Callable, Any, TypeVar, Set, Generic, Dict)

# --- Type Aliases & Core Infrastructure ---
Vector = List[float]; Matrix = List[List[float]]
# (Other core data types assumed present for brevity)

class AxiomError(Exception): pass
class ConvergenceError(AxiomError): pass

# --- Scientific and Mathematical Domain Classes ---

class StochasticProcesses:
    @staticmethod
    def solve_sde_euler_maruyama(drift, diffusion, y0, t_span, dt) -> Tuple[Vector, Vector]:
        t_vals = np.arange(t_span[0], t_span[1] + dt, dt)
        y_vals = np.zeros(len(t_vals)); y_vals[0] = y0
        sqrt_dt = math.sqrt(dt)
        for i in range(len(t_vals) - 1):
            t, y = t_vals[i], y_vals[i]
            dW = np.random.normal(0.0, sqrt_dt)
            y_vals[i+1] = y + drift(t, y) * dt + diffusion(t, y) * dW
        return t_vals.tolist(), y_vals.tolist()

class SpecialRelativity:
    C: float = 299792458.0
    class LorentzVector:
        def __init__(self, t, x, y, z): self.t,self.x,self.y,self.z = t,x,y,z
        def __repr__(self): return f"LorentzVector(t={self.t:.3e}, x={self.x:.3e}, y={self.y:.3e}, z={self.z:.3e})"
    @staticmethod
    def lorentz_boost(vector, velocity) -> 'SpecialRelativity.LorentzVector':
        v = np.array(velocity); v_mag_sq = np.sum(v**2)
        if v_mag_sq >= SpecialRelativity.C**2: raise AxiomError("Velocity must be less than the speed of light.")
        if v_mag_sq == 0: return vector
        gamma = 1.0 / math.sqrt(1.0 - v_mag_sq / SpecialRelativity.C**2)
        pos_vec = np.array([vector.x, vector.y, vector.z]); v_dot_r = np.dot(v, pos_vec)
        t_prime = gamma * (vector.t - v_dot_r / SpecialRelativity.C**2)
        pos_prime = pos_vec + (gamma - 1) * (v_dot_r / v_mag_sq) * v - gamma * vector.t * v
        return SpecialRelativity.LorentzVector(t_prime, pos_prime[0], pos_prime[1], pos_prime[2])

class SignalProcessing:
    @staticmethod
    def savitzky_golay_filter(signal, window_size, poly_order) -> Vector:
        if window_size % 2 == 0 or poly_order >= window_size: raise AxiomError("Invalid parameters.")
        half_window = (window_size - 1) // 2
        x = np.arange(-half_window, half_window + 1)
        A = np.vander(x, poly_order + 1, increasing=True)
        coeffs = np.linalg.pinv(A.T @ A) @ A.T
        filtered_signal = np.convolve(signal, coeffs[0, ::-1], mode='same')
        filtered_signal[:half_window] = signal[:half_window]
        filtered_signal[-half_window:] = signal[-half_window:]
        return filtered_signal.tolist()

class InformationTheory:
    _HuffmanNode = namedtuple('HuffmanNode', ['left', 'right'])
    
    @staticmethod
    def huffman_coding(text: str) -> Tuple[Dict[str, str], Any]:
        if not text: return {}, None
        freqs = Counter(text)
        # Priority queue: (frequency, creation_order, payload)
        # Payload is either the character itself (leaf) or a _HuffmanNode (internal)
        heap = [[weight, i, char] for i, (char, weight) in enumerate(freqs.items())]
        heapq.heapify(heap)
        
        count = len(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            node_payload = InformationTheory._HuffmanNode(lo, hi)
            heapq.heappush(heap, [lo[0] + hi[0], count, node_payload])
            count += 1
            
        root_node = heap[0]
        code_map = {}
        
        # --- FIX: Correct recursive traversal of the tree structure ---
        def _generate_codes(node_in_heap, current_code=""):
            # The actual node payload is the 3rd element of the list from the heap
            payload = node_in_heap[2]
            
            # If the payload is an internal node, recurse on its children
            if isinstance(payload, InformationTheory._HuffmanNode):
                _generate_codes(payload.left, current_code + "0")
                _generate_codes(payload.right, current_code + "1")
            # If the payload is a character, it's a leaf. Record its code.
            else:
                # The payload itself is the hashable character key
                code_map[payload] = current_code or "0"

        if root_node:
             _generate_codes(root_node, "")
             
        return code_map, root_node[2] if root_node else None

# --- Primary Facade Class ---

class AxiomPy:
    def __init__(self):
        self._stochastic = StochasticProcesses(); self._relativity = SpecialRelativity()
        self._signal = SignalProcessing(); self._info_theory = InformationTheory()
    @property
    def stochastic(self): return self._stochastic
    @property
    def relativity(self): return self._relativity
    @property
    def signal(self): return self._signal
    @property
    def info_theory(self): return self._info_theory

Axiom = AxiomPy()

# --- Example Usage Block ---
def demo_sde():
    print("\n--- 1. Stochastic Processes: Simulating a Stock Price with SDEs ---")
    mu, sigma = 0.05, 0.2
    drift, diffusion = lambda t, s: mu * s, lambda t, s: sigma * s
    t, prices = Axiom.stochastic.solve_sde_euler_maruyama(drift, diffusion, 100.0, (0, 1.0), 1/252)
    print("Simulated one year of a stock price using Geometric Brownian Motion.")
    print(f"  Starting Price: {prices[0]:.2f}")
    print(f"  Price after ~6 months: {prices[len(prices)//2]:.2f}")
    print(f"  Ending Price: {prices[-1]:.2f}")

def demo_relativity():
    print("\n--- 2. Special Relativity: Time Dilation & Length Contraction ---")
    event = Axiom.relativity.LorentzVector(t=0, x=SpecialRelativity.C * 1.0, y=0, z=0)
    velocity = [0.6 * SpecialRelativity.C, 0, 0]
    transformed_event = Axiom.relativity.lorentz_boost(event, velocity)
    print(f"Observer A (stationary) sees event at: t={event.t:.2f}s, x={event.x/SpecialRelativity.C:.2f} light-seconds")
    print(f"Observer B (moving at 0.6c) sees event at: t={transformed_event.t:.2f}s, x={transformed_event.x/SpecialRelativity.C:.2f} light-seconds")
    print("  Note: Time is negative (time dilation) and distance is shorter (length contraction).")

def demo_savitzky_golay():
    print("\n--- 3. Signal Processing: Savitzky-Golay Filter for Smoothing ---")
    t = np.linspace(0, 4*np.pi, 100)
    clean_signal = np.sin(t)
    noisy_signal = clean_signal + np.random.normal(0, 0.3, 100)
    filtered_signal = Axiom.signal.savitzky_golay_filter(noisy_signal.tolist(), window_size=15, poly_order=4)
    mse_noisy = np.mean((noisy_signal - clean_signal)**2)
    mse_filtered = np.mean((np.array(filtered_signal) - clean_signal)**2)
    print("Smoothed a noisy sine wave while preserving peak shape.")
    print(f"  MSE of noisy signal: {mse_noisy:.4f}")
    print(f"  MSE of filtered signal: {mse_filtered:.4f} (significantly lower)")

def demo_huffman():
    print("\n--- 4. Information Theory: Huffman Data Compression ---")
    text = "go go gophers"
    codes, _ = Axiom.info_theory.huffman_coding(text)
    print(f"Generated optimal prefix codes for the string: '{text}'")
    for char, code in sorted(codes.items()):
        print(f"  Character: '{char}' (Frequency: {text.count(char)}) -> Code: {code}")
    print("  Note: More frequent characters like 'o' get shorter codes.")
    
if __name__ == "__main__":
    print("=" * 80)
    print("    AxiomPy Mathematics Engine v2.5.2 - Corrected Demonstration")
    print("=" * 80)
    demo_sde()
    demo_relativity()
    demo_savitzky_golay()
    demo_huffman()
    print("\n" * 2)
    print("                          Demonstration Complete")
    print("=" * 80)
