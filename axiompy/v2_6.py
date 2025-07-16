# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 2.6.0
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A premier scientific computing engine. This version introduces
#                a General Relativity engine for spacetime calculations, the
#                Power Iteration method for finding dominant eigenvalues, a
#                Quadtree for advanced spatial searches, and a Fourier Optics
#                simulator for modeling light diffraction.
#
################################################################################

import numpy as np
import random
import math
from typing import (List, Tuple, Callable, Any, TypeVar, Set, Generic, Dict)
from collections import namedtuple

# --- Type Aliases & Core Infrastructure ---
Vector = List[float]; Matrix = List[List[float]]
Point = namedtuple('Point', ['x', 'y'])
# (Other core data types assumed present for brevity)

class AxiomError(Exception): pass
class ConvergenceError(AxiomError): pass

class Constants:
    """A collection of fundamental physical and mathematical constants."""
    C: float = 299792458.0  # Speed of light in m/s
    G: float = 6.67430e-11 # Gravitational constant in m^3 kg^-1 s^-2

# --- Scientific and Mathematical Domain Classes ---

class LinearAlgebra:
    """Core and advanced linear algebra operations."""
    @staticmethod
    def power_iteration(
        matrix: Matrix, max_iter: int = 1000, tol: float = 1e-6
    ) -> Tuple[float, Vector]:
        """
        Finds the dominant eigenvalue and eigenvector of a matrix using Power Iteration.
        """
        A = np.array(matrix)
        n = A.shape[0]
        b_k = np.random.rand(n)
        
        for _ in range(max_iter):
            b_k1 = A @ b_k
            b_k1_norm = np.linalg.norm(b_k1)
            b_k_next = b_k1 / b_k1_norm
            if np.linalg.norm(b_k_next - b_k) < tol: break
            b_k = b_k_next
            
        eigenvalue = (b_k.T @ A @ b_k) / (b_k.T @ b_k)
        return float(eigenvalue), b_k.tolist()
    #... Other linalg functions assumed present

class SignalProcessing:
    """Advanced algorithms for signal analysis and processing."""
    @staticmethod
    def fft2d(matrix: Matrix) -> Matrix:
        """Computes the 2D Fast Fourier Transform of a matrix."""
        transformed = np.fft.fft2(matrix)
        # We must return a list of lists of complex numbers for API consistency
        return [[complex(c) for c in row] for row in transformed]

class GeneralRelativity:
    """Functions for calculations based on Einstein's General Relativity."""
    @staticmethod
    def schwarzschild_radius(mass_kg: float) -> float:
        """Calculates the Schwarzschild radius (event horizon) for a given mass."""
        return (2 * Constants.G * mass_kg) / (Constants.C**2)
    @staticmethod
    def gravitational_time_dilation(mass_kg: float, distance_from_center: float, proper_time: float) -> float:
        """
        Calculates the time elapsed for a distant observer given the proper time
        elapsed for an observer within the gravitational field.
        """
        rs = GeneralRelativity.schwarzschild_radius(mass_kg)
        if distance_from_center <= rs: raise AxiomError("Distance is inside the event horizon.")
        return proper_time / math.sqrt(1 - rs / distance_from_center)

class Spatial:
    """Advanced data structures and algorithms for spatial problems."""
    class Quadtree:
        class _Node:
            def __init__(self, boundary: 'Spatial.Quadtree.Rectangle', capacity: int):
                self.boundary, self.capacity = boundary, capacity
                self.points, self.divided = [], False
            def subdivide(self):
                x, y, w, h = self.boundary
                hw, hh = w/2, h/2
                self.nw = Spatial.Quadtree._Node(Spatial.Quadtree.Rectangle(x-hw/2, y+hh/2, hw, hh), self.capacity)
                self.ne = Spatial.Quadtree._Node(Spatial.Quadtree.Rectangle(x+hw/2, y+hh/2, hw, hh), self.capacity)
                self.sw = Spatial.Quadtree._Node(Spatial.Quadtree.Rectangle(x-hw/2, y-hh/2, hw, hh), self.capacity)
                self.se = Spatial.Quadtree._Node(Spatial.Quadtree.Rectangle(x+hw/2, y-hh/2, hw, hh), self.capacity)
                self.divided = True
        
        Rectangle = namedtuple('Rectangle', ['x', 'y', 'w', 'h'])
        def __init__(self, boundary: 'Spatial.Quadtree.Rectangle', capacity: int = 4):
            self.root = Spatial.Quadtree._Node(boundary, capacity)
        def insert(self, point: Point) -> bool:
            def _insert_recursive(node, pt):
                if not node.boundary.x - node.boundary.w/2 <= pt.x <= node.boundary.x + node.boundary.w/2 or \
                   not node.boundary.y - node.boundary.h/2 <= pt.y <= node.boundary.y + node.boundary.h/2:
                    return False
                if len(node.points) < node.capacity:
                    node.points.append(pt); return True
                if not node.divided: node.subdivide()
                return _insert_recursive(node.ne, pt) or _insert_recursive(node.nw, pt) or \
                       _insert_recursive(node.se, pt) or _insert_recursive(node.sw, pt)
            return _insert_recursive(self.root, point)
        def query_range(self, search_range: 'Spatial.Quadtree.Rectangle') -> List[Point]:
            found = []
            def _query_recursive(node, rng):
                if not (abs(node.boundary.x - rng.x) <= (node.boundary.w + rng.w)/2 and \
                        abs(node.boundary.y - rng.y) <= (node.boundary.h + rng.h)/2):
                    return
                for p in node.points:
                    if rng.x - rng.w/2 <= p.x <= rng.x + rng.w/2 and \
                       rng.y - rng.h/2 <= p.y <= rng.y + rng.h/2:
                        found.append(p)
                if node.divided:
                    _query_recursive(node.nw, rng); _query_recursive(node.ne, rng)
                    _query_recursive(node.sw, rng); _query_recursive(node.se, rng)
            _query_recursive(self.root, search_range)
            return found

class FourierOptics:
    """Functions for simulating optical phenomena using Fourier transforms."""
    @staticmethod
    def fraunhofer_diffraction(aperture_matrix: Matrix) -> Matrix:
        """
        Simulates the far-field (Fraunhofer) diffraction pattern of an aperture.
        Returns the intensity pattern.
        """
        fft_result = SignalProcessing.fft2d(aperture_matrix)
        # Shift the zero-frequency component to the center for visualization
        shifted_fft = np.fft.fftshift([[complex(c) for c in row] for row in fft_result])
        # Intensity is the squared magnitude of the complex amplitude
        intensity = np.abs(shifted_fft)**2
        return intensity.tolist()
        
# --- Primary Facade Class ---
class AxiomPy:
    def __init__(self):
        self._linalg = LinearAlgebra(); self._relativity = GeneralRelativity()
        self._spatial = Spatial(); self._fourier_optics = FourierOptics()
    @property
    def linalg(self): return self._linalg
    @property
    def relativity(self): return self._relativity
    @property
    def spatial(self): return self._spatial
    @property
    def fourier_optics(self): return self._fourier_optics

Axiom = AxiomPy()

# --- Example Usage Block ---
def demo_relativity():
    print("\n--- 1. General Relativity: Spacetime Calculations ---")
    earth_mass = 5.972e24 # kg
    earth_radius = 6.371e6 # meters
    rs_earth = Axiom.relativity.schwarzschild_radius(earth_mass)
    print(f"Schwarzschild Radius (Event Horizon) of Earth: {rs_earth*1000:.2f} mm")
    
    gps_altitude = 20.2e6 # meters
    time_on_gps = 24 * 3600 # 1 day
    time_on_earth = Axiom.relativity.gravitational_time_dilation(earth_mass, earth_radius + gps_altitude, time_on_gps)
    time_gain_us = (time_on_earth - time_on_gps) * 1e6
    print(f"A GPS satellite clock after 1 day on Earth would be ahead by: {time_gain_us:.2f} microseconds (due to GR)")

def demo_power_iteration():
    print("\n--- 2. Numerical Linear Algebra: Power Iteration Method ---")
    A = [[6, 5], [1, 2]]
    # True dominant eigenvalue is 7
    eigenvalue, eigenvector = Axiom.linalg.power_iteration(A)
    print("Finding dominant eigenvalue/vector for matrix:", A)
    print(f"  Calculated Eigenvalue: {eigenvalue:.4f} (Exact: 7)")
    print(f"  Calculated Eigenvector: {[f'{v:.4f}' for v in eigenvector]}")

def demo_quadtree():
    print("\n--- 3. Spatial Data Structures: Quadtree Range Search ---")
    boundary = Axiom.spatial.Quadtree.Rectangle(50, 50, 100, 100)
    qtree = Axiom.spatial.Quadtree(boundary, 4)
    for _ in range(100):
        qtree.insert(Point(random.uniform(0, 100), random.uniform(0, 100)))
    
    search_area = Axiom.spatial.Quadtree.Rectangle(60, 60, 40, 40)
    found_points = qtree.query_range(search_area)
    print("Inserted 100 random points into a Quadtree.")
    print(f"Found {len(found_points)} points within the search area centered at (60,60).")
    print(f"  Example point found: {found_points[0]}" if found_points else "  No points found.")

def demo_fourier_optics():
    print("\n--- 4. Fourier Optics: Simulating Slit Diffraction ---")
    size = 128
    aperture = np.zeros((size, size))
    # Create a tall, thin vertical slit
    aperture[:, size//2 - 2 : size//2 + 2] = 1
    
    intensity = Axiom.fourier_optics.fraunhofer_diffraction(aperture.tolist())
    print("Generated diffraction pattern for a single slit. (ASCII representation)")
    # Print a small, scaled down ASCII version
    chars = " .,-~:;=!*#$@"
    for i in range(0, 30, 2):
        row = intensity[size//2 - 15 + i]
        line = ""
        for j in range(0, 60, 2):
            val = row[size//2 - 30 + j]
            log_val = math.log10(1 + val) if val > 0 else 0
            max_log = math.log10(1 + np.max(intensity))
            char_idx = int((log_val / max_log) * (len(chars) - 1)) if max_log > 0 else 0
            line += chars[char_idx]
        print(f"  {line}")

if __name__ == "__main__":
    print("=" * 80)
    print("    AxiomPy Mathematics Engine v2.6.0 - Premier Scientific Demonstration")
    print("=" * 80)
    demo_relativity()
    demo_power_iteration()
    demo_quadtree()
    demo_fourier_optics()
    print("\n" * 2)
    print("                          Demonstration Complete")
    print("=" * 80)
