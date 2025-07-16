# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 1.6.0
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A scientific and professional-grade mathematics module.
#                This version introduces powerful capabilities for applied science,
#                including a Quantum Physics engine for solving the Schrödinger
#                equation, a Control Systems simulator, and foundational tools
#                for Signal and Image Processing like convolution and edge detection.
#
################################################################################

import numpy as np
from functools import reduce
from typing import (List, Tuple, Callable, Dict, Union, TypeVar, Literal)

# --- Type Aliases for Enhanced Clarity ---
NodeType = TypeVar('NodeType')
Vector = List[float]
Matrix = List[List[float]]

# --- Core Data Types and Foundational Classes ---
# (Assuming full implementation of ComplexNumber, Polynomial, Graph, Statistics etc. for brevity)
class Constants:
    PI: float = 3.141592653589793; E: float = 2.718281828459045
    H_BAR: float = 1.054571817e-34  # Reduced Planck Constant (J*s)
    ELECTRON_MASS: float = 9.1093837e-31 # Electron Mass (kg)
class AdvancedOps:
    @staticmethod
    def gcd(a,b): #...
        while b: a, b = b, a % b
        return abs(a)
class RationalNumber: #...
    def __init__(self,n,d=1): self.num,self.den=n,d

# --- Scientific and Mathematical Domain Classes ---

class LinearAlgebra:
    """Core linear algebra operations, powered by NumPy for performance."""
    @staticmethod
    def eigenvalues_eigenvectors(matrix: Matrix) -> Tuple[Vector, Matrix]:
        w, v = np.linalg.eig(matrix)
        # Sort eigenvalues and corresponding eigenvectors
        idx = w.argsort()
        eigenvalues = w[idx]
        eigenvectors = v[:, idx]
        return eigenvalues.tolist(), eigenvectors.T.tolist()
    #... Other linalg functions assumed present

class SignalProcessing:
    """Algorithms for signal analysis, filtering, and processing."""
    @staticmethod
    def convolve(signal: Vector, kernel: Vector, mode: Literal['full', 'same', 'valid'] = 'full') -> Vector:
        """Performs 1D convolution of a signal with a kernel."""
        n_sig, n_ker = len(signal), len(kernel)
        n_conv = n_sig + n_ker - 1
        result = [0.0] * n_conv
        
        for i in range(n_conv):
            for j in range(n_ker):
                if 0 <= i - j < n_sig:
                    result[i] += signal[i - j] * kernel[j]
        
        if mode == 'valid':
            start = n_ker - 1
            end = n_sig
            return result[start:end]
        if mode == 'same':
            start = (n_ker - 1) // 2
            end = start + n_sig
            return result[start:end]
        return result

    @staticmethod
    def hann_window(size: int) -> Vector:
        """Generates a Hann (or Hanning) window."""
        return [0.5 * (1 - np.cos(2 * Constants.PI * i / (size - 1))) for i in range(size)]

    @staticmethod
    def hamming_window(size: int) -> Vector:
        """Generates a Hamming window."""
        return [0.54 - 0.46 * np.cos(2 * Constants.PI * i / (size - 1)) for i in range(size)]

class ImageProcessing:
    """Foundational algorithms for digital image processing."""
    @staticmethod
    def _convolve2d(image: Matrix, kernel: Matrix) -> Matrix:
        """Helper for 2D convolution with zero-padding."""
        img_h, img_w = len(image), len(image[0])
        ker_h, ker_w = len(kernel), len(kernel[0])
        pad_h, pad_w = ker_h // 2, ker_w // 2
        
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        output = np.zeros_like(image, dtype=float)
        
        for r in range(img_h):
            for c in range(img_w):
                region = padded[r:r+ker_h, c:c+ker_w]
                output[r, c] = np.sum(region * kernel)
        return output.tolist()

    @staticmethod
    def sobel_edge_detection(image: Matrix) -> Matrix:
        """
        Detects edges in an image (2D matrix) using the Sobel operator.
        Returns a matrix of gradient magnitudes.
        """
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        grad_x = np.array(ImageProcessing._convolve2d(image, kernel_x))
        grad_y = np.array(ImageProcessing._convolve2d(image, kernel_y))
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # Normalize to 0-255 range for typical display
        if magnitude.max() > 0:
            magnitude = (magnitude / magnitude.max()) * 255
        
        return magnitude.tolist()

class ControlSystems:
    """Algorithms for modeling and simulating dynamical systems."""
    @staticmethod
    def simulate_lti(
        A: Matrix, B: Matrix, C: Matrix, D: Matrix,
        u: Vector, x0: Vector = None
    ) -> Tuple[Matrix, Vector]:
        """
        Simulates a discrete-time Linear Time-Invariant (LTI) system.
        State-space form: x[k+1]=Ax[k]+Bu[k], y[k]=Cx[k]+Du[k]
        :param u: Input signal vector over time.
        :param x0: Initial state vector.
        :return: Tuple of (state_history, output_history).
        """
        A, B, C, D = map(np.array, [A, B, C, D])
        n_states = A.shape[0]
        n_outputs = C.shape[0]
        n_steps = len(u)
        
        x = np.zeros(n_states) if x0 is None else np.array(x0)
        x_history = np.zeros((n_steps, n_states))
        y_history = np.zeros((n_steps, n_outputs))
        
        for k in range(n_steps):
            x_history[k, :] = x
            y_history[k, :] = C @ x + D @ np.array([u[k]])
            x = A @ x + B @ np.array([u[k]])
            
        return x_history.tolist(), y_history.tolist()

class QuantumPhysics:
    """Solvers for fundamental problems in quantum mechanics."""
    @staticmethod
    def solve_particle_in_box(
        L: float = 1e-9,  # Width of the box (e.g., 1 nm)
        n_points: int = 100
    ) -> Dict[str, Union[Vector, Matrix]]:
        """
        Solves the 1D time-independent Schrödinger equation for an infinite
        potential well ("particle in a box").
        :return: A dictionary with 'energies' (in eV) and 'wavefunctions'.
        """
        dx = L / (n_points + 1)
        
        # Construct the Hamiltonian matrix
        # Kinetic part: -(hbar^2 / 2m) * d^2/dx^2
        # Approximated by a finite difference matrix
        main_diag = np.full(n_points, 2.0)
        off_diag = np.full(n_points - 1, -1.0)
        
        H = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        
        # Constant factor
        factor = -(Constants.H_BAR**2) / (2 * Constants.ELECTRON_MASS * dx**2)
        H *= factor
        
        # Solve the eigenvalue problem H * psi = E * psi
        eigenvalues, eigenvectors = LinearAlgebra.eigenvalues_eigenvectors(H.tolist())
        
        # Convert energies from Joules to electron-volts (eV)
        energies_eV = [E / 1.60218e-19 for E in eigenvalues]
        
        # Normalize wavefunctions: integral |psi|^2 dx = 1
        normalized_wavefuncs = []
        for psi in eigenvectors:
            norm = np.sqrt(np.sum(np.array(psi)**2) * dx)
            normalized_wavefuncs.append((np.array(psi) / norm).tolist())
        
        return {'energies_eV': energies_eV, 'wavefunctions': normalized_wavefuncs}


# --- Primary Facade Class: Composition-based Design ---

class AxiomPy:
    def __init__(self):
        # Instantiate each domain as a property-like attribute
        self._linalg = LinearAlgebra()
        self._signal = SignalProcessing()
        self._image = ImageProcessing()
        self._control = ControlSystems()
        self._quantum = QuantumPhysics()

        # Expose data types directly
        self.Rational = RationalNumber
        self.Graph = None # Placeholder for Graph class

    @property
    def linalg(self) -> LinearAlgebra: return self._linalg
    @property
    def signal(self) -> SignalProcessing: return self._signal
    @property
    def image(self) -> ImageProcessing: return self._image
    @property
    def control(self) -> ControlSystems: return self._control
    @property
    def quantum(self) -> QuantumPhysics: return self._quantum
    @property
    def constants(self) -> Constants: return Constants()

# Create a single, ready-to-use instance
Axiom = AxiomPy()

# --- Example Usage Block ---
if __name__ == "__main__":
    print("=" * 70)
    print("    AxiomPy Mathematics Engine v1.6.0 - Scientific Demonstration")
    print("=" * 70)

    # 1. Quantum Physics: Particle in a Box
    print("\n--- 1. Quantum Physics Engine ---")
    solution = Axiom.quantum.solve_particle_in_box()
    print("Solved Schrödinger equation for a particle in a 1 nm box.")
    print("Lowest 3 energy levels (eigenvalues):")
    for i in range(3):
        analytical_E = (i+1)**2 * Constants.PI**2 * Constants.H_BAR**2 / (2 * Constants.ELECTRON_MASS * (1e-9)**2 * 1.602e-19)
        print(f"  n={i+1}: {solution['energies_eV'][i]:.4f} eV (Analytical: {analytical_E:.4f} eV)")
    print("Wavefunctions (eigenvectors) are also computed.")
    
    # 2. Control Systems: LTI Simulation
    print("\n--- 2. Control Systems Simulation ---")
    # A simple damped second-order system (e.g., mass-spring-damper)
    A = [[0, 1], [-0.5, -0.2]]
    B = [[0], [1]]
    C = [[1, 0]]
    D = [[0]]
    step_input = [1.0] * 50 # Apply a constant force
    _, output = Axiom.control.simulate_lti(A, B, C, D, u=step_input)
    print("Simulating a damped oscillator with a step input:")
    print(f"  Output at t=0: {output[0][0]:.4f}")
    print(f"  Output at t=10: {output[10][0]:.4f}")
    print(f"  Output at t=49: {output[49][0]:.4f} (approaching steady-state)")
    
    # 3. Signal Processing: Convolution and Windowing
    print("\n--- 3. Advanced Signal Processing ---")
    raw_signal = [0, 0, 1, 1, 1, 0, 0]
    moving_avg_kernel = [1/3, 1/3, 1/3]
    smoothed_signal = Axiom.signal.convolve(raw_signal, moving_avg_kernel, mode='same')
    print(f"Original Signal:      {raw_signal}")
    print(f"Smoothed (convolved): {[f'{x:.2f}' for x in smoothed_signal]}")
    hann_win = Axiom.signal.hann_window(5)
    print(f"Hann window (size=5): {[f'{x:.2f}' for x in hann_win]}")

    # 4. Image Processing: Sobel Edge Detection
    print("\n--- 4. Digital Image Processing ---")
    # A simple image of a light square on a dark background
    simple_image = [[0, 0, 0, 0, 0, 0],
                    [0, 100, 100, 100, 0, 0],
                    [0, 100, 100, 100, 0, 0],
                    [0, 100, 100, 100, 0, 0],
                    [0, 0, 0, 0, 0, 0]]
    edge_map = Axiom.image.sobel_edge_detection(simple_image)
    print("Detecting edges in a simple image (values are gradient magnitudes):")
    for row in edge_map:
        print(f"  [{' '.join(f'{int(p):>3d}' for p in row)}]")

    print("\n" + "=" * 70)
    print("               Demonstration Complete")
    print("=" * 70)
