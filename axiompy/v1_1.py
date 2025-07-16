# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 1.1.0
#   Author: Master Python Developer & Mathematics Expert
#
#   Behavior: Building a universe of complex mathematical operations from
#             fundamental, first-principle implementations.
#
#   Description: A powerful, self-sufficient, and professional-grade
#                mathematics module for Python. It provides a vast array of
#                mathematical functionalities, from basic arithmetic to advanced
#                calculus, linear algebra, and number theory, all without
#                relying on external math libraries like `math` or `scipy`.
#
#   Permitted Libraries: `numpy` is used for high-performance array and
#                        matrix operations, as per the design specification.
#
################################################################################

import numpy as np
from functools import reduce

# --- Helper functions used internally for high-precision calculations ---

def _calculate_pi(precision_terms=10000):
    """Calculates PI using the Nilakantha series for faster convergence."""
    pi = 3.0
    sign = 1
    for i in range(2, 2 * precision_terms + 1, 2):
        pi += sign * (4 / (i * (i + 1) * (i + 2)))
        sign *= -1
    return pi

def _calculate_e(precision_terms=20):
    """Calculates Euler's number (e) using its series expansion."""
    e = 1.0
    factorial_term = 1.0
    for i in range(1, precision_terms + 1):
        factorial_term *= i
        e += 1.0 / factorial_term
    return e


class Constants:
    """A collection of fundamental mathematical constants."""
    PI = _calculate_pi()
    E = _calculate_e()
    TAU = 2 * PI
    GOLDEN_RATIO = (1 + 5**0.5) / 2
    SQRT_2 = 2**0.5


class BasicOps:
    """Encapsulates fundamental arithmetic operations."""
    @staticmethod
    def add(a, b): return a + b
    @staticmethod
    def subtract(a, b): return a - b
    @staticmethod
    def multiply(a, b): return a * b
    @staticmethod
    def divide(a, b):
        if b == 0: raise ValueError("Error: Division by zero is not allowed.")
        return a / b
    @staticmethod
    def power(base, exp): return base ** exp
    @staticmethod
    def root(n, x, precision=1e-12):
        """Calculates the nth root of x using Newton's method."""
        if x == 0: return 0
        if x < 0 and n % 2 == 0:
            raise ValueError("Cannot calculate an even root of a negative number.")
        # Handle negative numbers for odd roots
        if x < 0:
            return -BasicOps.root(n, -x, precision)
            
        guess = x / n
        while True:
            next_guess = ((n - 1) * guess + x / (guess ** (n - 1))) / n
            if abs(guess - next_guess) < precision: return next_guess
            guess = next_guess
    @staticmethod
    def sqrt(x, precision=1e-12): return BasicOps.root(2, x, precision)
    @staticmethod
    def abs(x): return x if x >= 0 else -x


# --- NEW: Advanced Data Type Classes ---

class ComplexNumber:
    """Represents a complex number and defines its arithmetic."""
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __repr__(self):
        if self.imag == 0: return f"{self.real}"
        if self.real == 0: return f"{self.imag}i"
        sign = "+" if self.imag > 0 else "-"
        return f"{self.real} {sign} {abs(self.imag)}i"

    def __add__(self, other):
        if not isinstance(other, ComplexNumber): other = ComplexNumber(other)
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        if not isinstance(other, ComplexNumber): other = ComplexNumber(other)
        return ComplexNumber(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        if not isinstance(other, ComplexNumber): other = ComplexNumber(other)
        real_part = self.real * other.real - self.imag * other.imag
        imag_part = self.real * other.imag + self.imag * other.real
        return ComplexNumber(real_part, imag_part)

    def __truediv__(self, other):
        if not isinstance(other, ComplexNumber): other = ComplexNumber(other)
        denom = other.real**2 + other.imag**2
        if denom == 0: raise ValueError("Division by zero complex number.")
        real_part = (self.real * other.real + self.imag * other.imag) / denom
        imag_part = (self.imag * other.real - self.real * other.imag) / denom
        return ComplexNumber(real_part, imag_part)
        
    def __eq__(self, other):
        if not isinstance(other, ComplexNumber): other = ComplexNumber(other)
        return self.real == other.real and self.imag == other.imag

    def conjugate(self): return ComplexNumber(self.real, -self.imag)
    def modulus(self): return BasicOps.sqrt(self.real**2 + self.imag**2)


class Polynomial:
    """Represents a polynomial and allows for symbolic-like operations."""
    def __init__(self, coeffs):
        self.coeffs = list(coeffs) # [c_n, c_{n-1}, ..., c_0]

    def __repr__(self):
        if not self.coeffs: return "0"
        parts = []
        degree = len(self.coeffs) - 1
        for i, c in enumerate(self.coeffs):
            if abs(c) < 1e-9: continue
            
            power = degree - i
            
            # Coefficient part
            if c == 1 and power != 0: c_str = ""
            elif c == -1 and power != 0: c_str = "-"
            else: c_str = f"{c:.4g}"
            
            # Variable part
            if power == 0: v_str = ""
            elif power == 1: v_str = "x"
            else: v_str = f"x^{power}"
            
            parts.append(f"{c_str}{v_str}")
            
        if not parts: return "0"
        return " + ".join(parts).replace(" + -", " - ")

    def __call__(self, x):
        # Evaluate using Horner's method for efficiency
        return reduce(lambda acc, c: acc * x + c, self.coeffs)

    def __add__(self, other):
        n1, n2 = len(self.coeffs), len(other.coeffs)
        d = abs(n1 - n2)
        c1 = [0]*d + self.coeffs if n1 < n2 else self.coeffs
        c2 = [0]*d + other.coeffs if n2 < n1 else other.coeffs
        return Polynomial([a + b for a, b in zip(c1, c2)])

    def __mul__(self, other):
        n1, n2 = len(self.coeffs), len(other.coeffs)
        new_coeffs = [0] * (n1 + n2 - 1)
        for i, c1 in enumerate(self.coeffs):
            for j, c2 in enumerate(other.coeffs):
                new_coeffs[i+j] += c1*c2
        return Polynomial(new_coeffs)

    def differentiate(self):
        deg = len(self.coeffs) - 1
        if deg < 1: return Polynomial([0])
        new_coeffs = [c * (deg - i) for i, c in enumerate(self.coeffs[:-1])]
        return Polynomial(new_coeffs)


class AdvancedOps:
    @staticmethod
    def factorial(n):
        if not isinstance(n, int) or n < 0: raise ValueError("Factorial is only defined for non-negative integers.")
        return 1 if n == 0 else reduce(lambda x, y: x*y, range(1, n + 1))
    @staticmethod
    def gcd(a, b):
        while b: a, b = b, a % b
        return a
    @staticmethod
    def lcm(a, b):
        if a == 0 or b == 0: return 0
        return abs(a * b) // AdvancedOps.gcd(a, b)
    @staticmethod
    def fibonacci(n):
        if n < 0: raise ValueError("Input must be a non-negative integer.")
        a, b = 0, 1
        for _ in range(n): a, b = b, a + b
        return a

class Trigonometry:
    _TAYLOR_TERMS = 20
    @staticmethod
    def sin(x):
        x = x % Constants.TAU
        if x > Constants.PI: x -= Constants.TAU
        return sum(((-1)**i) * (x**(2*i+1)) / AdvancedOps.factorial(2*i+1) for i in range(Trigonometry._TAYLOR_TERMS))
    @staticmethod
    def cos(x):
        x = x % Constants.TAU
        if x > Constants.PI: x -= Constants.TAU
        return sum(((-1)**i) * (x**(2*i)) / AdvancedOps.factorial(2*i) for i in range(Trigonometry._TAYLOR_TERMS))
    @staticmethod
    def tan(x):
        cosine_val = Trigonometry.cos(x)
        if abs(cosine_val) < 1e-15: raise ValueError("Tangent undefined.")
        return Trigonometry.sin(x) / cosine_val

class Calculus:
    @staticmethod
    def differentiate(func, x, h=1e-7):
        return (func(x + h) - func(x - h)) / (2 * h)
    @staticmethod
    def integrate(func, a, b, n=10000):
        if n % 2 != 0: n += 1
        h = (b - a) / n
        integral = func(a) + func(b)
        integral += 4 * sum(func(a + i * h) for i in range(1, n, 2))
        integral += 2 * sum(func(a + i * h) for i in range(2, n, 2))
        return integral * h / 3

class NumericalAnalysis:
    """Advanced numerical methods for solving complex problems."""
    @staticmethod
    def find_root(func, derivative, x0, tol=1e-7, max_iter=100):
        """Finds a root of a function using Newton-Raphson method."""
        x = x0
        for _ in range(max_iter):
            fx = func(x)
            if abs(fx) < tol: return x
            dfx = derivative(x)
            if dfx == 0: raise ValueError("Derivative is zero. Cannot continue.")
            x = x - fx / dfx
        raise RuntimeError("Root finding failed to converge.")

    @staticmethod
    def minimize(func, x0, learning_rate=0.01, tol=1e-7, max_iter=1000):
        """Finds a local minimum of a function using Gradient Descent."""
        x = x0
        for _ in range(max_iter):
            grad = Calculus.differentiate(func, x)
            if abs(grad) < tol: return x
            x = x - learning_rate * grad
        raise RuntimeError("Minimization failed to converge.")

    @staticmethod
    def solve_ode(f, y0, t_span, h=0.01):
        """Solves an ODE y'(t) = f(t, y) using the 4th-order Runge-Kutta method."""
        t_values = np.arange(t_span[0], t_span[1] + h, h)
        y_values = np.zeros(len(t_values))
        y_values[0] = y0
        for i in range(len(t_values) - 1):
            t, y = t_values[i], y_values[i]
            k1 = h * f(t, y)
            k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
            k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
            k4 = h * f(t + h, y + k3)
            y_values[i + 1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return t_values.tolist(), y_values.tolist()

class LinearAlgebra:
    @staticmethod
    def dot_product(v1, v2): return np.dot(v1, v2)
    @staticmethod
    def cross_product(v1, v2): return np.cross(v1, v2).tolist()
    @staticmethod
    def transpose(matrix): return np.transpose(matrix).tolist()
    @staticmethod
    def inverse(matrix):
        try: return np.linalg.inv(matrix).tolist()
        except np.linalg.LinAlgError: raise ValueError("Matrix is not invertible.")
    @staticmethod
    def determinant(matrix):
        try: return np.linalg.det(matrix)
        except np.linalg.LinAlgError: raise ValueError("Matrix must be square.")
    @staticmethod
    def solve_system(A, b):
        try: return np.linalg.solve(A, b).tolist()
        except np.linalg.LinAlgError: raise ValueError("System may be unsolvable.")

class AdvancedLinearAlgebra:
    """Wraps advanced NumPy functions for matrix decompositions."""
    @staticmethod
    def lu_decomposition(matrix):
        """Performs LU decomposition: A = PLU."""
        import numpy.linalg as la
        p, l, u = la.lu(np.array(matrix))
        return p.tolist(), l.tolist(), u.tolist()
    @staticmethod
    def qr_decomposition(matrix):
        """Performs QR decomposition."""
        q, r = np.linalg.qr(matrix)
        return q.tolist(), r.tolist()
    @staticmethod
    def eigenvalues_eigenvectors(matrix):
        """Calculates eigenvalues and eigenvectors of a matrix."""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return eigenvalues.tolist(), eigenvectors.tolist()

class Statistics:
    @staticmethod
    def mean(data): return sum(data) / len(data)
    @staticmethod
    def median(data):
        s, n = sorted(data), len(data)
        mid = n // 2
        return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]
    @staticmethod
    def variance(data, is_sample=False):
        n = len(data)
        if n < 2: return 0
        mean_val = Statistics.mean(data)
        denom = (n - 1) if is_sample else n
        return sum((x - mean_val) ** 2 for x in data) / denom
    @staticmethod
    def std_dev(data, is_sample=False): return BasicOps.sqrt(Statistics.variance(data, is_sample))
    @staticmethod
    def pearson_correlation(x_data, y_data):
        """Calculates the Pearson correlation coefficient between two datasets."""
        n = len(x_data)
        if n != len(y_data): raise ValueError("Datasets must have equal length.")
        mean_x, mean_y = Statistics.mean(x_data), Statistics.mean(y_data)
        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_data, y_data))
        std_dev_x = BasicOps.sqrt(sum((x - mean_x) ** 2 for x in x_data))
        std_dev_y = BasicOps.sqrt(sum((y - mean_y) ** 2 for y in y_data))
        if std_dev_x == 0 or std_dev_y == 0: return 0
        return cov_xy / (std_dev_x * std_dev_y)
    @staticmethod
    def linear_regression(x_data, y_data):
        """Performs simple linear regression, returning slope (m) and intercept (b)."""
        n = len(x_data)
        if n != len(y_data): raise ValueError("Datasets must have equal length.")
        mean_x, mean_y = Statistics.mean(x_data), Statistics.mean(y_data)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_data, y_data))
        denominator = sum((x - mean_x) ** 2 for x in x_data)
        if denominator == 0: raise ValueError("Cannot perform regression on data with zero variance in x.")
        m = numerator / denominator
        b = mean_y - m * mean_x
        return {'slope': m, 'intercept': b}
        
class NumberTheory:
    @staticmethod
    def is_prime(n):
        if n <= 1: return False
        if n <= 3: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0: return False
            i += 6
        return True
    @staticmethod
    def prime_factors(n):
        factors = []
        while n % 2 == 0: factors.append(2); n //= 2
        for i in range(3, int(BasicOps.sqrt(n)) + 1, 2):
            while n % i == 0: factors.append(i); n //= i
        if n > 2: factors.append(n)
        return factors

################################################################################
#
#   Primary Facade Class: Axiom
#
################################################################################

class Axiom(
    Constants, BasicOps, AdvancedOps, Trigonometry, Calculus, NumericalAnalysis,
    LinearAlgebra, AdvancedLinearAlgebra, Statistics, NumberTheory
):
    """
    The unified interface for the AxiomPy Mathematics Engine.
    Also provides access to advanced data types like ComplexNumber and Polynomial.
    """
    # Expose the data type classes directly for easy instantiation
    Complex = ComplexNumber
    Poly = Polynomial
    
    def __init__(self):
        raise TypeError("AxiomPy is a static utility class and cannot be instantiated.")

# --- Example Usage Block ---
if __name__ == "__main__":
    print("=" * 60)
    print("    AxiomPy Mathematics Engine v1.1.0 - Demonstration")
    print("=" * 60)

    # 1. Complex Numbers
    print("\n--- 1. Complex Numbers ---")
    c1 = Axiom.Complex(3, 4)
    c2 = Axiom.Complex(1, -2)
    print(f"c1 = {c1}, c2 = {c2}")
    print(f"c1 + c2 = {c1 + c2}")
    print(f"c1 * c2 = {c1 * c2}")
    print(f"c1 / c2 = {c1 / c2}")
    print(f"Modulus of c1: {c1.modulus():.4f}")
    print(f"sqrt(-9) is represented as: {Axiom.Complex(0, Axiom.sqrt(9))}")

    # 2. Polynomials
    print("\n--- 2. Polynomials ---")
    # P(x) = x^2 - 4
    p1 = Axiom.Poly([1, 0, -4])
    # Q(x) = 2x + 1
    p2 = Axiom.Poly([2, 1])
    print(f"P(x) = {p1}")
    print(f"Q(x) = {p2}")
    print(f"P(3) = {p1(3)}")
    print(f"P(x) + Q(x) = {p1 + p2}")
    print(f"P(x) * Q(x) = {p1 * p2}")
    print(f"Derivative of P(x): {p1.differentiate()}")

    # 3. Numerical Analysis
    print("\n--- 3. Numerical Analysis ---")
    # a) Root Finding: Find root of f(x) = x^3 - x - 2
    f = lambda x: x**3 - x - 2
    df = lambda x: 3*x**2 - 1
    root = Axiom.find_root(f, df, x0=1.5)
    print(f"Root of x^3 - x - 2 is approx: {root:.6f}")
    # b) Optimization: Find minimum of f(x) = (x-4)^2 + 5
    f_min = lambda x: (x-4)**2 + 5
    minimum = Axiom.minimize(f_min, x0=10)
    print(f"Minimum of (x-4)^2 + 5 is at x = {minimum:.6f}")
    # c) ODE Solver: Solve y' = y, y(0)=1 (solution is e^t)
    f_ode = lambda t, y: y
    t_vals, y_vals = Axiom.solve_ode(f_ode, y0=1, t_span=[0, 2], h=0.2)
    print("Solving y' = y, y(0)=1:")
    print(f"  t={t_vals[5]:.1f}, y_numerical={y_vals[5]:.4f}, y_exact={Axiom.E**t_vals[5]:.4f}")
    print(f"  t={t_vals[-1]:.1f}, y_numerical={y_vals[-1]:.4f}, y_exact={Axiom.E**t_vals[-1]:.4f}")

    # 4. Advanced Linear Algebra
    print("\n--- 4. Advanced Linear Algebra ---")
    matrix = [[4, 3], [6, 3]]
    print(f"Matrix A = {matrix}")
    w, v = Axiom.eigenvalues_eigenvectors(matrix)
    print(f"Eigenvalues: {[round(val, 4) for val in w]}")
    print(f"Eigenvectors:\n{np.round(v, 4)}") # Use np here for nice printing
    
    # 5. Advanced Statistics
    print("\n--- 5. Advanced Statistics ---")
    x_data = [1, 2, 3, 4, 5, 6]
    y_data = [1.8, 4.5, 6.1, 8.3, 10.1, 11.8] # Strong positive correlation
    corr = Axiom.pearson_correlation(x_data, y_data)
    print(f"Pearson Correlation: {corr:.6f}")
    regression_fit = Axiom.linear_regression(x_data, y_data)
    print(f"Linear Regression: y = {regression_fit['slope']:.4f}x + {regression_fit['intercept']:.4f}")

    print("\n" + "=" * 60)
    print("           Demonstration Complete")
    print("=" * 60)
