# operator.py

################################################################################
#
#   AxiomPy: The Python Mathematics & Computation Engine
#
#   Version: 1.0.0
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
    """
    A collection of fundamental mathematical constants, calculated to a high
    degree of precision without using the standard `math` library.
    """
    PI = _calculate_pi()
    E = _calculate_e()
    TAU = 2 * PI
    GOLDEN_RATIO = (1 + 5**0.5) / 2
    SQRT_2 = 2**0.5


class BasicOps:
    """Encapsulates fundamental arithmetic operations."""

    @staticmethod
    def add(a, b):
        """Returns the sum of a and b."""
        return a + b

    @staticmethod
    def subtract(a, b):
        """Returns the difference of a and b."""
        return a - b

    @staticmethod
    def multiply(a, b):
        """Returns the product of a and b."""
        return a * b

    @staticmethod
    def divide(a, b):
        """
        Returns the quotient of a and b.
        Raises ValueError on division by zero.
        """
        if b == 0:
            raise ValueError("Error: Division by zero is not allowed.")
        return a / b

    @staticmethod
    def power(base, exp):
        """Returns the base raised to the power of the exponent."""
        return base ** exp

    @staticmethod
    def root(n, x, precision=1e-12):
        """
        Calculates the nth root of x using Newton's method.
        :param n: The root degree (e.g., 2 for square root).
        :param x: The number to find the root of.
        :param precision: The desired accuracy of the result.
        :return: The nth root of x.
        """
        if x < 0 and n % 2 == 0:
            raise ValueError("Cannot calculate an even root of a negative number.")
        if x == 0:
            return 0
            
        guess = x / n # Initial guess
        while True:
            next_guess = ((n - 1) * guess + x / (guess ** (n - 1))) / n
            if abs(guess - next_guess) < precision:
                return next_guess
            guess = next_guess

    @staticmethod
    def sqrt(x, precision=1e-12):
        """Calculates the square root of x."""
        return BasicOps.root(2, x, precision)

    @staticmethod
    def abs(x):
        """Returns the absolute value of a number."""
        return x if x >= 0 else -x


class AdvancedOps:
    """Provides more advanced, non-domain-specific operations."""

    @staticmethod
    def factorial(n):
        """
        Calculates the factorial of a non-negative integer.
        :param n: A non-negative integer.
        :return: n!
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError("Factorial is only defined for non-negative integers.")
        if n == 0:
            return 1
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    @staticmethod
    def gcd(a, b):
        """
        Calculates the Greatest Common Divisor (GCD) of two integers
        using the Euclidean algorithm.
        """
        while b:
            a, b = b, a % b
        return a

    @staticmethod
    def lcm(a, b):
        """
        Calculates the Least Common Multiple (LCM) of two integers.
        """
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // AdvancedOps.gcd(a, b)

    @staticmethod
    def fibonacci(n):
        """
        Returns the nth number in the Fibonacci sequence (iterative).
        :param n: The index in the sequence (starts from 0).
        """
        if n < 0:
            raise ValueError("Input must be a non-negative integer.")
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a


class Trigonometry:
    """
    Implements trigonometric functions using Taylor series expansions.
    This demonstrates from-scratch implementation without the `math` module.
    """
    _TAYLOR_TERMS = 20

    @staticmethod
    def sin(x):
        """
        Calculates the sine of x (in radians) using a Taylor series.
        """
        # Normalize x to the range [-PI, PI] for accuracy and convergence
        x = x % Constants.TAU
        if x > Constants.PI:
            x -= Constants.TAU

        result = 0.0
        for i in range(Trigonometry._TAYLOR_TERMS):
            sign = (-1) ** i
            term = x ** (2 * i + 1) / AdvancedOps.factorial(2 * i + 1)
            result += sign * term
        return result

    @staticmethod
    def cos(x):
        """
        Calculates the cosine of x (in radians) using a Taylor series.
        """
        # Normalize x to the range [-PI, PI]
        x = x % Constants.TAU
        if x > Constants.PI:
            x -= Constants.TAU

        result = 0.0
        for i in range(Trigonometry._TAYLOR_TERMS):
            sign = (-1) ** i
            term = x ** (2 * i) / AdvancedOps.factorial(2 * i)
            result += sign * term
        return result

    @staticmethod
    def tan(x):
        """Calculates the tangent of x (in radians)."""
        cosine_val = Trigonometry.cos(x)
        if abs(cosine_val) < 1e-15:
            raise ValueError("Tangent is undefined for this input (division by zero).")
        return Trigonometry.sin(x) / cosine_val


class Calculus:
    """Provides numerical methods for calculus operations."""

    @staticmethod
    def differentiate(func, x, h=1e-7):
        """
        Computes the numerical derivative of a function at a point x
        using the central difference formula.
        :param func: A single-variable function (e.g., lambda x: x**2).
        :param x: The point at which to evaluate the derivative.
        :param h: A small step size for the calculation.
        :return: The approximate derivative of func at x.
        """
        return (func(x + h) - func(x - h)) / (2 * h)

    @staticmethod
    def integrate(func, a, b, n=10000):
        """
        Computes the numerical definite integral of a function from a to b
        using Simpson's rule.
        :param func: The single-variable function to integrate.
        :param a: The lower limit of integration.
        :param b: The upper limit of integration.
        :param n: The number of intervals (must be even).
        :return: The approximate area under the curve.
        """
        if n % 2 != 0:
            n += 1 # Ensure n is even for Simpson's rule
        
        h = (b - a) / n
        integral = func(a) + func(b)

        for i in range(1, n, 2):
            integral += 4 * func(a + i * h)
        for i in range(2, n, 2):
            integral += 2 * func(a + i * h)
            
        return integral * h / 3


class LinearAlgebra:
    """
    Wraps powerful NumPy functions for linear algebra to provide a consistent
    API within the AxiomPy module. It accepts standard Python lists and
    handles the conversion to NumPy arrays.
    """

    @staticmethod
    def dot_product(v1, v2):
        """Calculates the dot product of two vectors."""
        return np.dot(v1, v2)

    @staticmethod
    def cross_product(v1, v2):
        """Calculates the cross product of two 3D vectors."""
        return np.cross(v1, v2).tolist()

    @staticmethod
    def transpose(matrix):
        """Transposes a matrix (rows become columns)."""
        return np.transpose(matrix).tolist()

    @staticmethod
    def inverse(matrix):
        """Calculates the inverse of a square matrix."""
        try:
            return np.linalg.inv(matrix).tolist()
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is not invertible (it is singular).")

    @staticmethod
    def determinant(matrix):
        """Calculates the determinant of a square matrix."""
        try:
            return np.linalg.det(matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Determinant can only be calculated for a square matrix.")

    @staticmethod
    def solve_system(A, b):
        """Solves a system of linear equations Ax = b."""
        try:
            return np.linalg.solve(A, b).tolist()
        except np.linalg.LinAlgError:
            raise ValueError("System may be unsolvable or have infinite solutions.")


class Statistics:
    """Provides common statistical measures for a list of numerical data."""

    @staticmethod
    def mean(data):
        """Calculates the arithmetic mean (average) of a dataset."""
        return sum(data) / len(data)

    @staticmethod
    def median(data):
        """Calculates the median of a dataset."""
        sorted_data = sorted(data)
        n = len(sorted_data)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_data[mid - 1] + sorted_data[mid]) / 2
        else:
            return sorted_data[mid]

    @staticmethod
    def variance(data, is_sample=False):
        """
        Calculates the variance of a dataset.
        :param is_sample: If True, calculates sample variance (N-1 denominator).
                          Otherwise, calculates population variance (N denominator).
        """
        n = len(data)
        if n < 2:
            return 0
        mean_val = Statistics.mean(data)
        denominator = (n - 1) if is_sample else n
        return sum((x - mean_val) ** 2 for x in data) / denominator

    @staticmethod
    def std_dev(data, is_sample=False):
        """Calculates the standard deviation of a dataset."""
        var = Statistics.variance(data, is_sample)
        return BasicOps.sqrt(var)


class NumberTheory:
    """A collection of functions for number theory investigations."""

    @staticmethod
    def is_prime(n):
        """
        Checks if an integer is a prime number using an efficient trial division.
        """
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def prime_factors(n):
        """Returns a list of the prime factors of an integer."""
        factors = []
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        for i in range(3, int(BasicOps.sqrt(n)) + 1, 2):
            while n % i == 0:
                factors.append(i)
                n //= i
        if n > 2:
            factors.append(n)
        return factors


################################################################################
#
#   Primary Facade Class: Axiom
#
#   This class inherits from all operational sub-modules to provide a single,
#   unified, and easy-to-use interface for all AxiomPy functionality.
#
#   Usage:
#       from operator import Axiom
#       result = Axiom.add(5, 3)
#       pi_val = Axiom.PI
#
################################################################################

class Axiom(
    Constants,
    BasicOps,
    AdvancedOps,
    Trigonometry,
    Calculus,
    LinearAlgebra,
    Statistics,
    NumberTheory
):
    """
    The unified interface for the AxiomPy Mathematics Engine.
    Access all mathematical functions and constants directly through this class.
    
    Example:
    >>> from operator import Axiom
    >>> print(f"PI is approximately {Axiom.PI}")
    >>> print(f"The sine of PI/2 is {Axiom.sin(Axiom.PI / 2)}")
    >>> print(f"The derivative of x^3 at x=2 is {Axiom.differentiate(lambda x: x**3, 2)}")
    """
    pass


# --- Example Usage Block ---
if __name__ == "__main__":
    print("=" * 50)
    print("    AxiomPy Mathematics Engine - Demonstration")
    print("=" * 50)

    # 1. Constants
    print("\n--- 1. Constants ---")
    print(f"PI (from scratch): {Axiom.PI}")
    print(f"Euler's Number (e): {Axiom.E}")
    print(f"Golden Ratio: {Axiom.GOLDEN_RATIO}")

    # 2. Basic and Advanced Operations
    print("\n--- 2. Basic & Advanced Operations ---")
    print(f"Square root of 1024: {Axiom.sqrt(1024)}")
    print(f"Factorial of 10: {Axiom.factorial(10)}")
    print(f"GCD of 54 and 24: {Axiom.gcd(54, 24)}")
    print(f"10th Fibonacci number: {Axiom.fibonacci(10)}")

    # 3. Trigonometry (from scratch)
    print("\n--- 3. Trigonometry ---")
    angle = Axiom.PI / 4  # 45 degrees
    print(f"Sine of PI/4: {Axiom.sin(angle)}")
    print(f"Cosine of PI/4: {Axiom.cos(angle)}")
    print(f"Tangent of PI/4: {Axiom.tan(angle)}")
    print(f"sin^2(x) + cos^2(x) = {Axiom.sin(angle)**2 + Axiom.cos(angle)**2} (should be 1)")

    # 4. Calculus (Numerical Methods)
    print("\n--- 4. Calculus ---")
    # Define a function f(x) = x^3 - 2x^2 + 1
    poly_func = lambda x: x**3 - 2*x**2 + 1
    print(f"Derivative of x^3 - 2x^2 + 1 at x=3: {Axiom.differentiate(poly_func, 3)}")
    print(f"Integral of x^2 from 0 to 1: {Axiom.integrate(lambda x: x**2, 0, 1)} (Exact is 1/3)")

    # 5. Linear Algebra
    print("\n--- 5. Linear Algebra ---")
    matrix_A = [[4, 7], [2, 6]]
    vector_b = [20, 10]
    print(f"Matrix A: {matrix_A}")
    print(f"Determinant of A: {Axiom.determinant(matrix_A)}")
    print(f"Inverse of A: {Axiom.inverse(matrix_A)}")
    print(f"Solving Ax=b for x where b={vector_b}: {Axiom.solve_system(matrix_A, vector_b)}")
    
    # 6. Statistics
    print("\n--- 6. Statistics ---")
    dataset = [2, 4, 4, 4, 5, 5, 7, 9]
    print(f"Dataset: {dataset}")
    print(f"Mean: {Axiom.mean(dataset)}")
    print(f"Median: {Axiom.median(dataset)}")
    print(f"Standard Deviation (Sample): {Axiom.std_dev(dataset, is_sample=True)}")

    # 7. Number Theory
    print("\n--- 7. Number Theory ---")
    num_to_test = 29
    print(f"Is {num_to_test} a prime number? {Axiom.is_prime(num_to_test)}")
    num_to_factor = 990
    print(f"Prime factors of {num_to_factor}: {Axiom.prime_factors(num_to_factor)}")
    
    print("\n" + "=" * 50)
    print("           Demonstration Complete")
    print("=" * 50)
