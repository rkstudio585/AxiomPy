# AxiomPy: The Python Mathematics & Computation Engine

[![PyPI version](https://img.shields.io/pypi/v/axiompy.svg)](https://pypi.org/project/axiompy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/axiompy.svg)](https://pypi.org/project/axiompy/)

**Author:** RK RIAD & RK STUDIO 585  
**GitHub:** [rkstudio585](https://github.com/rkstudio585)

---

## Overview

**AxiomPy** is a powerful and elegant Python mathematics engine designed for both computation and education. Built from first principles, it provides a vast array of mathematical functionalities, from basic arithmetic to advanced calculus, linear algebra, number theory, and even specialized domains like graph analysis and automatic differentiation.

The core philosophy of AxiomPy is to expose the inner workings of mathematical algorithms. By implementing them from the ground up, it serves not only as a high-performance tool for scientists and developers but also as a valuable educational resource for students and enthusiasts who wish to explore the beauty of mathematics. Its clean, object-oriented design and unified API make complex computations intuitive and straightforward.

## Features

- **First-Principle Implementations:** Understand the core logic behind the algorithms you use.
- **Comprehensive Functionality:** Covers a wide spectrum of mathematical fields:
  - Linear Algebra (Vectors, Matrices, Decompositions)
  - Calculus (Numerical Differentiation & Integration)
  - Number Theory (Primes, CRT)
  - Statistics (Mean, Median, Variance)
  - Graph Theory (PageRank Algorithm)
  - Automatic Differentiation Engine
- **Object-Oriented Design:** A clean, organized, and extensible codebase.
- **Unified API:** A single, easy-to-use `Axiom` class provides access to all features.
- **Advanced Data Types:** Includes first-class `Vector` and `Matrix` objects with full operator overloading for intuitive operations.
- **Zero Dependencies (almost):** Relies only on `numpy` for high-performance array operations, with all mathematical logic built from scratch.
- **ASCII Visualizations:** Plot functions and vector fields directly in your terminal.

## Installation

You can install AxiomPy directly from PyPI:

```bash
pip install axiompy
```

## Usage

Here’s a quick look at how you can use AxiomPy:

```python
from axiompy import Axiom

# --- 1. Intuitive Linear Algebra ---
M = Axiom.Matrix([[1, 2], [3, 4]])
v = Axiom.Vector([5, 6])

# Perform matrix-vector multiplication
Mv = M @ v
print(f"Matrix-vector product: {Mv}")

# Get the determinant
print(f"Determinant of M: {M.determinant}")


# --- 2. Graph Analysis with PageRank ---
g = Axiom.Graph()
g.add_edge('A', 'B'); g.add_edge('A', 'C'); g.add_edge('B', 'C');
g.add_edge('C', 'A'); g.add_edge('D', 'C');

ranks = Axiom.graph_analysis.pagerank(g)
print("\nPageRank Scores:")
for node, rank in ranks.items():
    print(f"  Node '{node}': {rank:.4f}")


# --- 3. ASCII Plotting ---
x_vals = [i * 0.4 for i in range(20)]
y_vals = [val**2 for val in x_vals]

print("\nPlotting y = x^2:")
Axiom.viz.plot_ascii(x_vals, y_vals)
```

## Contributing

Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, please open an issue or submit a pull request on our [GitHub repository](https://github.com/rkstudio585/AxiomPy).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.