
from axiompy import Axiom

def run_demonstrations():
    """Runs a series of demonstrations for the AxiomPy library."""
    print("=" * 80)
    print("    AxiomPy Mathematics Engine v3.0.0 - Demonstration")
    print("=" * 80)

    # 1. Matrix and Vector Operations
    print("\n--- 1. Intuitive Matrix and Vector Operations ---")
    M = Axiom.Matrix([[1, 2], [3, 4]])
    v = Axiom.Vector([5, 6])
    I = Axiom.linalg.identity(2)
    print(f"Matrix M:\n{M}")
    print(f"Vector v: {v}")
    Mv = M @ v
    print(f"M @ v = {Mv}")
    M_sq = M @ M
    print(f"M @ M:\n{M_sq}")
    print(f"M ** 2:\n{M**2}")
    print(f"M + I:\n{M + I}")

    # 2. Graph Analysis: PageRank
    print("\n--- 2. Graph Analysis: PageRank Algorithm ---")
    g = Axiom.Graph()
    g.adj['A']; g.adj['B']; g.adj['C']; g.adj['D']
    g.add_edge('A', 'B'); g.add_edge('A', 'C'); g.add_edge('B', 'C');
    g.add_edge('C', 'A'); g.add_edge('D', 'C');
    ranks = Axiom.graph_analysis.pagerank(g)
    print("Calculated PageRank for a simple web graph:")
    for node, rank in sorted(ranks.items(), key=lambda item: item[1], reverse=True):
        print(f"  Node '{node}': Rank = {rank:.4f}")

    # 3. Number Theory: Chinese Remainder Theorem
    print("\n--- 3. Advanced Number Theory: Chinese Remainder Theorem ---")
    n = [3, 5, 7]
    a = [2, 3, 2]
    solution = Axiom.number_theory.chinese_remainder_theorem(n, a)
    print(f"Solving system of congruences: x = a_i (mod n_i)")
    print(f"  n={n}, a={a}")
    print(f"  Solution: {solution} (e.g., 23 % 3 = 2, 23 % 5 = 3, 23 % 7 = 2)")

    # 4. ASCII Visualization
    print("\n--- 4. ASCII Visualization Engine ---")
    x = [i * 0.4 for i in range(20)]
    y = [val**2 for val in x]
    print("Plotting y = x^2:")
    Axiom.viz.plot_ascii(x, y)

    print("\n" + "=" * 80)
    print("                          Demonstration Complete")
    print("=" * 80)

if __name__ == "__main__":
    run_demonstrations()
