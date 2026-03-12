'''
Question 2 - Discrete and continuous
Hongze Lin
'''
# ==================== Q2(a) ====================
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.interpolate import BarycentricInterpolator


def cheb_D1(N: int):
    """
    Chebyshev first-derivative matrix on Chebyshev–Gauss–Lobatto nodes.
    Returns:
      x: nodes in [-1,1] (length N)
      D: NxN first-derivative matrix
    """
    if N < 2:
        raise ValueError("N must be >= 2")

    k = np.arange(N)
    x = np.cos(np.pi * k / (N - 1))  # CGL nodes, x[0]=1, x[-1]=-1

    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** k)

    X = np.tile(x, (N, 1))
    dX = X - X.T

    D = np.outer(c, 1 / c) / (dX + np.eye(N))  # off-diagonal correct, diagonal temp
    D = D - np.diag(np.sum(D, axis=1))         # enforce row sums = 0 -> correct diagonal

    return x, D


def solve_bvp_cheb(n: int):
    """
    Solve:
      s^2 y'' + s y' + (64 s^2 - 1) y = 0,  s in [-1,1]
      y(-1) = -1, y(1) = 1
    via Chebyshev collocation with n nodes (CGL).
    """
    s, D = cheb_D1(n)
    D2 = D @ D

    # Build operator matrix A
    A = np.diag(s**2) @ D2 + np.diag(s) @ D + np.diag(64 * s**2 - 1.0)
    b = np.zeros(n)

    # Enforce boundary conditions by replacing rows:
    # Note: s[0]=+1, s[-1]=-1 because CGL ordering.
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 1.0      # y(1)=1  -> x(8)=1

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = -1.0    # y(-1)=-1 -> x(-8)=-1

    y = np.linalg.solve(A, b)

    # Map back to t = 8s, x(t)=y(s)
    t = 8.0 * s
    x = y.copy()
    return t, x, s


def x_exact(t):
    return jv(1, t) / jv(1, 8.0)


def max_error_on_dense_grid(t_nodes, x_nodes, m=4001):
    """
    Use barycentric interpolation from Chebyshev nodes to a dense uniform grid in [-8,8],
    then compute max error vs exact.
    """
    # BarycentricInterpolator expects x sorted increasing for best behavior.
    idx = np.argsort(t_nodes)
    t_sorted = t_nodes[idx]
    x_sorted = x_nodes[idx]

    interp = BarycentricInterpolator(t_sorted, x_sorted)

    t_dense = np.linspace(-8.0, 8.0, m)
    x_num = interp(t_dense)
    err = np.max(np.abs(x_num - x_exact(t_dense)))
    return err


def main():
    # Scan n values
    n_list = list(range(8, 81))  # you can widen if you want
    errs = []

    for n in n_list:
        t_nodes, x_nodes, _ = solve_bvp_cheb(n)
        err = max_error_on_dense_grid(t_nodes, x_nodes, m=5001)
        errs.append(err)

    errs = np.array(errs)

    # Find best n (minimum error)
    i_best = int(np.argmin(errs))
    n_best = n_list[i_best]
    err_best = float(errs[i_best])

    # Find "no longer improves" point:
    # A practical definition: first n after which improvement is < 5% of best (or within factor 1.05).
    # You can adjust tolerance.
    tol_factor = 1.05
    plateau_candidates = np.where(errs <= tol_factor * err_best)[0]
    n_plateau = n_list[int(plateau_candidates[0])] if plateau_candidates.size else n_best

    print(f"Best (min) error occurs at n = {n_best}, max error ≈ {err_best:.3e}")
    print(f"Plateau begins around n ≈ {n_plateau}  (within {tol_factor:.2f}× of best error)")

    # Plot convergence
    plt.figure()
    plt.semilogy(n_list, errs, marker='o', markersize=3, linewidth=1)
    plt.axvline(n_best, linestyle='--')
    plt.title("Chebyshev collocation convergence: max error vs n")
    plt.xlabel("Number of collocation points n")
    plt.ylabel("max |x_num - x_exact| on dense grid")
    plt.grid(True, which='both')

    # Plot best solution vs exact
    t_nodes, x_nodes, _ = solve_bvp_cheb(n_best)
    idx = np.argsort(t_nodes)
    interp = BarycentricInterpolator(t_nodes[idx], x_nodes[idx])

    t_dense = np.linspace(-8.0, 8.0, 2001)
    x_num = interp(t_dense)
    x_ex = x_exact(t_dense)

    plt.figure()
    plt.plot(t_dense, x_ex, linewidth=2, label="exact  J1(t)/J1(8)")
    plt.plot(t_dense, x_num, linestyle='--', linewidth=2, label=f"Cheb collocation (n={n_best})")
    plt.title("Best numerical solution vs exact")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.grid(True)
    plt.legend()

    # Optional: plot pointwise error for best n
    plt.figure()
    plt.plot(t_dense, x_num - x_ex, linewidth=2)
    plt.title(f"Pointwise error for n={n_best}")
    plt.xlabel("t")
    plt.ylabel("x_num(t) - x_exact(t)")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()