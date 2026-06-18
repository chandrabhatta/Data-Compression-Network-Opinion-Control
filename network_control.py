import numpy as np
from numpy.linalg import matrix_rank, eig
import matplotlib.pyplot as plt
import scipy.io as sio
import cvxpy as cp
from sklearn.linear_model import OrthogonalMatchingPursuit
import time


def load_mat(filename):
    """
    Function that extracts data from '.mat'-files into a python dictionary
    The data columns can be accessed as such: 'variableName' = mat_data['variableName']
    """
    if filename.endswith('.mat'):
        mat_data = sio.loadmat(filename)
        print(f"Your .mat file contains the following variables: {mat_data.keys()}")
        return mat_data
    else:
        print(f"Please enter a filename ending with .mat")


def test_controllability(A, B):
    """
    Check Hautus criteria given A and B matrices to verify controllability

    :param A: any [N x N] matrix
    :param B: any [N x N] matrix
    :return: 'True' or 'False' based on whether the Hautus criteria is satisfied
    """
    #
    if np.size(A) != np.size(B) or np.size(A, 0) != np.size(A, 1) or np.size(B, 0) != np.size(B, 1):
        print("Please use square matrices A and B")

    N = np.size(A, 0)
    rank_A = matrix_rank(A)
    eig_A = eig(A).eigenvalues
    print(f"Shape of A = {A.shape}")
    print(f"Shape of eig_A = {eig_A.shape}")
    print(f"Rank A = {rank_A}")

    all_controllable = all(matrix_rank(np.hstack([A - λ * np.eye(N), B])) == N for λ in eig_A)
    print(
        f"\nOverall system is controllable: {all_controllable}")  # Always true when B = I --> can add extra check for this to save computations


def obj_selector(obj_type, H, U, b, mu=0.1):
    """
    Wrapper for objective function for the linear model: H U = b

    :param obj_type: name of the respective objective function, choose from: { 'L1', 'L2', 'LASSO', 'RIDGE' }
    :param mu: regularization parameter
    :return: respective objective function
    """

    types = ['L1', 'L2', 'LASSO', 'RIDGE', 'ELASTIC']
    if obj_type not in types:
        print(f"Please select from: {types}")
    elif obj_type == 'L1':
        # Basis Pursuit: min ||U||_1 s.t. HU = b
        return cp.Minimize(cp.norm1(U))

    elif obj_type == 'L2':
        # Least squares
        return cp.Minimize(cp.sum_squares(H @ U - b))

    elif obj_type == 'LASSO':
        # LASSO: unconstrained, trades data fit vs sparsity via mu
        return cp.Minimize(cp.sum_squares(H @ U - b) + mu * cp.norm1(U))

    elif obj_type == 'RIDGE':
        # Ridge: unconstrained, trades data fit vs energy via mu
        return cp.Minimize(cp.sum_squares(H @ U - b) + mu * cp.sum_squares(U))

    elif obj_type == 'ELASTIC':
        # Elastic Net: sparsity via L1 + stability via L2
        mu1 = mu  # L1 weight (sparsity)
        mu2 = mu * 0.5  # L2 weight (grouping/stability), tune ratio as needed
        return cp.Minimize(cp.sum_squares(H @ U - b) + mu1 * cp.norm1(U) + mu2 * cp.sum_squares(U))

def row_normalizer(A, type="stoch_avg"):
    """
    Function that preprocesses a matrix A such that it satisfied certain normalization properties

    :param A: any [N x N] matrix
    :param type: normalization scheme, choose from: 'row_avg', 'stoch_row'
    :return: A_bar = normalized A
    """
    N = A.shape[0]
    A_bar = np.zeros([N,N])
    types = ["row_avg", "stoch_row"]
    if type not in types: print(f"Please select from: {types}")

    elif type == "row_avg": # ensures row mean = 1
        for i in range(N):
            row_avg = sum(A[i][:]) / N
            A_bar[i][:] = A_bar[i][:] / row_avg
    elif type == "stoch_row": # ensures rows sum to 1 (bounding)
        row_sums = A.sum(axis=1, keepdims=True)
        A_bar = A / np.where(row_sums == 0, 1.0, row_sums)

    return A_bar

def H_normalizer(H, type="column"):
    norms = np.linalg.norm(H, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    H_bar = H / norms

    return H_bar, norms

def check_conv(A, x0, xf, U):
    """
    Verify that control achieves desired final state
    """
    n, N = U.shape
    x = x0.copy()

    for k in range(N):
        x = A @ x + U[:, k]

    err = np.linalg.norm(x - xf)
    print(f"Error: {err:.2e}")
    return err

def matching_pursuit(H, b, max_iter=1000):
    """
    Implementation of MP that recovers s-sparse vector u for the model: H u = b

    Parameters:
     - H    = Dictionary matrix (m x n), each column is an atom
     - b    = Measurement vector (m,)
     - s    = Sparsity level (number of non-zero entries)

    outputs:
     - u        = Recovered sparse vector (n,)
     - supp     = Indices of non-zero entries in u
    """
    #b = np.squeeze(b) # (N,1) --> (N,) for size=1 only
    b = b.ravel() # (N,M) --> (NxM,) for any size
    N = H.shape[1]
    res = b.copy()  # residual
    supp = []  # support(x)
    u_supp = []  # x value at support index

    for i in range(max_iter):  # Need s iterations for s-sparse recovery generally

        correlations = H.T @ res  # inner-product or cols H with residual <H_i,r>
        k = np.argmax(np.abs(correlations))  # select the highest correlated index
        supp.append(k)  # store the index as the support

        # subtract the residue (for MP)
        coeff = correlations[k]
        u_supp.append(coeff)

        # H \in [25, 625]
        res = res - coeff * H[:, k]  # [25,1] - [1,] * [25,1]

        if np.linalg.norm(res) < 1e-6: break  # optional convergence criteria
        if i == max_iter: print(f"MP did not converge in {max_iter} iterations.")

    # recover the full u vector
    u = np.zeros(N)
    for idx, coeff in zip(supp, u_supp): u[idx] = coeff

    return u, supp, res

def orthogonal_matching_pursuit(H, b, s):
    """
    Implementation of OMP that recovers s-sparse vector u for the model: H u = b

    Parameters:
     - H    = Dictionary matrix (m x n), each column is an atom
     - b    = Measurement vector (m,)
     - s    = Sparsity level (number of non-zero entries)

    outputs:
     - u        = Recovered sparse vector (n,)
     - supp     = Indices of non-zero entries in u
    """
    #b = np.squeeze(b)
    b = b.ravel()
    N = H.shape[1]
    res = b.copy()  # residual
    supp = []       # support(x)
    u_supp = None

    for _ in range(s):  # Need s iterations for s-sparse recovery generally

        correlations = H.T @ res  # inner-product or cols H with residual <H_i,r>
        k = np.argmax(np.abs(correlations))  # select the highest correlated index
        supp.append(k)  # store the index as the support

        # solve least square, orthogonal projection (for OMP)
        H_selected = H[:, supp]
        u_supp = np.linalg.pinv(H_selected) @ b  # LS on the support
        # u_supp = np.linalg.lstsq(H[:, supp], b, rcond=None)[0] # can use lstsq function alternatively

        res = b - H_selected @ u_supp

        # if np.linalg.norm(res) < 1e-6: break # optional convergence criteria

    # recover the full u vector
    u = np.zeros(N)
    for idx, coeff in zip(supp, u_supp): u[idx] = coeff # The final u_LS contains the LS estimates over the full support of u
    return u, supp, res

# def block_OMP(H, b, s, B):
#     """
#     Implementation of block-OMP that recovers s-sparse vector u for the model: H u = b
#
#     Parameters:
#      - H    = Dictionary matrix (m x n), each column is an atom [Per block]
#      - b    = Measurement vector (m,)
#      - s    = Sparsity level (number of non-zero entries)
#      - B    = #blocks
#
#     outputs:
#      - u        = Recovered sparse vector (n,)
#      - supp     = Indices of non-zero entries in u
#     """
#     #b = np.squeeze(b)
#     b = b.ravel()
#     N = H.shape[1]
#     res = b.copy()  # residual
#     supp = []       # support(x)
#     u_supp = None
#
#     blocklength = N/B
#     # split H into pieces: H = [H1, H2, ..., H_B]
#     for _ in range(s):  # Need s iterations for s-sparse recovery generally
#         block_supports = []  # Support for this iteration only
#
#         # Step 1: Find best atom in each block
#         for block_idx in range(B):
#             # Get columns for current block
#             start_col = block_idx * blocklength
#             end_col = (block_idx + 1) * blocklength
#             H_block = H[:, start_col:end_col]
#
#             # Find best atom in this block
#             correlations = H_block.T @ res
#             best_in_block = np.argmax(np.abs(correlations))
#             global_idx = start_col + best_in_block
#
#             block_supports.append(global_idx)
#
#         # MERGE supports from all blocks
#         supp.extend(block_supports)  # Add all B atoms to global support
#
#         # Solve least squares on merged support
#         H_selected = H[:, supp]
#         u_supp = np.linalg.lstsq(H_selected, b, rcond=None)[0]
#         res = b - H_selected @ u_supp
#
#         # if np.linalg.norm(res) < 1e-6: break # optional convergence criteria
#
#     # recover the full u vector
#     u = np.zeros(N)
#     print(f'supp={supp}')
#     for idx, coeff in zip(supp, u_supp): u[idx] = coeff # The final u_LS contains the LS estimates over the full support of u
#     print(f'u={u}')
#     return u, supp

def piecewise_OMP(H, b, s_per_block, B, max_iter=1000, tol=1e-6):
    """
    Piecewise OMP (P-OMP) Recovery Algorithm.

    Recovers a piecewise sparse vector u from H u = b, where u is partitioned
    into B blocks, each with sparsity s_per_block[i].

    Parameters:
     - H              : Dictionary matrix (m x n)
     - b              : Measurement vector (m,)
     - s_per_block    : List or array of length B, where s_per_block[i] is
                        the sparsity level for block i.
     - B              : Number of blocks (time steps)
     - max_iter       : Maximum number of iterations
     - tol            : Stopping tolerance (residual norm)

    Outputs:
     - u              : Recovered piecewise sparse vector (n,)
     - supp           : Final support set (indices of non-zero entries)
     - residual_history : List of residual norms per iteration
    """

    # Ensure b is a column vector
    b = b.ravel()
    N = H.shape[1]  # Total number of unknowns
    block_length = N // B  # Length of each block (assumes N is divisible by B)

    # Convert s_per_block to a list if it's a single integer
    if np.isscalar(s_per_block):
        s_per_block = [int(s_per_block)] * B
    else:
        # Ensure it's a list and convert all entries to int
        s_per_block = [int(si) for si in s_per_block]

    # Validate inputs
    if len(s_per_block) != B:
        raise ValueError(f"s_per_block must have length {B}, got {len(s_per_block)}")
    if N % B != 0:
        raise ValueError(f"N={N} must be divisible by B={B}") # TODO --> check nonuniform blocking cases: choose blockscaling or zero-pad

    # Initialization
    u_new = np.zeros(N)
    res = b.copy()
    supp = []
    residual_history = []

    for iteration in range(max_iter):
        # Step 1 & 2: Form piecewise proxy, pick top s_i per block
        Omega = []
        for block_idx in range(B):
            start = block_idx * block_length
            end = start + block_length
            correlations = H[:, start:end].T @ res
            s_i = s_per_block[block_idx]
            if s_i > 0:
                local_indices = np.argsort(np.abs(correlations))[-s_i:]
                Omega.extend((start + local_indices).tolist())

        # Step 3: Merge supports
        merged_supp = list(set(supp + Omega))

        # Step 4: Least-squares on merged support
        u_tilde = np.zeros(N)
        if len(merged_supp) > 0:
            H_sel = H[:, merged_supp]
            coeffs = np.linalg.lstsq(H_sel, b, rcond=None)[0]
            for idx, c in zip(merged_supp, coeffs):
                u_tilde[idx] = c

        # Step 5: Piecewise truncation, keep top s_i per block
        u_new = np.zeros(N)
        for block_idx in range(B):
            start = block_idx * block_length
            end = start + block_length
            block_coeffs = u_tilde[start:end]
            s_i = s_per_block[block_idx]
            if s_i > 0:
                keep = np.argsort(np.abs(block_coeffs))[-s_i:]
                u_new[start + keep] = block_coeffs[keep]

        # Step 6: Update support and residual
        supp = np.where(np.abs(u_new) > 1e-10)[0].tolist()
        res = b - H @ u_new
        residual_norm = np.linalg.norm(res)
        residual_history.append(residual_norm)

        if residual_norm < tol:
            print(f"P-OMP converged after {iteration + 1} iterations "
                  f"(residual = {residual_norm:.2e})")
            break
    else:
        print(f"P-OMP reached max iterations ({max_iter}) with residual = {residual_norm:.2e}")

    return u_new, supp, residual_history

def CoSAMP(H,b,s,max_iter = 1000,tol=1e-6):

    """
    Implementation of CoSAMP for sparse recovery of vector u from Hu = b

    Parameters:
     - H    = Dictionary matrix (m x n), each column is an atom
     - b    = Measurement vector (m,)
     - s    = Sparsity level (number of non-zero entries)
     - max_iter : maximum number of iterations
     -tol : stopping tolerance

    Returns:
        u : recovered sparse vector (n,)
        supp : final support set
    """

    b = b.ravel() # converts b to a 1D vector
    N = H.shape[1]
    res = b.copy()  # residual
    supp = set()      # support(x)
    u_supp = None

    for _ in range(max_iter):

        correlations = H.T @ res
        omega = np.argpartition(np.abs(correlations),-2*s)[-2*s:] #obtain the indices of top 2s  correlations

        # Merge supports:
        T = np.union1d(list(supp), omega).astype(int)

        # take the columns of H corresponding to the support set and calculate z interms of the psudoinv via LS
        H_T = H[:,T]
        z, *_ = np.linalg.lstsq(H_T, b, rcond=None)

        # Keep the top s entries of z
        idx = np.argpartition(np.abs(z),-s)[-s:]
        supp = set(T[idx])

        # Update the estimate
        u = np.zeros(N)
        u[list(supp)] = z[idx] # delete later-> set() converts to a set and list() converts to a list

        res = b - H @ u

        if np.linalg.norm(res) < tol:
            break

    return u, sorted(list(supp)), res

def optimize_input(A, x0, xf, s, K, B=1, mu=None, u_max=None, method='L1'):
    """
    Solver for convex norm minimization with affine constraint using CVXpy: argmin_u { ||u||_0 } s.t. Ax=u,
    The goal is to recover an s-sparse input vector 'u' given a linear constraint and desired final state vector 'xf'.

    inputs:
     - 'A'      = any square and non-singular matrix
     - 'x0'     = initial state vector
     - 'xf'     = desired final state vector
     - 's'      = sparsity level in 'u_opt'
     - 'K'      = time horizon (number of control inputs)
     - 'u_max'  = maximum allowed input
     - 'method' = selects recovery method for U, choose from {'L1', 'L2', 'LASSO', 'RIDGE', 'ELASTIC', 'MP', 'OMP', 'POMP', 'COSAMP'}
    outputs:
     - 'U'      = optimal s-sparse vector (N*K,)
     - 'supp'   = support set (indices of non-zero entries)
     - 'H'      = original (unnormalized) controllability matrix (N, N*K)
     - 'b'      = target vector (N,)
    """
    N = A.shape[0]

    # Build controllability matrix H and target vector b
    H_list = [np.linalg.matrix_power(A, K - i - 1) for i in range(K)]
    H = np.hstack(H_list)                                      # (N, N*K)
    b = (xf - np.linalg.matrix_power(A, K) @ x0).ravel()      # (N,)

    # Column-normalize H for numerical stability
    H_norm, norms = H_normalizer(H)

    supp = []
    U = np.zeros(N * K)

    if method == 'L1':
        # Basis Pursuit: min ||U||_1 s.t. H_norm U = b  (exact sparse recovery)
        U_var = cp.Variable(N * K)
        objective = cp.Minimize(cp.norm1(U_var))
        constraints = [H_norm @ U_var == b]
        if u_max is not None:
            constraints += [U_var <= u_max, U_var >= -u_max]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if U_var.value is None:
            print(f"[{method}] SCS failed (status={problem.status}), trying CLARABEL...")
            problem.solve(solver=cp.CLARABEL)
        if U_var.value is None:
            print(f"[{method}] All solvers failed, returning zeros")
        else:
            U = U_var.value / norms.ravel()
            U = hard_threshold(U, s)  # force s-sparsity
            supp = sorted(np.where(np.abs(U) > 1e-10)[0].tolist())

    elif method in ('L2', 'LASSO', 'RIDGE', 'ELASTIC'):
        if mu is None:
            #mu = 2*K / (s + 1e-8) # s controls sparsity via mu
            mu_max = np.max(np.abs(H_norm.T @ b))
            alpha = 1.0 - s / (N * K)
            mu = mu_max * alpha
            print(f'mu = {mu}')
        # Unconstrained regularized objectives — no equality constraint
        U_var = cp.Variable(N * K)
        objective = obj_selector(method, H_norm, U_var, b, mu=mu)
        constraints = []
        if u_max is not None:
            constraints += [U_var <= u_max, U_var >= -u_max]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if U_var.value is None:
            print(f"[{method}] SCS failed (status={problem.status}), trying CLARABEL...")
            problem.solve(solver=cp.CLARABEL)
        if U_var.value is None:
            print(f"[{method}] All solvers failed, returning zeros")
        else:
            U = U_var.value / norms.ravel()
            U = hard_threshold(U, s)  # force s-sparsity
            supp = sorted(np.where(np.abs(U) > 1e-10)[0].tolist())

    elif method == 'MP':
        U, supp, _ = matching_pursuit(H_norm, b)
        U = U / norms.ravel()

    elif method == 'OMP':
        U, supp, _ = orthogonal_matching_pursuit(H_norm, b, s)
        U = U / norms.ravel()

    elif method == 'OMP_ref':
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(s, K))
        omp.fit(H_norm, b)
        U = omp.coef_ / norms.ravel()

    elif method == 'POMP':
        U, supp, _ = piecewise_OMP(H_norm, b, s, B=K)
        U = U / norms.ravel()

    elif method == 'COSAMP':
        U, supp, _ = CoSAMP(H_norm,b,s)
        U = U / norms.ravel()

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'L1', 'L2', 'LASSO', 'RIDGE', 'ELASTIC', 'MP', 'OMP', 'POMP', 'COSAMP'")

    # Verify reconstruction using original unnormalized H
    residual = np.linalg.norm(b - H @ U)
    print(f"[{method}] Reconstruction error: {residual:.6e} | Sparsity: {np.sum(np.abs(U) > 1e-10)}/{N*K}")

    return U, supp, H, b

def adj_matrix(N, sym=True):

    A = np.random.randn(N, N) # values between [0, 1]
    if sym: A = 0.5*(A.T + A )

    return A

def plot_results(sparsity_list, results, metric):
    plt.figure()

    for method, data in results.items():
        plt.plot(sparsity_list, data[metric], marker='o', label=method)

    plt.xlabel("Sparsity s")
    plt.ylabel(metric)
    # plt.title(f"{metric} vs Sparsity")
    plt.grid(True)
    plt.legend()
    plt.show()

def benchmark_all_metrics(
    A, x0, xf,
    K,
    sparsity_list,
    methods,
    B=1
):
    """
    Computes:
        - reconstruction error
        - input energy (||U||_2)
        - runtime

    for multiple sparsities and methods.
    """

    N = A.shape[0]

    # Build system once
    H = np.hstack([
        np.linalg.matrix_power(A, K - i - 1)
        for i in range(K)
    ])

    x_free = np.linalg.matrix_power(A, K) @ x0

    results = {
        m: {
            "error": [],
            "energy": [],
            "time": []
        } for m in methods
    }

    for s in sparsity_list:

        for method in methods:

            start = time.perf_counter()

            out = optimize_input(A, x0, xf, s, K, B=B, method=method)

            elapsed = time.perf_counter() - start

            if isinstance(out, tuple):
                U = out[0]
            else:
                U = out

            if U is None:
                results[method]["error"].append(np.nan)
                results[method]["energy"].append(np.nan)
                results[method]["time"].append(np.nan)
                continue

            U = U.reshape(-1, 1)

            # Reconstruction
            xf_hat = x_free + H @ U

            err = np.linalg.norm(xf - xf_hat)

            energy = np.linalg.norm(U)

            results[method]["error"].append(err)
            results[method]["energy"].append(energy)
            results[method]["time"].append(elapsed)

        print(f"[OK] sparsity {s}")

    return results

def hard_threshold(U, s):
    """Keep only the s largest entries by magnitude, zero out the rest."""
    threshold_idx = np.argsort(np.abs(U))[-s:]  # indices of s largest
    U_thresh = np.zeros_like(U)
    U_thresh[threshold_idx] = U[threshold_idx]
    return U_thresh

def generate_sparsity_list(N):
    """
    Generate a list of sparsity values s such that:
    1. 1 <= s <= N (sparsity per time step cannot exceed number of people)
    2. Values are roughly in the desired ranges
    3. Provides a good spread for the energy vs sparsity tradeoff

    Parameters:
     - N : Number of people (rows of H)
     - K : Number of time steps (blocks)

    Returns:
     - sparsity_list : Sorted list of valid s values
    """
    # Desired approximate values (s is per time step, so max is N)
    desired_ratios = [1 / 20, 1 / 10, 1 / 5, 1 / 4, 1 / 2, 1]
    desired_values = [int(ratio * N) for ratio in desired_ratios]

    # Clamp to [1, N] and ensure uniqueness
    sparsity_list = []
    for val in desired_values:
        val = max(1, min(val, N))
        if val not in sparsity_list:
            sparsity_list.append(val)

    # Ensure we have the extremes
    if 1 not in sparsity_list:
        sparsity_list.insert(0, 1)
    if N not in sparsity_list:
        sparsity_list.append(N)

    return sorted(sparsity_list)

def generate_symmetric_stable_A(N, lambda_min=0.1, lambda_max=0.9):
    """
    Generate a symmetric N x N matrix with eigenvalues in [lambda_min, lambda_max].

    This creates an undirected network where influence is mutual.

    Parameters:
     - N           : Size of the network
     - lambda_min  : Minimum eigenvalue (must be > 0 for stability)
     - lambda_max  : Maximum eigenvalue (must be < 1 for stability)

    Returns:
     - A           : Symmetric N x N matrix with eigenvalues in (0,1)
    """
    # Step 1: Generate a random symmetric matrix
    B = np.random.randn(N, N)
    A_random = (B + B.T) / 2  # Symmetrize

    # Step 2: Eigendecomposition
    eigvals, V = np.linalg.eigh(A_random)

    # Step 3: Scale eigenvalues to lie in [lambda_min, lambda_max]
    # Map the current eigenvalues to the desired range
    eigvals_min = np.min(eigvals)
    eigvals_max = np.max(eigvals)

    if eigvals_max - eigvals_min < 1e-12:
        # All eigenvalues are equal (unlikely for random matrices)
        eigvals_scaled = np.ones(N) * (lambda_min + lambda_max) / 2
    else:
        # Linear scaling to [lambda_min, lambda_max]
        eigvals_scaled = lambda_min + (lambda_max - lambda_min) * (eigvals - eigvals_min) / (eigvals_max - eigvals_min)

    # Step 4: Reconstruct the matrix
    A = V @ np.diag(eigvals_scaled) @ V.T

    return A

def plot_heatmap(matrix, title="Heatmap"):
    """
    Plots a heatmap of the given 2D matrix with values displayed in each cell.

    Parameters:
        matrix (list[list[float]] or np.ndarray): 2D data to plot.
        title (str): Title of the heatmap.
    """
    # Convert to NumPy array for safety
    data = np.array(matrix)

    # Validate input
    if data.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap="viridis", aspect="auto")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")

    # Show all ticks and label them
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(range(data.shape[1]))
    ax.set_yticklabels(range(data.shape[0]))

    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()

def scale_eigenvalues(A, lambda_min=0.1, lambda_max=0.9):
    """
    Scale the eigenvalues of A to lie in [lambda_min, lambda_max].
    Preserves the eigenvectors (network modes) of A.
    """
    # Step 1: Eigendecomposition
    eigvals, V = np.linalg.eig(A)
    V_inv = np.linalg.inv(V)

    # Step 2: Scale eigenvalues
    eigvals_real = np.real(eigvals)
    eigvals_min = np.min(eigvals_real)
    eigvals_max = np.max(eigvals_real)

    if eigvals_max - eigvals_min < 1e-12:
        eigvals_scaled = np.ones_like(eigvals) * (lambda_min + lambda_max) / 2
    else:
        eigvals_scaled = lambda_min + (lambda_max - lambda_min) * (eigvals_real - eigvals_min) / (
                    eigvals_max - eigvals_min)

    # Step 3: Reconstruct
    A_new = V @ np.diag(eigvals_scaled) @ V_inv

    # Ensure real (discard tiny imaginary parts)
    A_new = np.real(A_new)

    return A_new

def random_stable_A(N, rho_target=0.9, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    # Normalize by spectral radius, then scale to desired rho
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    A = A / rho * rho_target
    return A

def test_RIP(H, sparsity_list):
    # Normalize columns of H
    H_norms = np.linalg.norm(H, axis=0, keepdims=True)  # (1, NK)
    H_norm = H / H_norms  # normalized columns

    # Mutual coherence: max |h_i^T h_j| for i != j
    G = np.abs(H_norm.T @ H_norm)  # (NK, NK) Gram matrix
    np.fill_diagonal(G, 0)
    mu = G.max()

    print(f"Mutual coherence mu(H) = {mu:.4f}")
    print()
    for s in sparsity_list:
        delta_s = (s - 1) * mu
        print(f"s={s}: RIP proxy delta_{s} = (s-1)*mu = {delta_s:.4f}  ->  RIP {'likely holds' if delta_s < 1 else 'may FAIL'}")

    return G, mu, delta_s

def main():
    mat_data = load_mat('PiecewiseSparse.mat')
    A = mat_data['A']
    Xf = mat_data[
        'FinalState']  # First off, note that A contains self loops (nonzero diag.) and both positive and negative values (directional)

    #A = row_normalizer(A)

    N = np.size(A, 0)
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    #A = A / rho * 0.99
    #A = random_stable_A(N)#
    #A = scale_eigenvalues(A)
    #print(f'A = {A}\n')
    #plot_heatmap(A)

    test_controllability(A, np.eye(N))
    #eigenvalues, eigenvectors = eig(A)
    #print(f'eigval: {np.abs(eigenvalues)}')
    x0 = np.random.randn(25, 1)
    print(f'x0 = {x0.shape}\n')

    # for i in range(Xf.shape[0]):
    xf = Xf[2, :].reshape(25, 1)
    s = 20
    K = 1
    B = K

    method_list_opt = ['L1', 'L2', 'LASSO', 'RIDGE', 'ELASTIC','MP', 'OMP', 'POMP']
    sparsity_list = np.arange(1, 6, dtype=int)
    u, supp, H, b = optimize_input(A, x0, xf, s, K, B=B, method='POMP')
    test_RIP(H, sparsity_list)
    ### --- TESTING --- ####
    # for i in range(Xf.shape[0]):
    #     xf = Xf[i, :].reshape(N, 1)
    #     print(f"\n===== Target state {i + 1}/{Xf.shape[0]} =====")
    #
    #     for method in method_list_opt:
    #         for s in sparsity_list:
    #             print(f"\n--- {method} | s={s} ---")
    #             t0 = time.time()
    #             u, supp, H, b = optimize_input(A, x0, xf, s, K, B=B, method=method)
    #             tf = time.time()
    #
    #             print(f"Time:                 {tf - t0:.3f}s")
    #             print(f"Sparsity:             {len(supp)}/{N * K}")
    #             print(f"Reconstruction error: {np.linalg.norm(b - H @ u):.6e}")
    #             print(f"Control energy:       {np.linalg.norm(u):.6e}")

    #xf_hat = np.linalg.matrix_power(A, K) @ x0 + H @ u
    # these two should be identical:
    # print(f"Reconstruction error: {np.linalg.norm(xf - xf_hat):.6e}")

    sparsity_list = generate_sparsity_list(N)
    #[K/20, K/10, K/5, K/4, K/2, K, 2*K, 3*K]

    # sparsity_list = np.arange(1, 250,10, dtype=int)
    # print(f'sparsity list: {sparsity_list}\n')
    # method_list_greedy = ['MP', 'OMP', 'POMP']
    # results = benchmark_all_metrics(A, x0, xf, K=K, sparsity_list=sparsity_list,
    #                                             methods=method_list_opt)
    #
    # plot_results(sparsity_list, results, "error")
    # plot_results(sparsity_list, results, "energy")
    # plot_results(sparsity_list, results, "time")

if __name__ == '__main__':
    main()