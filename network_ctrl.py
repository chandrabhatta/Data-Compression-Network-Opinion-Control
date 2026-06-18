import numpy as np
from numpy.linalg import matrix_rank, eig
import matplotlib.pyplot as plt
import scipy.io as sio
import cvxpy as cp
from sklearn.linear_model import OrthogonalMatchingPursuit


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

    types = ['L1', 'L2', 'LASSO', 'RIDGE']
    if obj_type not in types:
        print(f"Please select from: {types}")
    elif obj_type == 'L1':
        return cp.Minimize(cp.norm1(U))
    elif obj_type == 'L2':
        return cp.Minimize(cp.norm2(H @ U - b))
    elif obj_type == 'LASSO':
        return cp.Minimize(cp.norm2(H @ U - b) + mu * cp.norm1(U))
    elif obj_type == 'RIDGE':
        return cp.Minimize(cp.norm2(H @ U - b) + mu * cp.norm2(U))


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


def matching_persuit(H, b, max_iter=100):
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
        print(f'res={res}')
        k = np.argmax(np.abs(correlations))  # select the highest correlated index
        print(f'k={k}')
        supp.append(k)  # store the index as the support

        # subtract the residue (for MP)
        coeff = correlations[k]
        print(f'coeff={coeff}')
        u_supp.append(coeff)

        # H \in [25, 625]
        res = res - coeff * H[:, k]  # [25,1] - [1,] * [25,1]

        if np.linalg.norm(res) < 1e-6: break  # optional convergence criteria
        if i == max_iter: print(f"MP did not converge in {max_iter} iterations.")

    # recover the full u vector
    u = np.zeros(N)
    for idx, coeff in zip(supp, u_supp): u[idx] = coeff

    return u, supp

def orthogonal_matching_persuit(H, b, s):
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
        print(f'u_supp={u_supp}')

        res = b - H_selected @ u_supp

        # if np.linalg.norm(res) < 1e-6: break # optional convergence criteria

    # recover the full u vector
    u = np.zeros(N)
    print(f'supp={supp}')
    for idx, coeff in zip(supp, u_supp): u[idx] = coeff # The final u_LS contains the LS estimates over the full support of u
    print(f'u={u}')
    return u, supp

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

def piecewise_OMP(H, b, s_per_block, B, max_iter=100, tol=1e-6):
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
    if isinstance(s_per_block, (int, float)):
        s_per_block = [int(s_per_block)] * B
    else:
        # Ensure it's a list and convert all entries to int
        s_per_block = [int(s) for s in s_per_block]

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

    return u_new, supp#, residual_history

def optimize_input(A, x0, xf, s, K, B=1, u_max=None, method='L1'):
    """
    Solver for convex norm minimization with affine constraint using CVXpy: argmin_u { ||u||_0 } s.t. Ax=u,
    The goal is to recover an s-sparse input vector 'u' given a linear constraint and desired final state vector 'xf'.

    inputs:
     - 'A'      = any square and non-singular matrix
     - 'x0'     = initial state vector
     - 'xf'     = desired final state vector
     - 's'      = sparsity level in 'u_opt'
     - 'K'      = time horizon (number of control inputs)
     - 'u_max'  = maximum allowed input --> could also change for max energy (uu')_max
     = 'method' = selects recovery method for U, choose from {'L1', 'L2', 'LASSO', 'RIDGE', 'MP', 'OMP', 'POMP', 'COSAMP'}
    outputs:
     - 'U'      = optimal s-sparse vector
    """
    np.random.seed(1)  # set seed for reproducability
    N = A.shape[0]  # extract dimensions from input

    # Build controllability matrix --> TODO: can also update H recursively to save memory and computation time
    H_list = [np.linalg.matrix_power(A, K - i - 1) for i in range(K)]
    print(f'H_list= [{np.shape(H_list)}]')
    H = np.hstack(H_list)
    print(f'H= [{np.shape(H)}]')
    b = xf - np.linalg.matrix_power(A, K) @ x0
    print(f'b=[{b.shape}]')

    #H,norms = H_normalizer(H) # We column-normalize H to equalize the horizon (since it tends to give higher values the lower the index)

    if method in ('L1', 'L2', 'LASSO', 'RIDGE'):
    # Standard Basis Persuit
        U = cp.Variable(N * K) # U is a stacked vector of size [N x K]
        objective = obj_selector(method, H, U, b, mu=0.1) # Get respective objective function from wrapper (using entrywise norms)

        # as constraints we need N<=s+rank(A)=s+N (always satisfied), and controllability (always satisfied for B=I): Hx=u, can also set constraints on u
        constraints = [H @ U == b]

        if u_max is not None:
            constraints.extend([
                U <= u_max,
                U >= -u_max
            ])

        problem = cp.Problem(objective, constraints)
        opt_cost = problem.solve(solver=cp.SCS) # The optimal objective value is returned by `prob.solve()`.

        U = U.value # The optimal value for u is stored in `u.value`.

    elif method == 'MP':
        U,supp = matching_persuit(H, b)[0]
    elif method == 'OMP_ref':
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(10, K))
        omp.fit(H.T, b)
        U,supp = omp.coef_
    elif method == 'OMP':
        U,supp = orthogonal_matching_persuit(H, b, s)[0]
    elif method == 'POMP':
        #s_per_block = s # Spread sparsity uniformly over the blocks --> can be any distribution
        U,supp = piecewise_OMP(H, b, s, B)
    elif method == 'COSAMP':
        U,supp = None
    return U,supp #/norms.ravel()  # .reshape(N, K, order='F')


def main():
    mat_data = load_mat('PiecewiseSparse.mat')
    A = mat_data['A']
    Xf = mat_data[
        'FinalState']  # First off, note that A contains self loops (nonzero diag.) and both positive and negative values (directional)

    #A = row_normalizer(A)

    N = np.size(A, 0)
    test_controllability(A, np.eye(N))

    x0 = np.random.randn(25, 1)

    print(Xf.shape[0])

    # for i in range(Xf.shape[0]):
    xf = Xf[5, :].reshape(25, 1)
    s = 10
    K = 25
    B = K
    u,supp = optimize_input(A, x0, xf, s, K, B=B, method='POMP')
    print(f'u={u}\n')
    print(f'sparsity={len(supp)}')

if __name__ == '__main__':
    main()