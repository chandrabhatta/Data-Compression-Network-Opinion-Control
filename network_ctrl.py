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
    else: print(f"Please enter a filename ending with .mat")

def test_controllability(A, B):
    """
    Check Hautus criteria given A and B matrices to verify controllability

    :param A: any [N x N] matrix
    :param B: any [N x N] matrix
    :return: 'True' or 'False' based on whether the Hautus criteria is satisfied
    """
    #
    if np.size(A) != np.size(B) or np.size(A,0) != np.size(A,1) or np.size(B,0) != np.size(B,1):
        print("Please use square matrices A and B")

    N = np.size(A,0)
    rank_A = matrix_rank(A)
    eig_A = eig(A).eigenvalues
    print(f"Shape of A = {A.shape}")
    print(f"Shape of eig_A = {eig_A.shape}")
    print(f"Rank A = {rank_A}")

    all_controllable = all(matrix_rank(np.hstack([A - λ * np.eye(N), B])) == N for λ in eig_A)
    print(f"\nOverall system is controllable: {all_controllable}") # Always true when B = I --> can add extra check for this to save computations

def obj_selector(obj_type, H, U, b, mu=0.1):
    """
    Wrapper for objective function for the linear model: H U = b

    :param obj_type: name of the respective objective function, choose from: { 'L1', 'L2', 'LASSO', 'RIDGE' }
    :param mu: regularization parameter
    :return: respective objective function
    """
    types = ['L1', 'L2', 'LASSO', 'RIDGE']
    if obj_type not in types: print(f"Please select from: {types}")
    elif obj_type == 'L1': return cp.Minimize(cp.norm1(U))
    elif obj_type == 'L2': return cp.Minimize(cp.norm2(H @ U - b))
    elif obj_type == 'LASSO': return cp.Minimize(cp.norm2(H @ U - b) + mu*cp.norm1(U))
    elif obj_type == 'RIDGE': return cp.Minimize(cp.norm2(H @ U - b) + mu*cp.norm2(U))

def row_normalizer(A):
    N = A.shape[0]
    for i in range(N):
        row_avg = sum(A[i][:])/N
        A[i][:] = A[i][:]/row_avg
    return A

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

def matching_persuit(H, b, s):
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
    N = H.shape[1]
    res = b.copy()  # residual
    supp = []       # support(x)
    x_supp = []     # x value at support index

    for _ in range(s): # Need s iterations for s-sparse recovery generally

        correlations = H.T @ res # inproduct or cols H with residual <H_i,r>
        k = np.argmax(np.abs(correlations)) # select the highest correlated index
        supp.append(k) # store the index as the support

        # subtract the residue (for MP)
        coeff = correlations[k]
        x_supp.append(coeff)

        res = b - coeff * H[:, k]

        # if np.linalg.norm(res) < 1e-6: break # optional convergence criteria

    # recover the full u vector
    u = np.zeros(N)
    for idx, coeff in zip(supp, x_supp): u[idx] = coeff

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
    N = H.shape[1]
    res = b.copy()  # residual
    supp = []       # support(x)
    x_supp = None   # x value at support index

    for _ in range(s): # Need s iterations for s-sparse recovery generally

        correlations = H.T @ res # inproduct or cols H with residual <H_i,r>
        k = np.argmax(np.abs(correlations)) # select the highest correlated index
        supp.append(k) # store the index as the support

        # solve least square, orthogonal projection (for OMP)
        H_selected = H[:, supp]
        x_supp = np.linalg.pinv(H_selected) @ b # LS on the support
        # x_supp = np.linalg.lstsq(H[:, supp], b, rcond=None)[0] # can use lstsq function alternatively

        res = b - H_selected

        if np.linalg.norm(res) < 1e-6: break # optional convergence criteria

    # recover the full u vector
    u = np.zeros(N)
    for idx, coeff in zip(supp, x_supp): u[idx] = coeff

    return u, supp

def optimize_input(A, x0, xf, s, K, u_max=None, method='L1'):
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
     = 'method' = selects recovery method for U, choose from {'L1', 'L2', 'LASSO', 'RIDGE', 'MP', 'OMP', 'BLK_OMP'}
    outputs:
     - 'U'      = optimal s-sparse vector
    """
    np.random.seed(1) # set seed for reproducability
    N = A.shape[0] # extract dimensions from input

    # Build controllability matrix --> TODO: can also update H recursively to save memory and computation time
    H_list = [np.linalg.matrix_power(A, K - i - 1) for i in range(K)]
    H = np.hstack(H_list)
    b = xf - np.linalg.matrix_power(A, N) @ x0

    if method == 'L1' or 'L2' or 'LASSO' or 'RIDGE':
    # Standard Basis Persuit
        U = cp.Variable(N * K) # U is a stacked vector of size [N x K]
        objective = obj_selector(method, H, U, b, mu=0.1) # Get respective objective function from wrapper (using entrywise norms)

        # as constraints we need N<=s+rank(A)=s+N (always satisfied), and controllability (always satisfied for B=I): Hx=u, can also set constraints on u
        constraints = [H @ U == b]

        if u_max is not None: # TODO --> get energy constraint (l2) from data of plotting energy vs. sparsity
            constraints.extend([
                U <= u_max,
                U >= -u_max
            ])

        problem = cp.Problem(objective, constraints)
        opt_cost = problem.solve(solver=cp.OSQP) # The optimal objective value is returned by `prob.solve()`.

        U = U.value # The optimal value for u is stored in `u.value`.

    elif method == 'MP':
        U = matching_persuit(H, b, s)[0]
    elif method == 'OMP_ref':
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(10, K))
        omp.fit(H.T, b)
        U = omp.coef_
    elif method == 'OMP':
        U = orthogonal_matching_persuit(H, b, s)[0]
    elif method == 'BLK_OMP':
        U = None
    elif method == 'COSAMP':
        U = None
    return U.value.reshape(N, K, order='F')

def main():
    mat_data = load_mat('PiecewiseSparse.mat')
    A = mat_data['A']; xf = mat_data['FinalState'] # First off, note that A contains self loops (nonzero diag.) and both positive and negative values (directional)
    N = np.size(A,0)
    test_controllability(A, np.eye(N))

if __name__ == '__main__':
    main()