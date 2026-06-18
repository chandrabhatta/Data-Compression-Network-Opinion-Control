import numpy as np


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



