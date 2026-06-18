import numpy as np

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

