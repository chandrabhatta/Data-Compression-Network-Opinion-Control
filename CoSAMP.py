def CoSAMP(H,b,s,max_iter = 100,tol=1e-6):

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

    return u, sorted(list(supp))