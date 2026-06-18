import numpy as np

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