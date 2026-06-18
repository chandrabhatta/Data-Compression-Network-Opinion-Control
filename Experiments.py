import numpy as np
import time
import main

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

            out = main.optimize_input(A, x0, xf, s, K, B=B, method=method)

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