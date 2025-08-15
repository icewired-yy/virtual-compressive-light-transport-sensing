import numpy as np
from src.solver.one_channel_solver import OneChannelSolver, Strategy
import scipy.linalg
from scipy.sparse import csc_matrix, lil_matrix, find


def generate_synthetic_data(p_sqrt, n_sqrt, m, k):
    """Generates synthetic data for testing the solver."""
    p = p_sqrt**2
    n = n_sqrt**2
    num_direct = 64
    num_comp = m - num_direct

    print("\n--- Generating Synthetic Data ---")
    print(f"Pixels: {p} ({p_sqrt}x{p_sqrt}), Light Dirs: {n} ({n_sqrt}x{n_sqrt}), Sparsity k: {k}")

    # 1. Ground Truth Coefficients (T_hat_true)
    # Create a p x n sparse matrix with k non-zero entries per row.
    t_hat_true = np.zeros((p, n), dtype=np.float32)
    for i in range(p):
        # Choose k random indices for non-zero coefficients
        coeffs_indices = np.random.choice(n, k, replace=False)
        # Assign random values from a standard normal distribution
        coeffs_values = np.random.randn(k)
        t_hat_true[i, coeffs_indices] = coeffs_values
    print(f"Generated T_hat_true with shape {t_hat_true.shape}")

    # 2. Measurement Matrices (L)
    # Direct part: Hadamard matrix
    hadamard_matrix = scipy.linalg.hadamard(num_direct)
    # Compressive part: Gaussian random ensemble
    phi_ensemble = np.random.randn(n, num_comp).astype(np.float32)
    print(f"Generated Hadamard matrix ({hadamard_matrix.shape}) and Phi ensemble ({phi_ensemble.shape})")

    # 3. Simulate Observations (C = T_hat * L)
    # We need to pad the hadamard matrix to match dimensions with T_hat
    l_direct = np.zeros((n, num_direct), dtype=np.float32)
    l_direct[:num_direct, :num_direct] = hadamard_matrix
    
    c_direct = t_hat_true @ l_direct
    c_comp = t_hat_true @ phi_ensemble
    c_full = np.hstack([c_direct, c_comp])
    print(f"Simulated full observation matrix C with shape {c_full.shape}")

    # 4. Create Mipmap structure from C
    mipmap_c = []
    num_levels = int(np.log2(p_sqrt)) + 1
    for i in range(num_levels):
        level_p_sqrt = 2**i
        level_p = level_p_sqrt**2
        
        if i == num_levels - 1:
            # Finest level is the original C matrix
            level_c = c_full
        else:
            # Coarser levels are block averages of the finest level
            block_size = p_sqrt // level_p_sqrt
            c_img_view = c_full.reshape(p_sqrt, p_sqrt, m)
            level_c_img = c_img_view.reshape(level_p_sqrt, block_size, level_p_sqrt, block_size, m).mean(axis=(1, 3))
            level_c = level_c_img.reshape(level_p, m)
        
        mipmap_c.append(level_c.astype(np.float32))
        print(f"  Created Mipmap Level {i} with shape {level_c.shape}")

    return t_hat_true, mipmap_c, phi_ensemble, hadamard_matrix


def evaluate_results(t_hat_true: np.ndarray, t_hat_solved: np.ndarray):
    """Calculates the error between true and solved coefficients."""
    print("\n--- Evaluating Results ---")
    p, n = t_hat_true.shape
    
    total_mse = 0
    total_support_recovered = 0

    for i in range(p):
        true_vec = t_hat_true[i, :].ravel()
        solved_vec = t_hat_solved[i].ravel()
        
        # Mean Squared Error
        mse = np.mean((true_vec - solved_vec)**2)
        total_mse += mse
        
        # Support recovery (did we find the correct non-zero indices?)
        true_support = set(find(lil_matrix(t_hat_true[i, :]))[1])
        solved_support = set(find(lil_matrix(t_hat_solved[i]))[1])

        # Jaccard index
        intersection_size = len(true_support.intersection(solved_support))
        union_size = len(true_support.union(solved_support))
        if union_size > 0:
            total_support_recovered += intersection_size / union_size

    avg_mse = total_mse / p
    avg_support_recovery = total_support_recovered / p

    print(f"Average Mean Squared Error (MSE): {avg_mse:.6f}")
    print(f"Average Support Recovery (Jaccard Index): {avg_support_recovery:.4f}")


if __name__ == '__main__':
    # --- Test Parameters ---
    # Using smaller dimensions for a quick test run.
    # The paper uses 128x128 images (p=16384, n=16384).
    PIXELS_SQRT = 128    # p = 256 pixels (16x16 image)
    LIGHT_DIRS_SQRT = 16 # n = 256 light directions
    K_SPARSITY = 128      # Each pixel's reflectance is described by 12 non-zero coeffs
    M_MEASUREMENTS = 1000 # Total measurements (64 direct + 36 compressive)
    THRESHOLD = 1e-4

    # --- Generate Data ---
    t_true, mipmap, phi, hadamard = generate_synthetic_data(
        p_sqrt=PIXELS_SQRT,
        n_sqrt=LIGHT_DIRS_SQRT,
        m=M_MEASUREMENTS,
        k=K_SPARSITY
    )

    # --- Run Solver ---
    solver = OneChannelSolver(
        k=K_SPARSITY,
        threshold=THRESHOLD,
        strategy=Strategy.LINEAR # Using the faster linear strategy for the test
    )
    t_solved = solver.solve(mipmap_c=mipmap, phi_ensemble=phi, hadamard_matrix=hadamard)
    
    # --- Get and Evaluate Results ---
    evaluate_results(t_hat_true=t_true, t_hat_solved=t_solved)