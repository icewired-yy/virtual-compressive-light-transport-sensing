import numpy as np
import scipy.linalg
from enum import Enum, auto
import time
import warnings

from util.linalg import solve_sparse_coefficients, solve_linear_system


# Suppress scikit-learn warnings about early stopping in OMP
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class Strategy(Enum):
    """
    Enum for selecting the hierarchical solving strategy.

    - DOUBLE: Roughly doubles the number of coefficients at each level. Corresponds to Eq. (18)[cite: 517].
              Potentially higher quality but slower.
    - LINEAR: Adds a fixed number of new coefficients at each level. Corresponds to Eq. (19) [cite: 525].
              Faster, more efficient computation.
    """
    DOUBLE = auto()
    LINEAR = auto()


class OneChannelSolver:
    """
    Solves for the sparse reflectance coefficients of a single color channel
    using a hierarchical compressive sensing reconstruction algorithm based on Peers et al., 2009.
    """

    def __init__(self, k: int, threshold: float, strategy: Strategy):
        """
        Initializes the solver.

        Args:
            k (int): The final target number of non-zero coefficients (sparsity).
            threshold (float): The pruning threshold 'delta' for removing near-zero coefficients[cite: 395].
            strategy (Strategy): The hierarchical strategy for adding coefficients (DOUBLE or LINEAR).
        """
        if k <= 64:
            raise ValueError("Sparsity 'k' must be greater than the number of direct measurements (64).")
        
        self.k = k
        self.threshold = threshold
        self.strategy = strategy

    def _direct_solve(self, c_direct: np.ndarray, hadamard_matrix: np.ndarray) -> np.ndarray:
        """
        Solves for the first 64 coefficients using direct measurements with a Hadamard matrix[cite: 697].

        Args:
            c_direct (np.ndarray): The observation matrix for the direct part. Shape (p, 64).
            hadamard_matrix (np.ndarray): The Hadamard matrix. Shape (64, 64).

        Returns:
            lil_matrix: A sparse matrix of shape (p, 64) containing the solved direct coefficients.
        """
        num_pixels, num_direct_measurements = c_direct.shape
        if hadamard_matrix.shape != (num_direct_measurements, num_direct_measurements):
            raise ValueError("Shape mismatch for Hadamard matrix.")

        # c = \hat{t} * H
        t_hat_direct = solve_linear_system(c_direct.T, hadamard_matrix.T).T
        
        return t_hat_direct

    def solve(self, mipmap_c: list[np.ndarray], phi_ensemble: np.ndarray, hadamard_matrix: np.ndarray):
        """
        Executes the full hierarchical solving process.

        Args:
            mipmap_c (list[np.ndarray]): A list of numpy arrays representing the multi-resolution
                                         observations (P^T * C). 
                                         Starts from the coarsest level (e.g., 1x_m) to the finest (p_x_m).
            phi_ensemble (np.ndarray): The measurement ensemble for the compressive part. Shape (n, m-64).
            hadamard_matrix (np.ndarray): The Hadamard matrix for the direct part. Shape (64, 64).
        """
        print("Starting hierarchical solve...")
        start_time = time.time()

        # Extract dimensions
        num_pixels_fine = mipmap_c[-1].shape[0]
        num_coeffs_total = phi_ensemble.shape[0]
        num_total_measurements = mipmap_c[0].shape[1]
        num_direct_measurements = hadamard_matrix.shape[0]
        num_comp_measurements = num_total_measurements - num_direct_measurements
        
        if phi_ensemble.shape[1] != num_comp_measurements:
            raise ValueError("Phi ensemble measurement dimension mismatch.")

        # --- Step 1: Handle Direct Measurements ---
        print(f"Step 1: Solving for the first {num_direct_measurements} direct coefficients.")
        # The observations for the direct part are the first 64 columns of C.
        # We can get C for the coarsest level from the first mipmap level.
        # find the level of 64
        coarse_level = 3
        c_direct_coarsest = mipmap_c[coarse_level][:, :num_direct_measurements] # (p x 64)
        num_coarsest_pixel = c_direct_coarsest.shape[0]

        # Initialize the transport matrix with the direct solution for the single coarsest "pixel".
        # We use lil_matrix for efficient row-wise modifications.
        t_hat_coarsest = self._direct_solve(c_direct_coarsest, hadamard_matrix)
        # Padding to (p, num_coeffs_total)
        t_hat = np.zeros((num_coarsest_pixel, num_coeffs_total), dtype=np.float32)
        t_hat[:, :num_direct_measurements] = t_hat_coarsest

        # The solved coefficients are stored in a list of sparse matrices, one for each level.
        solved_levels = [t_hat]

        # --- Step 2: Hierarchical Iteration for Compressive Measurements ---
        print("Step 2: Starting hierarchical refinement for compressive coefficients.")
        
        # Prepare the compressive measurement ensemble and OMP solver
        # From C = T*L, for one pixel c_i = t_i*L. In CS form y = Phi*x, we have c_i^T = L^T * t_i^T.
        # So the effective measurement matrix for OMP is L^T, which is phi_ensemble.T.
        phi_T = phi_ensemble.T 
        comp_observations = [level[:, num_direct_measurements:] for level in mipmap_c]
        
        num_levels = len(mipmap_c)
        k_l = num_direct_measurements
        k_d = num_direct_measurements

        for l in range(1, num_levels):
            print(f"  Processing Level {l+1}/{num_levels} ({mipmap_c[l].shape[0]} pixels)...")
            
            num_pixels_current_level = mipmap_c[l].shape[0]
            
            # This level's solved coefficients, initialized empty.
            t_hat_current = np.zeros((num_pixels_current_level, num_coeffs_total), dtype=np.float32)
            
            # --- Determine sparsity targets for this level based on the chosen strategy ---
            if self.strategy == Strategy.DOUBLE:
                # Corresponds to Eq. (18) logic [cite: 517]
                k_d = max(1, k_l + 1)
                k_l = min(self.k, k_l + k_d)
            else: # Strategy.LINEAR
                # Corresponds to Eq. (19) logic [cite: 525]
                k_l = min(self.k, int((l + 1) * (self.k / num_levels)))
                k_d = max(1, int(self.k / num_levels))

            # Iterate through each "pixel" (row) of the current level
            for i in range(num_pixels_current_level):
                # The parent pixel index in the previous (coarser) level.
                # In a 2D quadtree structure, 4 children share 1 parent.
                parent_idx = self._find_parent_index(i, num_pixels_current_level)

                # Copy the initial solution from the parent [cite: 328, 329]
                x_init = solved_levels[l-1][parent_idx, :].ravel()
                
                # Calculate the residual in the measurement domain [cite: 331, 366]
                predicted_measurements = phi_T @ x_init
                actual_measurements = comp_observations[l][i, :]
                residual = actual_measurements - predicted_measurements

                # Use OMP to find a sparse difference vector 'd' that explains the residual [cite: 332]
                d = solve_sparse_coefficients(residual, phi_T, k_d)
                
                # Merge the initial solution with the difference [cite: 333]
                x_merged = x_init + d
                
                # --- Enforce sparsity: Prune then Limit Size ---
                # 1. Prune: Remove coefficients with magnitude below the threshold [cite: 390]
                x_merged[np.abs(x_merged) < self.threshold] = 0.0

                # 2. Limit Size: Keep only the k_l largest coefficients [cite: 384]
                if np.count_nonzero(x_merged) > k_l:
                    k_largest_indices = np.argsort(np.abs(x_merged))[-k_l:]
                    x_final = np.zeros_like(x_merged)
                    x_final[k_largest_indices] = x_merged[k_largest_indices]
                else:
                    x_final = x_merged
                
                # Store the final coefficient vector for this pixel
                t_hat_current[i, :] = x_final
            
            solved_levels.append(t_hat_current)

        # Store the final, finest-level result
        final_t_hat = solved_levels[-1]

        return final_t_hat

    def _find_parent_index(self, curr_index, curr_num_pixel):
        resolution = int(np.sqrt(curr_num_pixel))
        curr_idx_x = curr_index % resolution
        curr_idx_y = curr_index // resolution

        if curr_idx_x % 2 != 0:
            curr_idx_x -= 1
        if curr_idx_y % 2 != 0:
            curr_idx_y -= 1

        prev_resolution = resolution // 2
        parent_idx = curr_idx_y // 2 * prev_resolution + curr_idx_x // 2
        return parent_idx