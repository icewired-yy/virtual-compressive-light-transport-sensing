from util.linalg import solve_sparse_coefficients, solve_linear_system
import numpy as np


def test_solving():
    # 1. Define the problem parameters for a well-determined system
    #    In your case, this would likely be n_coeffs = 64
    n_coeffs = 64
    
    print("--- Test for a standard linear system solver ---")
    print(f"Number of coefficients to solve: {n_coeffs}\n")
    
    # 2. Generate synthetic data
    # Create a random, non-singular square matrix A.
    # For a real application, A would be your Hadamard-based matrix.
    np.random.seed(123)
    A_matrix = np.random.rand(n_coeffs, n_coeffs)
    
    # Ensure the matrix is well-conditioned for a stable solution
    # (This step is for the test case; your Hadamard matrix should already be well-conditioned)
    A_matrix += np.eye(n_coeffs) 
    
    # Create a known ground-truth solution vector x_true
    x_true = np.random.rand(n_coeffs)
    
    # Calculate the corresponding measurement vector y = Ax
    y_vec = A_matrix @ x_true
    
    # 3. Call the implemented function to solve the system
    print("Solving the linear system Ax = y...")
    x_solved = solve_linear_system(y_vec, A_matrix)
    print("Solving complete.\n")
    
    # 4. Verify the result
    print("--- Result Verification ---")
    
    # The solved x should be very close to the true x
    # We use np.allclose for robust floating-point comparison
    is_correct = np.allclose(x_true, x_solved)
    
    print(f"Is the solved solution correct? -> {is_correct}")
    
    # Calculate the L2 norm of the difference between true and solved x
    error = np.linalg.norm(x_true - x_solved)
    print(f"Error (L2 norm) between x_true and x_solved: {error:e}")
    
    # Calculate the reconstruction error using the solved x
    reconstruction_error = np.linalg.norm(A_matrix @ x_solved - y_vec)
    print(f"Reconstruction error ||Ax_solved - y||: {reconstruction_error:e}")
    
    # Print a few values for visual comparison
    print("\n--- Value Comparison (first 5 coefficients) ---")
    print(f"{'Index':<7} {'True Value':<15} {'Solved Value':<15}")
    print("-" * 40)
    for i in range(5):
        print(f"{i:<7} {x_true[i]:<15.6f} {x_solved[i]:<15.6f}")


def test_solving_sparse_coefficients():
    # 1. Define the problem parameters
    m = 50   # Dimension of the signal (length of the pixel vector)
    n = 100  # Number of atoms in the dictionary (number of wavelet bases)
    k = 10   # Desired sparsity level

    print("--- Problem Setup ---")
    print(f"Signal dimension m = {m}")
    print(f"Dictionary size n = {n}")
    print(f"Target sparsity k = {k}\n")

    # 2. Generate synthetic data for testing
    # Create a random dictionary matrix A and normalize its columns (improves algorithm stability)
    np.random.seed(42)
    A = np.random.randn(m, n)
    A /= np.linalg.norm(A, axis=0)

    # Create a ground-truth k-sparse vector x_true
    x_true = np.zeros(n)
    true_indices = np.random.choice(n, k, replace=False)
    x_true[true_indices] = 100 * np.random.randn(k)

    # Generate the signal y = Ax + noise
    noise_level = 0.01
    noise = noise_level * np.random.randn(m)
    y = A @ x_true + noise
    
    # 3. Call the implemented function to solve for the sparse coefficients
    print("Calling solve_sparse_coefficients function...")
    x_recovered = solve_sparse_coefficients(y, A, k)
    print("Solving complete.\n")

    # 4. Verify the results
    recovered_indices = np.where(x_recovered != 0)[0]

    print("--- Result Verification ---")
    print(f"True non-zero indices: \n{np.sort(true_indices)}")
    print(f"Recovered non-zero indices: \n{np.sort(recovered_indices)}")
    
    # Calculate the accuracy of support recovery
    recovery_accuracy = len(set(true_indices) & set(recovered_indices)) / k
    print(f"\nSupport recovery accuracy: {recovery_accuracy:.2%}")

    # Calculate the reconstruction error
    reconstruction_error = np.linalg.norm(y - A @ x_recovered)
    print(f"Signal reconstruction L2 error: {reconstruction_error:.4f}")

    # Compare the coefficient values at the true support locations
    print("\n--- Coefficient Value Comparison (at true non-zero locations) ---")
    print(f"{'Index':<7} {'True Value':<15} {'Recovered Value':<15}")
    print("-" * 40)
    for idx in np.sort(true_indices):
        print(f"{idx:<7} {x_true[idx]:<15.4f} {x_recovered[idx]:<15.4f}")


if __name__ == '__main__':
    test_solving()
    test_solving_sparse_coefficients()
