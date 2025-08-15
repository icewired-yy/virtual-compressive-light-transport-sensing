import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

def solve_sparse_coefficients(y, A, k):
    """
    Solves for sparse coefficients using the Orthogonal Matching Pursuit (OMP) algorithm.

    This function aims to find a k-sparse solution x that minimizes the reconstruction
    error ||y - Ax||_2^2, subject to the constraint ||x||_0 <= k.

    Args:
        y (numpy.ndarray): The signal vector to be represented, with a shape of 
                           (m, 1) or (m,). In your context, this is the m x 1 pixel vector.
        A (numpy.ndarray): The dictionary or basis matrix, with a shape of (m, n).
                           In your context, this is the m x n transform matrix (e.g., wavelet basis).
        k (int):           The desired sparsity level, representing the maximum number
                           of non-zero elements in the solution vector x.

    Returns:
        numpy.ndarray: The computed sparse coefficient vector x, with a shape of (n,).
    """
    # Validate the input parameter k for correctness
    if not (0 < k < A.shape[1]):
        raise ValueError(f"Sparsity level k must be between 1 and {A.shape[1]-1}. Got k={k}")
        
    # Ensure y is a 1-dimensional vector to meet scikit-learn's input requirements
    if y.ndim != 1:
        y_vec = y.ravel()
    else:
        y_vec = y

    # Initialize the OMP solver, specifying the target number of non-zero coefficients
    omp_solver = OrthogonalMatchingPursuit(n_nonzero_coefs=k)

    # Fit the model using the dictionary A and the signal y
    omp_solver.fit(A, y_vec)

    # Return the solved sparse coefficient vector
    x_sparse = omp_solver.coef_
    
    return x_sparse


def solve_linear_system(y, A):
    """
    Solves a standard linear system of equations Ax = y.

    This function finds the vector x that minimizes the Euclidean 2-norm ||Ax - y||^2.
    It is a general-purpose solver suitable for well-determined, over-determined,
    or under-determined systems.

    Args:
        y (numpy.ndarray): The measurement or observation vector, with a shape of 
                           (m, 1) or (m,).
        A (numpy.ndarray): The coefficient matrix of the linear system, with a shape 
                           of (m, n). In your case, this is derived from Hadamard patterns.

    Returns:
        numpy.ndarray: The solution vector x, with a shape of (n,).
    """
    # Using numpy.linalg.lstsq for a robust solution.
    # It solves for x in Ax = y.
    # rcond=None is recommended to use the machine precision for the condition number check.
    # The function returns a tuple: (solution_x, residuals, rank, singular_values).
    # We are primarily interested in the first element, which is the solution vector x.
    
    x, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    
    # Optional: Check if the system was well-posed.
    # For a square matrix A, a low rank might indicate a problem (singular matrix).
    if rank < A.shape[1]:
        print(f"Warning: The rank of matrix A ({rank}) is less than the number of columns ({A.shape[1]}). "
              "The solution is a least-squares approximation.")

    return x
