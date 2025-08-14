import numpy as np


# --- Measurement Pattern / Ensemble Generators ---

def generate_hadamard_matrix(n):
    """
    Generates a Sylvester-type Hadamard matrix of size n x n.

    Args:
        n (int): The dimension of the matrix. Must be a power of 2.

    Returns:
        np.ndarray: An n x n Hadamard matrix with entries +1 or -1.
    """
    assert n > 0 and (n & (n - 1) == 0), "n must be a power of 2."
    
    if n == 1:
        return np.array([[1]])
    
    # Recursively build the matrix using Sylvester's construction
    h_half = generate_hadamard_matrix(n // 2)
    
    # H_2n = [[H_n, H_n], [H_n, -H_n]]
    h_full = np.block([
        [h_half, h_half],
        [h_half, -h_half]
    ])
    
    return h_full

def generate_bernoulli_matrix(rows, cols):
    """
    Generates a matrix with entries drawn from a Bernoulli distribution {-1, 1}.
    Each entry is +1 with probability 0.5 and -1 with probability 0.5.

    Args:
        rows (int): The number of rows for the matrix.
        cols (int): The number of columns for the matrix.

    Returns:
        np.ndarray: A rows x cols matrix with entries +1 or -1.
    """
    return np.random.choice([-1, 1], size=(rows, cols))

def generate_gaussian_matrix(rows, cols):
    """
    Generates a matrix with entries drawn from a standard normal (Gaussian)
    distribution (mean=0, variance=1).

    Args:
        rows (int): The number of rows for the matrix.
        cols (int): The number of columns for the matrix.
        
    Returns:
        np.ndarray: A rows x cols matrix with Gaussian random entries.
    """
    return np.random.randn(rows, cols)
