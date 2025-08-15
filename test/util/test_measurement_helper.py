import numpy as np
import matplotlib.pyplot as plt
import util.measurement_helper as mh

def run_pattern_tests():
    """
    Runs tests to verify the measurement pattern generator functions.
    """
    print("--- Starting Measurement Pattern Generator Test ---")

    # --- 1. Test Hadamard Matrix Generation ---
    n_hadamard = 16
    print(f"\n1. Testing Hadamard Matrix Generation for size {n_hadamard}x{n_hadamard}...")
    hadamard_matrix = mh.generate_hadamard_matrix(n_hadamard)

    # Verify the core property of a Hadamard matrix: H * H^T = n * I
    identity_matrix = np.identity(n_hadamard)
    product = np.dot(hadamard_matrix, hadamard_matrix.T)
    
    is_orthogonal = np.allclose(product, n_hadamard * identity_matrix)
    print(f"-> Orthogonality property verified: {'Yes' if is_orthogonal else 'No'}")
    
    # --- 2. Test Bernoulli Matrix Generation ---
    n_bernoulli = 64
    print(f"\n2. Testing Bernoulli Matrix Generation for size {n_bernoulli}x{n_bernoulli}...")
    bernoulli_matrix = mh.generate_bernoulli_matrix(n_bernoulli, n_bernoulli)
    
    # Verify that the matrix only contains -1 and 1
    unique_values = np.unique(bernoulli_matrix)
    contains_only_pm1 = np.all(np.isin(unique_values, [-1, 1]))
    print(f"-> Contains only +1 and -1: {'Yes' if contains_only_pm1 else 'No'}")

    # --- 3. Test Gaussian Matrix Generation ---
    n_gaussian = 64
    print(f"\n3. Testing Gaussian Matrix Generation for size {n_gaussian}x{n_gaussian}...")
    gaussian_matrix = mh.generate_gaussian_matrix(n_gaussian, n_gaussian)
    
    # Verify the statistics (mean should be close to 0, std dev close to 1)
    mean = np.mean(gaussian_matrix)
    std_dev = np.std(gaussian_matrix)
    print(f"-> Statistics: Mean={mean:.4f} (expected ~0), StdDev={std_dev:.4f} (expected ~1)")

    # --- 4. Visualize the generated matrices ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(hadamard_matrix, cmap='gray')
    axes[0].set_title(f'Hadamard Matrix ({n_hadamard}x{n_hadamard})')
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(bernoulli_matrix, cmap='gray')
    axes[1].set_title(f'Bernoulli Matrix ({n_bernoulli}x{n_bernoulli})')
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(gaussian_matrix, cmap='gray')
    axes[2].set_title(f'Gaussian Matrix ({n_gaussian}x{n_gaussian})')
    fig.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

    print("\n--- Pattern Test Finished ---")

if __name__ == '__main__':
    run_pattern_tests()