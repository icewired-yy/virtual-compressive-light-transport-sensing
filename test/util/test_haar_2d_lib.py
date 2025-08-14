# test_haar.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import os
import util.haar_2d_lib as hwl

def setup_test_image(path="test_image.png", size=(128, 128)):
    """
    Downloads a sample image if it doesn't exist, then loads, converts to
    grayscale, and resizes it.
    """
    if not os.path.exists(path):
        print(f"Downloading sample image to {path}...")
        url = R"https://th.bing.com/th/id/R.207480d574b4346e6cc51e02fa8fc81b?rik=etBG%2fzmgUJezEw&pid=ImgRaw&r=0"
        urllib.request.urlretrieve(url, path)
    
    img = Image.open(path).convert('L') # Convert to grayscale
    img = img.resize(size, Image.Resampling.LANCZOS)
    return np.array(img)

def run_tests():
    """
    Runs tests to verify the refactored Haar wavelet library.
    """
    print("--- Starting Refactored Haar Wavelet Library Test ---")
    
    # 1. Load test image from a file
    resolution = 128
    image = setup_test_image(size=(resolution, resolution))
    print(f"\n1. Loaded and resized test image to {resolution}x{resolution}.")

    # 2. Test creation from image and perfect reconstruction
    print("\n2. Testing HaarCoefficient.from_image() and to_image()...")
    haar_obj = hwl.HaarCoefficient.from_image(image)
    reconstructed_image = haar_obj.to_image()
    
    is_perfect = np.allclose(image, reconstructed_image)
    print(f"-> Perfect reconstruction successful: {'Yes' if is_perfect else 'No'}")

    # 3. Test the 'downgrade' method and partial reconstruction
    print("\n3. Testing downgrade() method...")
    # Downgrade to level 4 (16x16 lowpass equivalent)
    level_to_keep = 4 
    downgraded_haar_obj = haar_obj.downgrade(level_to_keep)
    blurry_image = downgraded_haar_obj.to_image()
    print(f"-> Downgraded to level {level_to_keep} and reconstructed blurry image.")

    # 4. Test getting coefficients at a specific level
    print("\n4. Testing get_coeffs_at_level()...")
    level_to_get = 3 # 4x4 -> 8x8 transition
    h_band, v_band, d_band = haar_obj.get_coeffs_at_level(level_to_get)
    print(f"-> Extracted Level {level_to_get} detail bands.")
    print(f"   - Horizontal band shape: {h_band.shape}")
    print(f"   - Vertical band shape:   {v_band.shape}")
    print(f"   - Diagonal band shape:   {d_band.shape}")

    # 5. Test the special constructor 'from_lowpass'
    print("\n5. Testing from_lowpass() constructor...")
    small_image = image[::16, ::16] # Create an 8x8 lowpass image
    total_levels = 7 # for a 128x128 space
    haar_from_lowpass = hwl.HaarCoefficient.from_lowpass(small_image, total_levels)
    print(f"-> Created a {2**total_levels}x{2**total_levels} coefficient space from an {small_image.shape[0]}x{small_image.shape[0]} image.")
    # Check if the top-left of the new coeffs is non-zero and the rest is zero
    top_left_sum = np.sum(np.abs(haar_from_lowpass.coeffs[:8, :8]))
    bottom_right_sum = np.sum(np.abs(haar_from_lowpass.coeffs[8:, 8:]))
    print(f"-> Sum of top-left 8x8 coeffs: {top_left_sum:.2f} (should be > 0)")
    print(f"-> Sum of other coeffs: {bottom_right_sum:.2f} (should be = 0)")

    # 6. Visualize all results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    
    plt.subplot(2, 3, 2)
    plt.imshow(np.log(1 + np.abs(haar_obj.coeffs)), cmap='gray')
    plt.title("Log of Full Coefficients")
    
    plt.subplot(2, 3, 3)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title("Perfect Reconstruction")
    
    plt.subplot(2, 3, 4)
    plt.imshow(blurry_image, cmap='gray')
    plt.title(f"Reconstruction from Level {level_to_keep} Coeffs")

    plt.subplot(2, 3, 5)
    recon_from_lowpass = haar_from_lowpass.to_image()
    plt.imshow(recon_from_lowpass, cmap='gray')
    plt.title("Recon from Lowpass Constructor")

    plt.subplot(2, 3, 6)
    # Just show a detail band for visualization
    level_to_show = 5
    h_band, _, _ = haar_obj.get_coeffs_at_level(level_to_show)
    plt.imshow(h_band, cmap='gray')
    plt.title(f"Level {level_to_show} Horizontal Details")
    
    plt.tight_layout()
    plt.show()

def run_down_grade_test():
    resolution = 128
    image = setup_test_image(size=(resolution, resolution))
    print(f"\n1. Loaded and resized test image to {resolution}x{resolution}.")

    # 2. Test creation from image and perfect reconstruction
    print("\n2. Testing HaarCoefficient.from_image() and to_image()...")
    haar_obj = hwl.HaarCoefficient.from_image(image)
    reconstructed_image = haar_obj.to_image()

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")

    plt.subplot(3, 3, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title("Reconstructed Image")

    # 3. Test the 'downgrade' method and partial reconstruction
    total_level = haar_obj.level
    for level_to_keep in range(0, total_level):
        downgraded_haar_obj = haar_obj.downgrade(level_to_keep)
        blurry_image = downgraded_haar_obj.to_image()
        plt.subplot(3, 3, level_to_keep + 3)
        plt.imshow(blurry_image, cmap='gray')
        plt.title(f"Level {level_to_keep} Blurry Image")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_down_grade_test()
