from util.mipmap import Mipmap
import numpy as np


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def create_test_image(width=256, height=256):
        """Creates a sample image with gradients for easy visualization."""
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        x = np.linspace(0, 255, width)
        y = np.linspace(0, 255, height)
        xv, yv = np.meshgrid(x, y)
        
        img_array[:, :, 0] = xv  # Red channel gradient
        img_array[:, :, 1] = yv  # Green channel gradient
        img_array[:, :, 2] = 128 # Blue channel constant
        
        return img_array

    # 1. Create a sample image
    # Note: Using a non-power-of-two and odd dimension to test robustness
    test_image = create_test_image(width=256, height=256) 
    print(f"Created a test image with shape: {test_image.shape}\n")

    # 2. Construct the Mipmap object
    print("Generating Mipmap pyramid...")
    mipmap = Mipmap(test_image)
    print(f"Mipmap generation complete. Total levels: {mipmap.num_levels}\n")

    # 3. Display all Mipmap levels using Matplotlib
    fig, axes = plt.subplots(1, mipmap.num_levels, figsize=(15, 4))
    fig.suptitle('Mipmap Pyramid Levels (Downscaled by Averaging)', fontsize=16)

    for i in range(mipmap.num_levels):
        # Access the level using the __getitem__ slicing interface
        level_image = mipmap[i]
        
        ax = axes[i]
        ax.imshow(level_image)
        h, w, _ = level_image.shape
        ax.set_title(f"Level {i}\n{w}x{h}")
        ax.axis('off')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

    # 4. Example of accessing a specific level
    try:
        level_3_img = mipmap.get_level(3)
        print(f"Successfully retrieved Level 3 image with shape: {level_3_img.shape}")
        
        # This will raise an error
        invalid_level_img = mipmap.get_level(99)
    except IndexError as e:
        print(f"\nCaught expected error: {e}")