import numpy as np
import math


# --- Core 2D FWT Functions  ---

def forward_transform(image):
    """
    Performs a 2D Fast Wavelet Transform (FWT) on an N x N image.
    N must be a power of 2.
    This function acts as the core engine for the HaarCoefficient class.
    """
    # Helper for 1D forward transform
    def _haar_1d_forward(data):
        assert len(data) % 2 == 0
        averages = (data[::2] + data[1::2]) / np.sqrt(2)
        details = (data[::2] - data[1::2]) / np.sqrt(2)
        return np.concatenate((averages, details))

    h, w = image.shape
    assert h == w and (h & (h - 1) == 0) and h != 0, "Image must be square and its size a power of 2"
    
    coeffs = image.astype(float)
    size = w
    
    while size > 1:
        # Transform along rows
        for i in range(size):
            coeffs[i, :size] = _haar_1d_forward(coeffs[i, :size])
        
        # Transform along columns
        for i in range(size):
            coeffs[:size, i] = _haar_1d_forward(coeffs[:size, i])
            
        size //= 2
        
    return coeffs

def inverse_transform(coeffs):
    """
    Performs a 2D inverse Fast Wavelet Transform on an N x N coefficient matrix.
    N must be a power of 2.
    This function acts as the core engine for the HaarCoefficient class.
    """
    # Helper for 1D inverse transform
    def _haar_1d_inverse(coeffs_1d):
        assert len(coeffs_1d) % 2 == 0
        n = len(coeffs_1d)
        half = n // 2
        averages = coeffs_1d[:half]
        details = coeffs_1d[half:]
        
        data = np.zeros(n)
        data[::2] = (averages + details) / np.sqrt(2)
        data[1::2] = (averages - details) / np.sqrt(2)
        return data

    h, w = coeffs.shape
    assert h == w and (h & (h - 1) == 0) and h != 0, "Coefficients must be square and its size a power of 2"
    
    image = coeffs.astype(float)
    size = 2
    
    while size <= w:
        # Inverse transform along columns
        for i in range(size):
            image[:size, i] = _haar_1d_inverse(image[:size, i])

        # Inverse transform along rows
        for i in range(size):
            image[i, :size] = _haar_1d_inverse(image[i, :size])
            
        size *= 2
        
    return image


class HaarCoefficient:
    """
    A class to encapsulate a Haar coefficient matrix and its properties.
    """
    def __init__(self, coeffs_matrix):
        """
        Primary constructor.
        
        Args:
            coeffs_matrix (np.ndarray): The N x N matrix of Haar coefficients.
        """
        h, w = coeffs_matrix.shape
        assert h == w and (h & (h - 1) == 0) and h != 0, "Coefficients matrix must be square and its size a power of 2"
        self._coeffs = coeffs_matrix
        self._resolution = w
        self._total_levels = int(math.log2(w))

    @classmethod
    def from_image(cls, image):
        """
        Factory method to create a HaarCoefficient object from an image.
        """
        coeffs = forward_transform(image)
        return cls(coeffs)

    @classmethod
    def from_lowpass(cls, lowpass_image, total_levels):
        """
        Special constructor (Interface 2b).
        Initializes coefficients from a low-resolution image, embedding it
        into a larger coefficient space.

        Args:
            lowpass_image (np.ndarray): The low-resolution source image.
            total_levels (int): The total number of levels for the target space
                                (e.g., 7 for a 128x128 space).
        """
        lowpass_coeffs = forward_transform(lowpass_image)
        lowpass_dim = lowpass_image.shape[0]
        
        full_dim = 2**total_levels
        full_coeffs = np.zeros((full_dim, full_dim))
        full_coeffs[:lowpass_dim, :lowpass_dim] = lowpass_coeffs
        
        return cls(full_coeffs)

    @property
    def coeffs(self):
        """Returns the full coefficient matrix."""
        return self._coeffs

    @property
    def level(self):
        """Returns the total number of levels in this coefficient space."""
        return self._total_levels
        
    def get_coeffs_at_level(self, level):
        """
        Gets the detail coefficient sub-matrices for a specific level (Interface 2a).
        
        Args:
            level (int): The target level (1 is the coarsest detail level).

        Returns:
            tuple: A tuple containing (Horizontal, Vertical, Diagonal) sub-matrices.
                   Returns (None, None, None) if level is invalid.
        """
        if not (1 <= level <= self._total_levels):
            return None, None, None
            
        dim_prev = 2**(level - 1)
        dim_curr = 2**level
        
        # Horizontal details (LH)
        h_band = self._coeffs[0:dim_prev, dim_prev:dim_curr]
        # Vertical details (HL)
        v_band = self._coeffs[dim_prev:dim_curr, 0:dim_prev]
        # Diagonal details (HH)
        d_band = self._coeffs[dim_prev:dim_curr, dim_prev:dim_curr]
        
        return h_band, v_band, d_band

    def get_lowpass_image_coeffs(self, target_level=None):
        """
        Gets the top-left part of the coefficient matrix, which represents
        the low-pass (downscaled) version of the image (Interface 2c).

        Args:
            target_level (int, optional): The level of the desired low-pass image. 
                                         Defaults to the max level.
        
        Returns:
            np.ndarray: The top-left sub-matrix of coefficients.
        """
        if target_level is None:
            target_level = self._total_levels
            
        dim = 2**target_level
        return self._coeffs[:dim, :dim]

    def downgrade(self, target_level):
        """
        Creates a new, lower-resolution HaarCoefficient object by truncating
        the high-frequency coefficients (Interface 2d).

        Args:
            target_level (int): The desired new maximum level.

        Returns:
            HaarCoefficient: A new HaarCoefficient object with high-frequency
                             coefficients zeroed out.
        """
        if target_level >= self._total_levels:
            return HaarCoefficient(self._coeffs.copy())
            
        new_coeffs = np.zeros_like(self._coeffs)
        dim = 2**target_level
        new_coeffs[:dim, :dim] = self._coeffs[:dim, :dim]
        
        return HaarCoefficient(new_coeffs)

    def to_image(self):
        """
        Reconstructs the image from the coefficients.
        This is one of the new main interfaces.
        """
        return inverse_transform(self._coeffs)
