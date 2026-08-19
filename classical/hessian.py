"""
Hessian Matrix Analysis
=======================
Compute Hessian matrix eigenvalues for ridge/filament detection in solar images.
Identifies elongated dark structures by analyzing local curvature.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from typing import Tuple


def compute_hessian(image: np.ndarray, sigma: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Hessian matrix of a 2D image at a given scale.

    Args:
        image: Grayscale image (float64)
        sigma: Gaussian scale for derivative computation

    Returns:
        Hxx, Hxy, Hyy: Second-order partial derivatives
    """
    img = image.astype(np.float64)

    # Smooth with Gaussian at scale sigma
    smoothed = gaussian_filter(img, sigma=sigma)

    # Compute second derivatives using finite differences on smoothed image
    Hyy, Hxx = np.gradient(np.gradient(smoothed, axis=0), axis=0), \
               np.gradient(np.gradient(smoothed, axis=1), axis=1)
    Hxy = np.gradient(np.gradient(smoothed, axis=0), axis=1)

    # Scale normalization (sigma^2 for second derivatives)
    scale = sigma ** 2
    return Hxx * scale, Hxy * scale, Hyy * scale


def compute_eigenvalues(Hxx: np.ndarray, Hxy: np.ndarray, Hyy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues of the Hessian matrix at each pixel.

    Returns lambda1, lambda2 where |lambda1| <= |lambda2|
    """
    # Eigenvalues of 2x2 symmetric matrix [[Hxx, Hxy], [Hxy, Hyy]]
    trace = Hxx + Hyy
    det = Hxx * Hyy - Hxy ** 2
    discriminant = np.sqrt(np.maximum(trace ** 2 - 4 * det, 0))

    lambda1 = (trace - discriminant) / 2
    lambda2 = (trace + discriminant) / 2

    # Sort by absolute value: |lambda1| <= |lambda2|
    abs1, abs2 = np.abs(lambda1), np.abs(lambda2)
    swap = abs1 > abs2
    lambda1_sorted = np.where(swap, lambda2, lambda1)
    lambda2_sorted = np.where(swap, lambda1, lambda2)

    return lambda1_sorted, lambda2_sorted


def hessian_ridge_response(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Compute ridge response from Hessian eigenvalues.

    For dark ridges (filaments in H-alpha): lambda2 >> 0 indicates a dark ridge.
    For bright ridges (inverted image): lambda2 << 0 indicates a bright ridge.

    Returns the ridge response map.
    """
    Hxx, Hxy, Hyy = compute_hessian(image, sigma)
    lambda1, lambda2 = compute_eigenvalues(Hxx, Hxy, Hyy)

    # Ridge response: strong when one eigenvalue is much larger in magnitude
    # For dark ridges on bright background: lambda2 > 0 (concave up = dark)
    response = np.maximum(lambda2, 0)

    return response


def multiscale_hessian_response(image: np.ndarray,
                                  scales: list = [1, 2, 3, 5, 8]) -> np.ndarray:
    """
    Compute multi-scale Hessian ridge response (max across scales).
    """
    responses = []
    for sigma in scales:
        resp = hessian_ridge_response(image, sigma)
        responses.append(resp)

    # Maximum response across scales
    return np.max(responses, axis=0)
