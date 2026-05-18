
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from scipy.ndimage import zoom

import numpy as np


def edges_zeroed_image(image: np.ndarray, consider_pixel: int) -> np.ndarray:
    """
    Zero the borders of a 2D image while preserving a centered region.

    Args:
        image: 2D image array.
        consider_pixel: Side length of the centered region to keep.
            Rounded up to the nearest even number so the border is symmetric
            on both sides.

    Returns:
        New array with the same shape as `image`, where everything outside the
        centered `consider_pixel x consider_pixel` region is zero.
        If `consider_pixel` is at least as large as the smaller image
        dimension, a copy of the image is returned unchanged.

    Raises:
        ValueError: If `image` is not 2D or `consider_pixel` is less than 2.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}.")

    if consider_pixel < 2:
        raise ValueError(f"`consider_pixel` must be >= 2, got {consider_pixel}.")

    # Round up to an even number so the border splits evenly on both sides.
    consider_pixel += consider_pixel % 2

    h, w = image.shape
    if consider_pixel >= min(h, w):
        return image.copy()

    # Per-axis offsets so the kept region is centered on both axes.
    diff_h = (h - consider_pixel) // 2
    diff_w = (w - consider_pixel) // 2

    output = np.zeros_like(image)
    output[diff_h:h - diff_h, diff_w:w - diff_w] = image[diff_h:h - diff_h, diff_w:w - diff_w]

    return output


def interpolated_center_crop(
    image: np.ndarray,
    consider_pixel: int,
    image_width: int = None,
    order: int = 1,
) -> np.ndarray:
    """
    Crop a centered image region and resize it to a square output.

    This is the interpolation analog of `edges_zeroed_image`: instead of
    preserving the original spatial extent and zeroing the border, it stretches
    the centered crop to fill the output image.

    Args:
        image: 2D image array.
        consider_pixel: Side length of the centered region to keep, before
            interpolation. Rounded up to the nearest even number so the border
            splits symmetrically on both sides. If this is at least the smaller
            image dimension, the whole image is used as the crop.
        image_width: Side length of the output image. Defaults to the smaller
            input image dimension.
        order: Spline interpolation order passed to scipy.ndimage.zoom.
            Common values are 0 for nearest neighbor, 1 for bilinear, and
            3 for bicubic.

    Returns:
        2D array of shape `(image_width, image_width)`.

    Raises:
        ValueError: If `image` is not 2D, `consider_pixel` is less than 2, or
        `image_width` is less than 1.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}.")
    if consider_pixel < 2:
        raise ValueError(f"`consider_pixel` must be >= 2, got {consider_pixel}.")
    if image_width is None:
        image_width = min(image.shape)
    if image_width < 1:
        raise ValueError(f"`image_width` must be >= 1, got {image_width}.")

    # Round up to even so the centered crop is symmetric.
    consider_pixel += consider_pixel % 2

    h, w = image.shape

    # If the requested crop is larger than the image, just use the whole image.
    if consider_pixel >= min(h, w):
        crop = image
    else:
        diff_h = (h - consider_pixel) // 2
        diff_w = (w - consider_pixel) // 2
        crop = image[diff_h:h - diff_h, diff_w:w - diff_w]

    # Interpolate the crop up (or down) to (image_width, image_width).
    # `zoom` takes per-axis scaling factors, so compute them from the
    # current crop size rather than assuming a fixed input shape.
    crop_h, crop_w = crop.shape
    zoom_factors = (image_width / crop_h, image_width / crop_w)

    out = zoom(crop, zoom=zoom_factors, order=order)

    # `zoom` can occasionally return shapes off by 1 from rounding.
    # Snap to the exact target size if that happens.
    if out.shape != (image_width, image_width):
        out = out[:image_width, :image_width]

    return out

def append_multiscale_data(images: np.ndarray, spectra: np.ndarray, scales: list,
                           coordinates: np.ndarray, include_ori_set: bool = True,
                           append_image_type =  'pad') -> tuple:
    """
    Build an augmented multiscale image/spectra dataset.

    Each requested scale creates one augmented copy of every input image. The
    scale value is appended to the corresponding coordinate vector so downstream
    models can distinguish the spatial context used for each sample.

    Args:
        images: Original image array of shape `(N, H, W)`.
        spectra: Original spectra array of shape `(N, S)`.
        scales: Side lengths of the central regions to keep.
        coordinates: Original coordinate array of shape `(N, 2)` containing
            `(x, y)` for each image.
        include_ori_set: If True, the original (un-zeroed) images are also included,
            with the image width appended as their scale.
        append_image_type: Augmentation mode. `"pad"` zeroes image edges while
            keeping the original image size. `"interpolate"` crops the central
            region and resizes it back to the original image size.

    Returns:
        Tuple `(aug_images, aug_spectra, aug_coordinates)` where augmented
        images have shape `(M, H, W)`, spectra have shape `(M, S)`, and
        coordinates have shape `(M, 3)` as `(x, y, scale)`.

    Raises:
        AssertionError: If image, spectra, and coordinate counts do not match.
        ValueError: If any scale is at least the image width, or if
            `append_image_type` is not `"pad"` or `"interpolate"`.
    """
    augmented_images = []
    augmented_spectra = []
    augmented_coordinates = []

    images = images.astype(np.float32)
    coordinates = coordinates.astype(np.float32)
    image_width = images.shape[2]

    assert images.shape[0] == spectra.shape[0], "Number of images and spectra must match."
    assert images.shape[0] == coordinates.shape[0], "Number of images and coordinates must match."

    for scale in scales:

        if scale >= image_width:
            raise ValueError(f"Scale {scale} is too large for image width {image_width}. "
                             f"Must be less than {image_width}.")

        for img, spec, coordinate in zip(images, spectra, coordinates):
            if append_image_type == 'pad':
                aug_img = edges_zeroed_image(img, consider_pixel=scale)
            elif append_image_type == 'interpolate':
                aug_img = interpolated_center_crop(img, consider_pixel=scale)
            else:
                raise ValueError(f"Invalid append_image_type: {append_image_type}. "
                                 f"Must be 'pad' or 'interpolate'.")

            augmented_images.append(aug_img)
            augmented_spectra.append(spec)
            augmented_coordinates.append(np.append(coordinate, scale))  # Append scale to coordinates for potential use in model

    if include_ori_set:
        for img, spec, coordinate in zip(images, spectra, coordinates):
            augmented_images.append(img)
            augmented_spectra.append(spec)
            augmented_coordinates.append(np.append(coordinate, image_width))  # Original images use full image_width as scale

    augmented_images = np.array(augmented_images)
    augmented_spectra = np.array(augmented_spectra)
    augmented_coordinates = np.array(augmented_coordinates)

    print(f"Augmented images shape: {augmented_images.shape}, "
          f"Augmented spectra shape: {augmented_spectra.shape}, "
          f"Augmented coordinates shape: {augmented_coordinates.shape}")

    return augmented_images, augmented_spectra, augmented_coordinates
