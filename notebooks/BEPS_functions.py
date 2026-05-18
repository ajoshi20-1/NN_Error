import numpy as np
import matplotlib.pyplot as plt
import math
import os

from sklearn.model_selection import train_test_split
from atomai.utils import get_coord_grid, extract_patches_and_spectra, extract_subimages


def BEPS_image_spectral_pairs(beps_file_path, window_size = 16, step = 1):


    """
    Extract image-patch/spectrum pairs from a BEPS `.npz` file.

    The input file is expected to contain `image`, `spectra`, and
    `spec_step_vol` arrays. Image patches are extracted on a regular coordinate
    grid, and the spectrum at each patch coordinate is paired with that patch.

    Args:
        beps_file_path: Path to the BEPS `.npz` file.
        window_size: Width and height of each extracted image patch in pixels.
        step: Pixel spacing between neighboring patch centers.

    Returns:
        Tuple `(patches, all_spectra, indices_all, v_step)` where `patches`
        has shape `(N, window_size, window_size)`, `all_spectra` has shape
        `(N, S)`, `indices_all` contains integer image coordinates, and
        `v_step` contains the voltage axis.
    """
    input_file = np.load(beps_file_path)

    image = input_file['image']
    #print(image.shape)
    spectra = input_file['spectra']
    #print(spectra.shape)
    v_step = input_file['spec_step_vol']



    # Extract patches
    coordinates = get_coord_grid(image, step = step, return_dict=False)

    # extract image patch for each point on a grid
    window_size = window_size
    features_all, coords, _ = extract_subimages(image, coordinates, window_size)
    patches = features_all[:,:,:,0]

    indices_all = np.array(coords, dtype = int)

    # extract spectra
    n = patches.shape[0]
    all_spectra = []

    for ind in range(n):
        spectrum =  spectra[indices_all[ind,0], indices_all[ind,1]]  # indices convention is reversed for the spectra
        all_spectra.append(spectrum)

    all_spectra = np.array(all_spectra)

    return patches, all_spectra, indices_all, v_step


def extract_beps_data(beps_file_path):
    """
    Load raw BEPS arrays from a `.npz` file.

    Args:
        beps_file_path: Path to a BEPS `.npz` file containing `image`,
            `spectra`, and `spec_step_vol` arrays.

    Returns:
        Tuple `(image, spectra, v_step)` with the morphology image, spectral
        cube, and voltage axis.
    """
    input_file = np.load(beps_file_path)

    image = input_file['image']
    #print(image.shape)
    spectra = input_file['spectra']
    #print(spectra.shape)
    v_step = input_file['spec_step_vol']

    return image, spectra, v_step

