# Functions and Class to analyze CITS data - .3ds files

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
import sklearn

# Documentation:
#https://github.com/hoffmanlabcoding/stmpy/blob/main/stmpy/doc/Stmpy%20101%20-%20getting%20started.ipynb
#https://github.com/hoffmanlabcoding/stmpy/blob/main/stmpy/doc/stmpy%20notebook%20template%20--%20topos%20and%20dos%20maps.ipynb

try:
    import stmpy
except ImportError:
    pass

import re

#import sklearn
from sklearn.linear_model import LinearRegression

import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy import optimize
import math
import scipy.optimize as optimization
import matplotlib.mlab as mlab




class CITS_Analysis():
    #import stmpy

    """
    Load and query Nanonis CITS spectroscopy data from a `.3ds` file.

    The class rearranges current and lock-in channels so the first two axes are
    spatial indices and the final axis is voltage/spectrum index.
    """
    def __init__(self, filename):
        """
        Load a `.3ds` file and cache spectroscopy arrays and metadata.

        Args:
            filename: Path to the CITS `.3ds` file.
        """
        self.biasOffset = False
        #self.data = str(filename)
        smd = stmpy.load(filename, biasOffset = self.biasOffset)
        self.data = smd
        self.header = smd.header
        self.V_range = np.asarray(smd.en)
        self.data_size = smd.header["Grid dim"]

        def rearrange_for_spectrum(array_3d):

            """
            Move the spectrum axis to the final dimension.

            Args:
                array_3d: Input array with spectrum/voltage on the first axis.

            Returns:
                Array with shape `(x, y, spectrum)`.
            """

            a1 = np.zeros((array_3d.shape[1], array_3d.shape[2], array_3d.shape[0]))
            for i in range(array_3d.shape[1]):
                for j in range(array_3d.shape[2]):
                    #print(self.current[i, j])
                    a1[i, j, :] = array_3d[:, i, j]
            return np.asarray(a1)

        self.current = rearrange_for_spectrum(smd.I)
        self.didv_x = rearrange_for_spectrum(smd.grid['LIX 1 omega (A)'])
        self.didv_y = rearrange_for_spectrum(smd.grid['LIY 1 omega (A)'])

    def get_frame_size(self):
        """
        Return the physical frame size stored in the CITS metadata.

        Returns:
            Tuple `(x_size, y_size)` parsed from the `"Grid settings"` header.
        """
        d_line = self.header["Grid settings"]
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        result = [float(x) for x in re.findall(match_number, d_line)]
        return result[2], result[3]


    def nearest_V(self, value):
        """
        Find the sampled voltage closest to a requested value.

        Args:
            value: Requested voltage.

        Returns:
            Tuple `(V_val, V_ind)` with the nearest voltage and its index.
        """
        V_val, V_ind = nearest_sample(value, self.V_range)
        return V_val, V_ind


    def nearest_point(self, coord):
        """
        Find the sampled spatial coordinate closest to a requested point.

        Args:
            coord: Requested `(x, y)` coordinate in the same physical units as
                the frame size.

        Returns:
            Tuple `([x_val, y_val], [x_ind, y_ind])` containing nearest
            coordinate values and array indices.
        """
        x_vector = np.linspace(0, self.get_frame_size()[0], self.current.shape[0])
        y_vector = np.linspace(0, self.get_frame_size()[1], self.current.shape[1])

        x_val, x_ind = nearest_sample(coord[0], x_vector)
        y_val, y_ind = nearest_sample(coord[1], y_vector)

        return [x_val, y_val], [x_ind, y_ind]

    def current_map(self, voltage):
        """
        Return the current map at the nearest sampled voltage.

        Args:
            voltage: Requested voltage.

        Returns:
            Tuple `(current_map, v_actual)`.
        """
        v_actual, v_ind = self.nearest_V(voltage)
        return self.current[:, :, v_ind], v_actual

    def didv_x_map(self, voltage):
        """
        Return the x lock-in dI/dV map at the nearest sampled voltage.

        Args:
            voltage: Requested voltage.

        Returns:
            Tuple `(didv_x_map, v_actual)`.
        """
        v_actual, v_ind = self.nearest_V(voltage)
        return self.didv_x[:, :, v_ind], v_actual



#Other functions:

def nearest_sample(value, array):
    """
    Find the element of an array closest to a requested value.

    Args:
        value: Requested scalar value.
        array: 1D array-like values to search.

    Returns:
        Tuple `(nearest_value, nearest_index)`.
    """
    d = 10000000 + np.max(array) + abs(value)
    ind = 0
    for i in range(len(array)):
        diff = abs(value - array[i])
        if diff <= d:
            ind = i
            d = diff
    return array[ind], ind
