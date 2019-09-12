# -*- coding: utf-8 -*-

from skimage import feature
import numpy as np


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        n_bins = int(lbp.max() + 1)
        (hist, _) = np.histogram(lbp.ravel(), density=True,
                                 bins=n_bins, range=(0, n_bins))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist
