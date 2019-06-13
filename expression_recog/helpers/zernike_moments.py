# -*- coding: utf-8 -*-

import cv2
import mahotas


class ZernikeMoments:
    def __init__(self, radius):
        self.radius = radius

    def describe(self, image):
        image = self.segment_image(image)
        return mahotas.features.zernike_moments(image, self.radius)

    def segment_image(self, image):
        thresh = cv2.adaptiveThreshold(image, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 5, 8)

        return thresh
