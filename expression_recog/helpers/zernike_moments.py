# -*- coding: utf-8 -*-

import cv2
import mahotas
import numpy as np


class ZernikeMoments:
    def __init__(self, radius):
        self.radius = radius

    def describe(self, image):
        return mahotas.features.zernike_moments(image, self.radius)

    def segment_image(self, image):
        thresh = cv2.bitwise_not(image)
        thresh[thresh > 0] = 255

        outline = np.zeros(image.shape, dtype="uint8")
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        cv2.drawContours(outline, [cnts], -1, 255, -1)

        cv2.imshow('image', image)
        cv2.imshow('thresh', thresh)
        cv2.imshow('outline', outline)
        cv2.waitKey(0)
        import ipdb; ipdb.set_trace()

        return outline
