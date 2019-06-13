# -*- coding: utf-8 -*-

from helpers.zernike_moments import ZernikeMoments
from helpers.local_binary_pattern import LocalBinaryPatterns

import cv2
from scipy.spatial import distance as dist


def get_dist(pointA, pointB):
    return dist.euclidean(tuple(pointA), tuple(pointB))


def normalize(image):
    return cv2.equalizeHist(image)


class PatchesFace:
    def __init__(self, shape, face):
        self.face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        self.face = normalize(self.face)
        self.shape = shape
        self.descritor_lbp = LocalBinaryPatterns(16, 1)
        self.descritor_zrnike = ZernikeMoments(4)
        self.size = (int(self.face.shape[0]/21), int(self.face.shape[1]/21))

    def compute_descritors(self, roi):
        data = []

        histogram = self.descritor_lbp.describe(roi)
        moments = self.descritor_zrnike.describe(roi)

        data.extend(histogram)
        data.extend(moments)

        return data

    def patch_p1(self):
        x, y = self.shape[48]
        w = self.size[0]
        h = self.size[1]

        roi = self.face[y-h: y + h, x-w: x + w]

        return self.compute_descritors(roi)

    def patch_p2(self):
        w = self.size[0]
        h = self.size[1]

        x1, y1 = self.shape[31]
        x = x1 - w
        y = y1 - h
        roi = self.face[y-h: y + h, x-w: x + w]

        return self.compute_descritors(roi)

    def patch_p4(self):
        x, y = self.shape[54]
        w = self.size[0]
        h = self.size[1]
        roi = self.face[y-h: y + h, x-w: x + w]

        return self.compute_descritors(roi)

    def patch_p5(self):
        w = self.size[0]
        h = self.size[1]

        x1, y1 = self.shape[35]
        x = x1 + w
        y = y1 - h
        roi = self.face[y-h: y + h, x-w: x + w]

        return self.compute_descritors(roi)

    def patch_p16(self):
        x, y = self.shape[27]
        w = self.size[0]
        h = self.size[1]

        roi = self.face[y-h: y + h, x-w: x + w]

        # cv2.rectangle(self.face, (x-w, y-h), (x+w, y+h), (0, 255, 0), 2)
        # cv2.imshow('roi', roi)
        # cv2.imshow('face', self.face)
        # cv2.waitKey(0)
        # import ipdb; ipdb.set_trace()
        return self.compute_descritors(roi)

    def patch_p18(self):
        x, y = self.shape[21]
        w = self.size[0]
        h = self.size[1]

        roi = self.face[y-h: y + h, x-w: x + w]

        return self.compute_descritors(roi)

    def patch_p19(self):
        x, y = self.shape[22]
        w = self.size[0]
        h = self.size[1]

        roi = self.face[y-h: y + h, x-w: x + w]

        return self.compute_descritors(roi)

    def patch_eyebrow(self):
        x1, y1 = self.shape[17]
        x2, y2 = self.shape[26]
        x3, y3 = self.shape[19]
        x4, y4 = self.shape[24]
        x5, y5 = self.shape[21]
        x6, y6 = self.shape[22]

        width = x2 - x1 + 4

        y = min(y1, y2, y3, y4) - 4
        x1 = x1 - 4
        height = max(y1, y2, y3, y4) - y + 4

        roi = self.face[y: y + height, x1: x1 + width]

        return self.compute_descritors(roi)

    def patch_mouth(self):
        x1, y1 = self.shape[48]
        x2, y2 = self.shape[51]
        x3, y3 = self.shape[57]
        x4, y4 = self.shape[54]

        width = x4 - x1 + 4

        y = min(y1, y2, y3, y4) - 4
        x1 = x1 - 4
        height = max(y1, y2, y3, y4) - y + 4

        roi = self.face[y: y + height, x1: x1 + width]

        return self.compute_descritors(roi)

    def dists_patches(self):
        dists = []

        dist_mouth_left = get_dist(self.shape[48], self.shape[29])
        dist_mouth_right = get_dist(self.shape[54], self.shape[29])
        dist_right_eyebrow_left = get_dist(self.shape[17], self.shape[29])
        dist_right_eyebrow_right = get_dist(self.shape[21], self.shape[29])
        dist_left_eyebrow_left = get_dist(self.shape[22], self.shape[29])
        dist_left_eyebrow_right = get_dist(self.shape[26], self.shape[29])
        dist_right_eye_left = get_dist(self.shape[36], self.shape[29])
        dist_right_eye_right = get_dist(self.shape[39], self.shape[29])
        dist_left_eye_left = get_dist(self.shape[42], self.shape[29])
        dist_left_eye_right = get_dist(self.shape[45], self.shape[29])

        dists.extend([dist_mouth_left, dist_mouth_right,
                      dist_right_eyebrow_left, dist_right_eyebrow_right,
                      dist_left_eyebrow_left, dist_left_eyebrow_right,
                      dist_right_eye_left, dist_right_eye_right,
                      dist_left_eye_left, dist_left_eye_right])

        return dists
