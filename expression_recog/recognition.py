# -*- coding: utf-8 -*-

from helpers.mlp import MLP
from helpers.patches_face import PatchesFace
from helpers.read_jafee import read_samples_jafee
from helpers.read_ck import read_samples_ck
from helpers.zernike_moments import ZernikeMoments
from helpers.local_binary_pattern import LocalBinaryPatterns

import cv2
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_face(file, detector, predictor):
    image = cv2.imread(file, 0)
    image = convertToRGB(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    rects = detector(image, 1)

    if len(rects) != 1:
        raise ValueError('Número de faces detectadas foi incorreta para a '
                         'imagem {}'.format(file))

    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    face = image[y:y+h, x:x+w]
    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)

        if len(shape) != 68:
            raise ValueError('Número de pontos na face foi incorreta para a '
                             'imagem {}'.format(file))

    return face, PatchesFace(shape, image, file)


def read_files(images):
    descritor_lbp = LocalBinaryPatterns(8*2, 2)
    descritor_zernike = ZernikeMoments(8)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('helpers/'
                                     'shape_predictor_68_face_landmarks.dat')

    datas = []

    for file in images:
        data = []
        face, patches_face = get_face(file, detector, predictor)

        data.extend(descritor_lbp.describe(face))
        data.extend(descritor_zernike.describe(face))

        data.extend(patches_face.patch_p1())
        data.extend(patches_face.patch_p2())
        data.extend(patches_face.patch_p3())
        data.extend(patches_face.patch_p4())
        data.extend(patches_face.patch_p5())
        data.extend(patches_face.patch_p6())
        data.extend(patches_face.patch_p7())
        data.extend(patches_face.patch_p8())
        data.extend(patches_face.patch_p9())
        data.extend(patches_face.patch_p10())
        data.extend(patches_face.patch_p11())
        data.extend(patches_face.patch_p12())
        data.extend(patches_face.patch_p13())
        data.extend(patches_face.patch_p14())

        datas.append(data)

    return datas


def normalize_labels(y_train, y_test):
    new_y_train = []
    new_y_test = []

    for i in y_train:
        y_new = np.zeros(max(y_train)+1)

        y_new[i] = 1
        new_y_train.append(y_new)

    for i in y_test:
        y_new = np.zeros(max(y_test)+1)

        y_new[i] = 1
        new_y_test.append(y_new)

    return new_y_train, new_y_test


if __name__ == '__main__':
    print('################################################################')
    print('Read samples .....')
    paths_jaff, labels_jaff = read_samples_jafee()
    paths_ck, labels_ck = read_samples_ck()

    paths = paths_jaff + paths_ck
    labels = labels_jaff + labels_ck
    X_train, X_test, y_train, y_test = train_test_split(paths, labels,
                                                        test_size=0.30)
    # y_train, y_test = normalize_labels(y_train, y_test)

    print('################################################################')
    print('Prepare samples .....')
    X_train = read_files(X_train)
    X_test = read_files(X_test)

    print('################################################################')
    print('Normalize samples with PCA .....')
    pca = PCA(n_components=270)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    print('################################################################')
    print('Create model .....')
    mlp = MLP(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
    mlp.compile_model()
    print('################################################################')
    print('Train model .....')
    history = mlp.fit_model()

    print('################################################################')
    score = mlp.evaluate_model()
    print('Test loss: {}'.format(score[0]))
    print('Test accuracy: {}%'.format(score[1] * 100))
