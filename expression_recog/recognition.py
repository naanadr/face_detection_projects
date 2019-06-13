# -*- coding: utf-8 -*-

from helpers.mlp import MLP
from helpers.patches_face import PatchesFace
from helpers.read_jafee import read_samples_jafee

import cv2
import dlib
from imutils import face_utils


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

    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)

        if len(shape) != 68:
            raise ValueError('Número de pontos na face foi incorreta para a '
                             'imagem {}'.format(file))

    return face, PatchesFace(shape, image)


def read_files(images):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('helpers/'
                                     'shape_predictor_68_face_landmarks.dat')

    datas = []

    for file in images:
        data = []
        face, patches_face = get_face(file, detector, predictor)

        data.extend(patches_face.patch_p1())
        data.extend(patches_face.patch_p2())
        data.extend(patches_face.patch_p4())
        data.extend(patches_face.patch_p5())
        data.extend(patches_face.patch_p16())
        data.extend(patches_face.patch_p18())
        data.extend(patches_face.patch_p19())

        data.extend(patches_face.patch_mouth())
        data.extend(patches_face.patch_eyebrow())

        datas.append(data)

    return datas


if __name__ == '__main__':
    print('################################################################')
    print('Read samples .....')
    test_image_x, test_y, train_image_x, train_y = read_samples_jafee()

    print('################################################################')
    print('Prepare samples .....')
    train_x = read_files(train_image_x)
    test_x = read_files(test_image_x)

    mlp = MLP(x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)
    mlp.compile_model()
    print('################################################################')
    print('Train model .....')
    mlp.fit_model()

    print('################################################################')
    score = mlp.evaluate_model()
    print('Test loss: {}'.format(score[0]))
    print('Test accuracy: {}%'.format(score[1] * 100))
