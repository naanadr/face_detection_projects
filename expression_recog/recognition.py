# -*- coding: utf-8 -*-

import cv2
from helpers.read_jafee import read_samples
from helpers.local_binary_pattern import LocalBinaryPatterns
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_face(file):
    image = cv2.imread(file, 0)
    image = convertToRGB(image)

    haar_face = cv2.CascadeClassifier(
            'DataSets/haarcascades_files/haarcascade_frontalface_default.xml')
    coordenates_face = haar_face.detectMultiScale(image, scaleFactor=1.2,
                                                  minNeighbors=5)

    if len(coordenates_face) != 1:
        raise ValueError('Número de faces detectadas foi incorreta para '
                         'imagem {}'.format(file))

    x, y, w, h = coordenates_face[0]
    return image[y:y+h, x:x+w]


def read_files(images):
    descritor = LocalBinaryPatterns(12, 4)
    datas = []

    for file in images:
        face = get_face(file)
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        histogram = descritor.describe(face)
        datas.append(histogram)

    return datas


def test_images(model, test_image_x):
    test_x = read_files(test_image_x)
    result_obt = []

    for i, image in enumerate(test_x):
        result_obt.append(model.predict(image.reshape(1, -1))[0])

    return result_obt


if __name__ == '__main__':
    test_image_x, test_y, train_image_x, train_y = read_samples()

    train_x = read_files(train_image_x)
    model = LinearSVC()
    model.fit(train_x, train_y)

    test_obt = test_images(model, test_image_x)

    print('Acurracy {0:.2f}%'.format(accuracy_score(test_y, test_obt)*100))
