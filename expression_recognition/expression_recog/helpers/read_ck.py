# -*- coding: utf-8 -*-

from pathlib import Path
import random
import re


def get_images_labels(paths):
    dict_labels = {
        '0': 4,
        '1': 0,
        '2': None,
        '3': 1,
        '4': 2,
        '5': 3,
        '6': 5,
        '7': 6,
    }

    images_paths = []
    labels_vec = []

    for path in paths:
        images = []
        labels = []

        content = path.read_text()
        label = re.findall('\s(\d)', content)[0]
        path_root = str(path.parent)
        path_root = path_root.replace('Emotion', 'cohn-kanade-images')

        path_images = sorted(Path(path_root).glob('*.png'))[-4:]

        if label != '2':
            for image in path_images:
                images.append(str(image))
                labels.append(dict_labels[label])

        label = '0'
        path_images = sorted(Path(path_root).glob('*.png'))[:1]

        for image in path_images:
            images.append(str(image))
            labels.append(dict_labels[label])

        images_paths.extend(images)
        labels_vec.extend(labels)

    return images_paths, labels_vec


def normalize(images, labels):
    min_count = min(labels.count(0), labels.count(1), labels.count(2),
                    labels.count(3), labels.count(4), labels.count(5),
                    labels.count(6))

    labels_copy = labels.copy()
    images_copy = images.copy()
    for i in range(7):
        while labels_copy.count(i) > min_count:
            pos_element = [j for j, e in enumerate(labels_copy) if e == i]
            pos = random.choice(pos_element)
            labels_copy.pop(pos)
            images_copy.pop(pos)

    return images_copy, labels_copy


def read_samples_ck():
    emotion_labels_paths = sorted(Path('./DataSets/CK+/Emotion/').glob(
                            '*/*/*.txt'))
    images, labels = get_images_labels(emotion_labels_paths)
    images, labels = normalize(images, labels)

    return images, labels
