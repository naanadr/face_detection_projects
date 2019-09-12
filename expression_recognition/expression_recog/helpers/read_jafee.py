# -*- coding: utf-8 -*-

from pathlib import Path
import random
import re


def train_test_index(models_path):
    len_models = len(models_path)

    train_model_index = [x for x in random.sample(range(len_models), 7)]
    teste_model_index = [x for x in range(len_models)
                         if x not in train_model_index]

    return sorted(teste_model_index), sorted(train_model_index)


def split_dataset(models_path):
    teste_i, train_i = train_test_index(models_path)

    test_x, test_y = get_labels(models_path, teste_i)
    train_x, train_y = get_labels(models_path, train_i)

    return test_x, test_y, train_x, train_y


def get_labels(paths, index):
    labels_dict = {
        'AN': 0,
        'DI': 1,
        'FE': 2,
        'HA': 3,
        'NE': 4,
        'SA': 5,
        'SU': 6
    }

    images_paths = []
    images_labels = []

    for i in range(index):
        paths_images = sorted(Path(str(paths[i])).glob('*.tiff'))

        for image in paths_images:
            image = str(image)
            pos = labels_dict[
                        re.findall('([A-Z]*)\d*.\d*.tiff', image)[0]]
            images_paths.append(image)
            images_labels.append(pos)

    return images_paths, images_labels


def read_samples_jafee():
    models_path = sorted(Path('./DataSets/JAFEE/').glob('*'))
    models_path = [x for x in models_path if x.is_dir()]

    return get_labels(models_path, len(models_path))
