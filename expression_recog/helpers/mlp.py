import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np


class MLP:
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None,
                 batch_size=32, num_classes=7, epochs=20):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.x_train = np.array(x_train).reshape(len(x_train), len(x_train[0]))
        self.y_train = y_train
        self.x_test = np.array(x_test).reshape(len(x_test), len(x_test[0]))
        self.y_test = y_test
        self.input_shape = len(x_train[0])
        self.model = None

    def normalize_labels(self):
        self.y_train = keras.utils.to_categorical(self.y_train,
                                                  self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test,
                                                 self.num_classes)

    def create_model(self):
        model = Sequential()
        model.add(Dense(512, activation='relu',
                        input_shape=(self.input_shape,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))

        self.model = model

    def compile_model(self):
        self.create_model()
        
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=RMSprop(),
                           metrics=['accuracy'])

    def fit_model(self):
        self.normalize_labels()
        self.model.fit(self.x_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,
                       validation_data=(self.x_test, self.y_test))

    def evaluate_model(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return score
