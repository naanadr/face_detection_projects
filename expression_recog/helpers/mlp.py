import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2


class MLP:
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None,
                 batch_size=8, num_classes=7, epochs=30):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
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
        model.add(Dense(570, activation='relu',
                        input_shape=(self.input_shape,),
                        kernel_initializer='uniform',
                        bias_initializer='normal',
                        # bias_regularizer=l2(0.0001),
                        kernel_regularizer=l2(1e-4),
                        ))
        model.add(Dropout((0.15)))
        model.add(Dense(420, activation='relu',
                        bias_regularizer=l2(1e-4),
                        kernel_regularizer=l2(1e-4),
                        ))
        model.add(Dense(420, activation='relu',
                        kernel_regularizer=l2(1e-4),
                        bias_regularizer=l2(1e-4),
                        ))
        model.add(Dropout((0.15)))
        model.add(Dense(38, activation='relu',
                        kernel_regularizer=l2(1e-4),
                        bias_regularizer=l2(1e-4),
                        ))
        model.add(Dense(self.num_classes, activation='softmax'))

        self.model = model

    def compile_model(self):
        self.create_model()
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])

    def fit_model(self):
        self.normalize_labels()
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 verbose=1,
                                 validation_data=(self.x_test, self.y_test))

        return history

    def evaluate_model(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return score
