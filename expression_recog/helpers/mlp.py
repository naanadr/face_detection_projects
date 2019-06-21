import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2


class MLP:
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None,
                 batch_size=16, num_classes=7, epochs=50):
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
        model.add(Dense(382, activation='relu',
                        input_shape=(self.input_shape,),
                        kernel_initializer='uniform',
                        bias_initializer='normal',
                        kernel_regularizer=l2(1e-5),
                        ))
        model.add(Dropout(0.2))
        model.add(Dense(382, activation='relu',
                        kernel_regularizer=l2(1e-5),
                        bias_regularizer=l2(1e-5),
                        ))
        model.add(Dropout(0.2))
        model.add(Dense(424, activation='relu',
                        kernel_regularizer=l2(1e-5),
                        bias_regularizer=l2(1e-5),
                        ))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation='softmax'))

        self.model = model

    def compile_model(self):
        self.create_model()
        opt = SGD(lr=0.001, decay=1e-4, momentum=0.8, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
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
