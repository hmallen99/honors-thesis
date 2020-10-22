import numpy as np
import mne
import sklearn
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

def load_behavioral_data(path):
    return loadmat(path)

def load_target_gabor(path):
    data = load_behavioral_data(path)
    return data["TargetGabor"][0]

def classify_target_gabors(path):
    gabor_lst = load_target_gabor(path)
    new_gabor_lst = []
    for gabor in gabor_lst:
        #new_gabor_lst.append((gabor + 90) / 180)
        #new_gabor_lst.append(gabor)
        new_gabor = 1 if gabor > 0 else 0
        new_gabor_lst.append(new_gabor)
    return new_gabor_lst

def generate_y_classes(path, n_classes=0):
    new_gabor_lst = []
    if n_classes == 0:
        gabor_lst = load_target_gabor(path)
        for gabor in gabor_lst:
            new_gabor_lst.append(gabor)
    else:
        gabor_lst = load_target_gabor(path)
        for gabor in gabor_lst:
            new_gabor_lst.append(np.floor((gabor + 90) / (180 / n_classes)))
    return new_gabor_lst

def gabor_loss(y_true, y_pred):
    cos_diff_orig = K.cos(2 * np.pi * (y_pred / 360)) - K.cos(2 * np.pi * (y_true / 360))
    cos_diff_flip = K.cos(2 * np.pi * (y_pred + 180) / 360) - K.cos(2 * np.pi * (y_true / 360))
    sin_diff_orig = K.sin(2 * np.pi * (y_pred / 360)) - K.sin(2 * np.pi * (y_true / 360))
    sin_diff_flip = K.sin(2 * np.pi * (y_pred + 180/ 360)) - K.sin(2 * np.pi * (y_true / 360))
    cos_diff = K.minimum(K.abs(cos_diff_orig), K.abs(cos_diff_flip))
    sin_diff = K.minimum(K.abs(sin_diff_orig), K.abs(sin_diff_flip))
    return K.square(cos_diff) + K.square(sin_diff)


class RNNModel(object):
    def __init__(self, n_epochs=5):
        self.n_epochs = n_epochs
        self.model = keras.Sequential()
        self.model.add(layers.SimpleRNN(204, activation="relu"))
        self.model.add(layers.Dense(32, activation="relu"))
        self.model.add(layers.Dense(units=1))
        self.model.compile(loss=gabor_loss, optimizer="adam", metrics="accuracy")


    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.model.fit(X, y, epochs=self.n_epochs, batch_size=10)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        _, accuracy = self.model.evaluate(X, y)
        return accuracy

