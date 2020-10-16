import numpy as np
import mne
import sklearn
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_behavioral_data(path):
    return loadmat(path)

def load_target_gabor(path):
    data = load_behavioral_data(path)
    return data["TargetGabor"][0]

def classify_target_gabors(path):
    gabor_lst = load_target_gabor(path)
    new_gabor_lst = []
    for gabor in gabor_lst:
        gabor = np.round((gabor + 90) / 180)
        new_gabor_lst.append(gabor)
    return new_gabor_lst


class RNNModel(object):
    def __init__(self, n_epochs=5):
        self.n_epochs = n_epochs
        self.model = keras.Sequential()
        #self.model.add(layers.GRU(256, return_sequences=True))
        self.model.add(layers.SimpleRNN(256, activation="relu"))
        self.model.add(layers.Dense(64, activation="relu"))
        self.model.add(layers.Dense(32, activation="relu"))
        self.model.add(layers.Dense(1, activation="sigmoid"))
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")


    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.model.fit(X, y, epochs=self.n_epochs, batch_size=10)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        _, accuracy = self.model.evaluate(X, y)
        return accuracy

