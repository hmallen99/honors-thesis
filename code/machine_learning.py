import numpy as np
import mne
import sklearn
import matplotlib.pyplot as plt

from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LinearRegression

from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator,
                          get_coef, Vectorizer, Scaler)

behavior_lst = {
    "KA": "01amano1101/amano1101_session_20161101T132416",
    "MF": "07fujita0131/fujita0131_session_20170131T145759",
    "MK":  "12kawaguchi0731/kawaguchi0731_session_20170731T153637",
    "NNo": "19noguchi0808/noguchi0808_session_20170808T102813",
    "KO": "04okahashi1101/okahashi1101_session_20161101T161904",
    "HHy": "18hashidume0807/hashidume0807_session_20170807T16225",
    "HO": "06oishi0131/oishi0131_session_20170131T134216",
    "AK": "05koizumi0131/koizumi0131_session_20170131T110526",
}

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

def generate_stc_X(stc_epoch, n_train=400, n_test=100, mode="sklearn"):
    X = []
    if mode == "sklearn":
        X = np.array([next(stc_epoch).crop(0, 0.4).bin(0.025).lh_data for i in range(n_train+n_test)])
    elif mode == "keras":
        X = np.einsum('ikj->ijk', np.array([next(stc_epoch).crop(0, 0.4, False).bin(0.025).data for i in range(n_train+n_test)]))
    X_train, X_test = X[:n_train], X[n_train:n_train+n_test]
    del stc_epoch
    del X
    return X_train, X_test

def generate_epoch_X(epochs, n_train=400, n_test=100):
    epochs.load_data().resample(40)
    meg_epochs = epochs.copy().pick_types(meg=True, eeg=False).crop(0, 0.4, False)
    X = meg_epochs.get_data()

    X_train, X_test = X[:n_train], X[n_train:n_train+n_test]
    return X_train, X_test

def generate_y_classes(path, n_classes=2):
    new_gabor_lst = []
    if n_classes == 0:
        gabor_lst = load_target_gabor(path)
        for gabor in gabor_lst:
            new_gabor_lst.append(gabor)
    else:
        gabor_lst = load_target_gabor(path)
        for gabor in gabor_lst:
            new_gabor = np.floor((gabor + 90) / (180 / n_classes))
            new_gabor_lst.append(np.minimum(new_gabor, n_classes-1))
    return new_gabor_lst

def generate_y(behavior_subj, n_trials, n_train=400, n_test=100, n_classes=2):
    y = []
    for i in range(n_trials):
        y_path = "../../../../MEG/Behaviour/" + behavior_lst[behavior_subj] + "_block%s_data.mat" % (i + 1)
        y += generate_y_classes(y_path, n_classes=n_classes)
        
    y = np.array(y)
    y_train, y_test = y[:n_train], y[n_train:n_train+n_test]
    return y_train, y_test

def gabor_loss(y_true, y_pred):
    cos_diff_orig = K.cos(2 * np.pi * (y_pred / 360)) - K.cos(2 * np.pi * (y_true / 360))
    cos_diff_flip = K.cos(2 * np.pi * (y_pred + 180) / 360) - K.cos(2 * np.pi * (y_true / 360))
    sin_diff_orig = K.sin(2 * np.pi * (y_pred / 360)) - K.sin(2 * np.pi * (y_true / 360))
    sin_diff_flip = K.sin(2 * np.pi * (y_pred + 180/ 360)) - K.sin(2 * np.pi * (y_true / 360))
    diff_flip = K.square(cos_diff_flip) + K.square(sin_diff_flip)
    diff_orig = K.square(cos_diff_orig) + K.square(sin_diff_orig)
    return K.minimum(diff_flip, diff_orig)

def calc_accuracy(y_pred, y_test):
    total = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            total+=1

    return total / len(y_pred)

def plot_results(time_scale, y_pred, ml_type, subj):
    plt.plot(time_scale, y_pred)
    plt.savefig('../Figures/ML/ml_results_%s_%s.png' % (ml_type, subj))
    plt.clf()


class CosineRNNModel(object):
    def __init__(self, n_epochs=5):
        self.n_epochs = n_epochs
        self.model = keras.Sequential()
        self.model.add(layers.SimpleRNN(64, activation="relu"))
        self.model.add(layers.Dense(32, activation="relu"))
        self.model.add(layers.Dense(units=1))
        self.model.compile(loss=gabor_loss, optimizer="adam", metrics="cosine_similarity")


    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.model.fit(X, y, epochs=self.n_epochs, batch_size=10)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        _, accuracy = self.model.evaluate(X, y)
        return accuracy


class LogisticRNNModel(object):
    def __init__(self, n_epochs=5, n_outputs=2):
        self.n_epochs = n_epochs
        self.model = keras.Sequential()
        self.model.add(layers.SimpleRNN(64, activation="relu"))
        self.model.add(layers.Dense(32, activation="relu"))
        self.model.add(layers.Dense(n_outputs, activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")


    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.model.fit(X, y, epochs=self.n_epochs, batch_size=10)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        _, accuracy = self.model.evaluate(X, y)
        return accuracy

class DenseSlidingModel(object):
    def __init__(self, n_epochs=5, n_classes=2, n_timesteps=16):
        self.n_epochs = n_epochs
        self.models = []
        self.n_timesteps = n_timesteps
        for _ in range(n_timesteps):
            model = keras.Sequential()
            model.add(layers.Dense(64, activation="relu"))
            model.add(layers.Dense(32, activation="relu"))
            model.add(layers.Dense(n_classes, activation="softmax"))
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")
            self.models.append(model)


    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        for i in range(self.n_timesteps):
            self.models[i].fit(X, y, epochs=self.n_epochs, batch_size=10)
        return self
    
    def predict(self, X):
        predictions = []
        for i in range(self.n_timesteps):
            predictions.append(self.models[i].predict(X[:, i]))
        return predictions

    def evaluate(self, X, y):
        accuracies = []
        for i in range(self.n_timesteps):
            _, accuracy = self.models[i].evaluate(X[:, i], y)
            accuracies.append(accuracy)
        return accuracy


class LogisticSlidingModel(object):
    def __init__(self, max_iter=100, n_classes=2, k=200, C=1, l1_ratio=0.9):
        clf = make_pipeline(StandardScaler(),  # z-score normalization
                    SelectKBest(f_classif, k=k),  # select features for speed
                    LinearModel(LogisticRegression(C=C, solver='saga', l1_ratio=l1_ratio, penalty='elasticnet', max_iter=max_iter)))
        self.model  = SlidingEstimator(clf, scoring="accuracy")

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, n_timesteps=16):
        results = self.model.predict(X)
        accuracy_lst = [calc_accuracy(results[:, i], y) for i in range(n_timesteps)]
        return accuracy_lst

    def cross_validate(self, X, y):
        scores = mne.decoding.cross_val_multiscore(self.model, X, y, cv=5, n_jobs=1)
        return scores.mean(0)
        