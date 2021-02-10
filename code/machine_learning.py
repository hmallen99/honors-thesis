import numpy as np
import mne
import sklearn
from source_localization import morph_to_fsaverage
import matplotlib.pyplot as plt

from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator,
                          get_coef, Vectorizer, Scaler)

behavior_lst = {
    "KA": "01amano1101/amano1101_session_20161101T132416",
    "MF": "07fujita0131/fujita0131_session_20170131T145759",
    "MK":  "12kawaguchi0731/kawaguchi0731_session_20170731T153637",
    "NNo": "19noguchi0808/noguchi0808_session_20170808T102813",
    "KO": "04okahashi1101/okahashi1101_session_20161101T161904",
    "HHy": "18hashidume0807/hashidume0807_session_20170807T162259",
    "HO": "06oishi0131/oishi0131_session_20170131T134216",
    "AK": "05koizumi0131/koizumi0131_session_20170131T110526",
}

new_beh_lst = {
    "KA": 1,
    "MF": 7,
    "MK": 12,
    "NNo": 19,
    "KO": 4,
    "HHy": 18,
    "HO": 6,
    "AK": 5,
    "HN": 21,
    "NN": 3,
    "JL": 9,
    "DI": 16,
    "SoM": 2,
    "TE": 17,
    "VA": 10,
    "RS": 14,
    "YMi": 11,
}

def load_behavioral_data(path):
    return loadmat(path)

def load_target_gabor(path):
    data = load_behavioral_data(path)
    return data["TargetGabor"][0]

def load_pred_data(path):
    data = load_behavioral_data(path)
    return data["GabOrSpec"]

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

def plot_results(time_scale, y_pred, ml_type, subj, training_err=[]):
    if len(training_err) > 0:
        plt.errorbar(time_scale, y_pred, yerr=training_err)
    else:
        plt.plot(time_scale, y_pred)
    plt.ylim((0.05, 0.25))
    plt.savefig('../Figures/ML/ml_results_%s_%s.png' % (ml_type, subj))
    plt.clf()

def plot_behavior(behavior_subj, n_trials):
    y_actual = []
    y_pred = []
    for i in range(n_trials):
        y_path = "../../../../MEG/Behaviour/" + behavior_lst[behavior_subj] + "_block%s_data.mat" % (i + 1)
        y_actual.append(list(load_target_gabor(y_path)))
        y_pred.append(list(load_pred_data(y_path)))

    plt.scatter(y_actual, y_pred)
    # TODO: Make Behavior directory
    plt.savefig("../Figures/Behavior/%s_behavior.png" % behavior_subj)
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
    def __init__(self, n_epochs=5, n_classes=4, n_timesteps=16):
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.n_timesteps = n_timesteps
        self.set_model()

    def set_model(self):
        self.model = keras.Sequential()
        self.model.add(layers.SimpleRNN(64, activation="relu"))
        self.model.add(layers.Dense(32, activation="relu"))
        self.model.add(layers.Dense(self.n_classes, activation="softmax"))
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

    def cross_validate(self, X, y):
        kfold = KFold(n_splits=5, shuffle=True)
        y = y.flatten().astype(int)
        y_hot = np.zeros((y.size, self.n_classes))
        y_hot[np.arange(y.size), y] = 1
        y = y_hot

        accuracies = []
        scale = StandardScaler()

        for i in range(self.n_timesteps):
            X[:, :, i] = scale.fit_transform(X[:, :, i])

        for train, test in kfold.split(X, y):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            self.model.fit(X_train, y_train, batch_size=40, epochs=self.n_epochs)
            _, accuracy = self.model.evaluate(X_test, y_test)
            accuracies.append(accuracy)
        
        return np.mean(np.array(accuracies))

class DenseSlidingModel(object):
    def __init__(self, n_epochs=5, n_classes=2, n_timesteps=16):
        self.n_epochs = n_epochs
        self.models = []
        self.n_timesteps = n_timesteps
        self.n_classes = n_classes
        self.set_models()

    def set_models(self):
        self.models = []
        for _ in range(self.n_timesteps):
            model = keras.Sequential()
            model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-2)))
            model.add(layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2)))
            model.add(layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2)))
            model.add(layers.Dense(self.n_classes, activation="softmax"))
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")
            self.models.append(model)
        print("set model")

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        for i in range(self.n_timesteps):
            self.models[i].fit(X[:, i], y, epochs=self.n_epochs, batch_size=10)
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
        return accuracies

    def cross_validate(self, X, y):
        # TODO: cross validation
        kfold = KFold(n_splits=5, shuffle=True)
        y = y.flatten().astype(int)
        y_hot = np.zeros((y.size, self.n_classes))
        y_hot[np.arange(y.size), y] = 1
        y = y_hot

        accuracies = []
        scale = StandardScaler()

        for i in range(self.n_timesteps):
            X[:, :, i] = scale.fit_transform(X[:, :, i])



        for train, test in kfold.split(X, y):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            split_accuracies = []
            for i in range(self.n_timesteps):
                print("timestep: %d" % i)
                self.models[i].fit(X_train[:, :, i], y_train, batch_size=5, epochs=self.n_epochs)
                print(X_train[:, :, i].shape)
                _, accuracy = self.models[i].evaluate(X_test[:, :, i], y_test)
                split_accuracies.append(accuracy)
            
            self.set_models()
            accuracies.append(split_accuracies)
        accuracies = np.array(accuracies)
        return accuracies.mean(0)
            


class LogisticSlidingModel(object):
    def __init__(self, max_iter=100, n_classes=2, k=200, C=1, l1_ratio=0.9):
        self.clf = Pipeline([('scaler', StandardScaler()), 
                        ('f_classif', SelectKBest(f_classif, k)),
                        ('linear', LinearModel(LogisticRegression(C=C, solver='saga', l1_ratio=l1_ratio, penalty='elasticnet', max_iter=max_iter, multi_class="multinomial")))])
        self.model  = SlidingEstimator(self.clf, scoring="accuracy")

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
        scores = mne.decoding.cross_val_multiscore(self.model, X, y, cv=5, n_jobs=-1)
        return scores.mean(0)

    def get_features(self, subj, i):
        features = self.model.estimators_[i].named_steps['f_classif'].get_support()
        np.save("%s_k_best" % subj, features)
        return features

    def plot_weights_stc(self, subj, vertices):
        filters = get_coef(self.model, "patterns_", inverse_transform=True)
        stc_feat = mne.SourceEstimate(np.abs(filters[:, -1, :]), vertices=vertices, 
                                        tmin=0, tstep=0.025, subject=subj)

        stc_feat = morph_to_fsaverage(stc_feat, subj)
        for i in np.arange(0, 0.39, 0.025):
            brain = stc_feat.plot(views="flat", transparent=True, initial_time=i, hemi="both",
                                    time_unit='s', subjects_dir="/usr/local/freesurfer/subjects", surface="flat")

            brain.save_image("../Figures/weights/%s_%.3fweights.png" % (subj, i))
            brain.close()

    def plot_weights_epochs(self, subj, epochs):
        patterns = get_coef(self.model, "patterns_", inverse_transform=True)
        evoked = mne.EvokedArray(patterns[:, -1, :], epochs.info, tmin=epochs.tmin)
        evoked.plot_topomap(title="%s weights" % subj, time_unit='s', times=np.arange(0,0.39, 0.025))
        plt.savefig('../Figures/weights/%s_epochs.png' % subj)
        plt.clf()




class SVMSlidingModel(object):
    def __init__(self, k=200, C=1):
        self.clf = Pipeline([('scaler', StandardScaler()), 
                        ('f_classif', SelectKBest(f_classif, k)),
                        ('linear', LinearModel(LinearSVC(C=C, max_iter=4000)))])
        self.model  = SlidingEstimator(self.clf, scoring="accuracy")

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
        scores = mne.decoding.cross_val_multiscore(self.model, X, y, cv=5, n_jobs=-1)
        return scores.mean(0)

    def get_features(self, subj, i):
        features = self.model.estimators_[i].named_steps['f_classif'].get_support()
        np.save("%s_k_best" % subj, features)
        return features


class RandomForestSlidingModel(object):
    def __init__(self, k=200, C=1):
        self.clf = Pipeline([('scaler', StandardScaler()), 
                        ('f_classif', SelectKBest(f_classif, k)),
                        ('linear', LinearModel(RandomForestClassifier()))])
        self.model  = SlidingEstimator(self.clf, scoring="accuracy")

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
        scores = mne.decoding.cross_val_multiscore(self.model, X, y, cv=5, n_jobs=-1)
        return scores.mean(0)

    def get_features(self, subj, i):
        features = self.model.estimators_[i].named_steps['f_classif'].get_support()
        np.save("%s_k_best" % subj, features)
        return features
        