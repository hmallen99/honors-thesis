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

from file_lists import new_beh_lst, behavior_lst

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

def gabor_loss2(y_true, y_pred):
    diff = y_true - y_pred
    diff = ((diff + 90) % 180) - 90
    return K.mean(K.square(diff), axis=-1) + K.mean(100 * (y_pred / 181))

def gabor_metric(y_true, y_pred):
    diff = y_true - y_pred
    diff = ((diff + 90) % 180) - 90
    return K.mean(K.square(diff), axis=-1)

def calc_accuracy(y_pred, y_test):
    total = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            total+=1

    return total / len(y_pred)

def plot_results(time_scale, y_pred, ml_type, subj, ymin=0.15, ymax=0.35, training_err=[]):
    if len(training_err) > 0:
        plt.errorbar(time_scale, y_pred, yerr=training_err)
    else:
        plt.plot(time_scale, y_pred)
    plt.ylim((ymin, ymax))
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
        self.model.add(layers.SimpleRNN(64, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-2)))
        self.model.add(layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-2)))
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
            self.set_model()
        
        return [np.mean(np.array(accuracies)) for _ in range(16)]

class DenseSlidingModel(object):
    def __init__(self, n_epochs=5, n_classes=2, n_timesteps=16, loss="categorical_crossentropy"):
        self.n_epochs = n_epochs
        self.models = []
        self.n_timesteps = n_timesteps
        self.n_classes = n_classes
        self.loss = loss
        self.set_models()
        # This can be a function or a string
        

    def set_models(self):
        self.models = []
        for _ in range(self.n_timesteps):
            model = keras.Sequential()
            #model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-2)))
            model.add(layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(layers.Dense(8, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.03, l2=0.03)))
            model.add(layers.Dense(self.n_classes, activation="softmax"))
            model.compile(loss=self.loss, optimizer="adam", metrics="categorical_accuracy")
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
                self.models[i].fit(X_train[:, :, i], y_train, batch_size=8, epochs=self.n_epochs)
                _, accuracy = self.models[i].evaluate(X_test[:, :, i], y_test)
                #print(self.models[i].predict(X_test[:, :, i]))
                split_accuracies.append(accuracy)
            
            self.set_models()
            accuracies.append(split_accuracies)
        accuracies = np.array(accuracies)
        return accuracies.mean(0)

class CNNSlidingModel(DenseSlidingModel):
    def __init__(self, input_shape, n_classes=4):
        self.input_shape = input_shape
        DenseSlidingModel.__init__(self, n_epochs=5, n_classes=n_classes)
        

    def set_models(self):
        model = keras.Sequential()
        #model.add(layers.BatchNormalization(axis=[-3, -2, -1]))
        model.add(layers.Conv3D(16, 3, activation='relu', input_shape=self.input_shape))
        model.add(layers.Conv3D(8, 3, activation='relu'))
        #model.add(layers.Conv3D(16, 3, activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(self.n_classes, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="categorical_accuracy")
        self.model = model


    def cross_validate(self, X, y):
        kfold = KFold(n_splits=5, shuffle=True)
        y = y.flatten().astype(int)
        y_hot = np.zeros((y.size, self.n_classes))
        y_hot[np.arange(y.size), y] = 1
        y = y_hot

        accuracies = []
        print(X.shape)


        for train, test in kfold.split(X, y):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            
            self.model.fit(X_train, y_train, batch_size=20, epochs=self.n_epochs)
            _, accuracy = self.model.evaluate(X_test, y_test)
            accuracies.append(accuracy)
            print(self.model.predict(X_test))
            print(y_test)
            self.set_models()

        avg_acc = np.mean(accuracies)
        return [avg_acc for i in range(16)]
            
class GaborSlidingModel(DenseSlidingModel):
    def __init__(self):
        DenseSlidingModel.__init__(self, n_epochs=30)

    def set_models(self):
        self.models = []
        for _ in range(self.n_timesteps):
            model = keras.Sequential()
            model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-1, l2=1e-1)))
            model.add(layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-1, l2=1e-1)))
            model.add(layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-1, l2=1e-1)))
            model.add(layers.Dense(1))
            model.compile(loss=gabor_loss2, optimizer="adam", metrics=gabor_metric)
            self.models.append(model)
        print("set model")

    def cross_validate(self, X, y):
        kfold = KFold(n_splits=5, shuffle=True)
        y = y.flatten()

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
                preds = self.models[i].predict(X_test[:, :, i])
                print(y_test)
                print(preds)
                split_accuracies.append(accuracy)
            
            self.set_models()
            accuracies.append(split_accuracies)
        accuracies = np.array(accuracies) / 90
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

    def get_patterns(self, X, y, epochs_info):
        self.model.fit(X, y)
        patterns = get_coef(self.model, "patterns_", inverse_transform=True)
        return mne.EvokedArray(patterns[:, -1, :], epochs_info, tmin=0)

class SVMSlidingModel(object):
    def __init__(self, k=200, C=1):
        self.clf = Pipeline([('scaler', StandardScaler()), 
                        ('f_classif', SelectKBest(f_classif, k)),
                        ('linear', LinearModel(LinearSVC(C=C, max_iter=6000)))])
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

    def get_patterns(self, X, y, epochs_info):
        self.model.fit(X, y)
        patterns = get_coef(self.model, "patterns_", inverse_transform=True)
        return mne.EvokedArray(patterns[:, -1, :], epochs_info, tmin=0)

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
        