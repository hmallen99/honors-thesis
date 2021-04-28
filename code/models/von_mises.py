from lmfit import model
import numpy as np

def von_mises(x, y_zero, A, k, mu):
    return y_zero + A * np.exp(k * np.cos(x - mu))


def fit_von_mises(y, x):
    vm_model = Model(von_mises)
    result = vm_model.fit(y, x=x, y_zero=0.0, A=1.0, k=1.0, mu=np.pi)