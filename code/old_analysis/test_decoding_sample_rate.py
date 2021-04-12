import numpy as np
import matplotlib.pyplot as plt
import machine_learning as ml
import load_data as ld
from file_lists import meg_subj_lst, ch_picks


def main():
    scores = []
    for subj in meg_subj_lst:
        X, _ , y, _ = ld.load_data(subj, n_train=500, n_test=0, ch_picks=ch_picks, sample_rate=128, data="epochs", n_classes=9, shuffle=True)
        model = ml.SVMSlidingModel(k=20)
        scores.append(model.cross_validate(X, y))

    scores = np.array(scores).mean(0)

    plt.plot(scores)
    plt.savefig("../Figures/128 sample rate")

if __name__ == "__main__":
    main()