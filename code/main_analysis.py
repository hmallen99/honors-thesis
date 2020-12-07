import mne
import re
import os
import numpy as np
import source_localization as srcl
import MEG_analysis as meg
import machine_learning as ml

meg_subj_lst = [
    "KA",
    "MF",
    "MK",
    "NNo",
    "KO",
    "HHy",
    "HO",
    "AK",
]

aligned_dir = {
    "KA": "KA-aligned",
    "MF": "MF-aligned",
    "MK":  "MK-aligned",
    "NNo": "NNo-4",
    "KO": "KO-aligned",
    "HHy": "HHy-aligned",
    "HO": "HO-aligned",
    "AK": "AK-aligned",
}

def save_evoked_figs(should_save, stc_fsaverage, subj, residual):
    if should_save:
        residual.plot_topo(title='Residual Plot', show=False).savefig('../Figures/%s_residual_erf.pdf' % subj)
        srcl.save_movie(stc_fsaverage, subj)

        for i in [0, 0.1, 0.2, 0.3]:
            srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="dorsal", hemi="both")
            srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="caudal", hemi="both")
            #srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="lateral", hemi="rh")
            srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="lateral", hemi="lh")

def run_subject(behavior_subj, should_save_evoked_figs=False, should_train_epoch_model=False, 
                should_train_stc_model=True, cross_val=True, evaluate=False, permutation_test=False):

    folder_dict = meg.get_folder_dict()
    subj = aligned_dir[behavior_subj]
    meg_dir = meg.meg_locations[behavior_subj]
    source_localization_dir = "/usr/local/freesurfer/subjects"
    results = []
    model = None
    X_train, X_test = [], []
    y_train, y_test = [], []
    figure_label = "default"
    n_classes = 4

    # Collect Data
    epochs, evoked = meg.get_processed_meg_data(subj, folder_dict, meg_dir)
    src, bem = srcl.get_processed_mri_data(subj, source_localization_dir)
    cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=False)
    fwd = srcl.make_forward_sol(evoked, src, bem, "%s/%s-trans.fif" % (meg_dir, subj))

    # Generate figures for the evoked object
    if should_save_evoked_figs:
        inv_op = srcl.make_inverse_operator(evoked, fwd, cov)
        stc, residual = srcl.apply_inverse(evoked, inv_op)
        stc_fsaverage = srcl.morph_to_fsaverage(stc, subj)
        save_evoked_figs(should_save_evoked_figs, stc_fsaverage, subj, residual)

    # Train Model with Epoch data
    if should_train_epoch_model:
        X_train, X_test = ml.generate_epoch_X(epochs, n_train=400, n_test=100)
        y_train, y_test = ml.generate_y(behavior_subj, 5, n_train=400, n_test=100, n_classes=n_classes)
        model = ml.LogisticSlidingModel(max_iter=1500, n_classes=5, k=20, C=0.1, l1_ratio=0.95)
        figure_label = "epochs"

    # Train Model with Source Estimate data
    if should_train_stc_model:
        inv_op_epoch = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, loose=0.2, depth=0.8)
        stc_epoch = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op_epoch, 0.11, return_generator=True)
        X_train, X_test = ml.generate_stc_X(stc_epoch, n_train=400, n_test=100, mode="sklearn")
        y_train, y_test = ml.generate_y(behavior_subj, 5, n_train=400, n_test=100, n_classes=n_classes)
        model = ml.LogisticSlidingModel(max_iter=1500, n_classes=5, k=1000, C=0.05, l1_ratio=0.95)
        figure_label = "source"
    
    print("\n\nHistogram:")
    hist = np.histogram(y_train, bins=np.arange(n_classes+1))
    print(hist)
    print("\n\n\n")

    if permutation_test:
        figure_label += "_permutation"
        np.random.shuffle(y_train)
        np.random.shuffle(y_test)

    if cross_val:
        figure_label += "_cross_val"
        results = model.cross_validate(X_train, y_train) 
    elif evaluate:
        figure_label += "_test"
        model.fit(X_train, y_train)
        results = model.evaluate(X_test, y_test)
        
    ml.plot_results(np.linspace(0, 0.375, 16), results, figure_label, subj)

    return results

def main():
    training_results = []
    for subject in meg_subj_lst:
        ml.plot_behavior(subject, 5)
        #result = run_subject(subject, permutation_test=True, should_train_epoch_model=True, should_train_stc_model=False)
        #training_results.append(result)
    
    #training_error = np.std(np.array(training_results), axis=0)
    #training_results = np.array(training_results).mean(0)
    #ml.plot_results(np.linspace(0, 0.375, 16), training_results, "cross_val_permutation_error", "epochs_average", training_err=training_error)

    return 0


if __name__ == "__main__":
    main()