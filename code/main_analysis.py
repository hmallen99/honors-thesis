import mne
import re
import os
import numpy as np
import source_localization as srcl
import MEG_analysis as meg

def save_evoked_figs(should_save, stc_fsaverage, subj, residual):
    if should_save:
        residual.plot_topo(title='Residual Plot', show=False).savefig('../Figures/%s_residual_erf.pdf' % subj)
        srcl.save_movie(stc_fsaverage, subj)

        for i in [0, 0.1, 0.15, 0.192, 0.25, 0.3]:
            srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="dorsal", hemi="both")
            srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="caudal", hemi="both")
            srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="lateral", hemi="rh")
            srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="lateral", hemi="lh")

def main():
    folder_dict = meg.get_folder_dict()

    # TODO: take these in as cmd line inputs
    subj = 'MF-aligned'
    meg_dir = '../../../../MEG_raw01/170131_fujita_SD'
    should_save_evoked_figs = False
    should_train_model = True
    source_localization_dir = "/usr/local/freesurfer/subjects"

    # Collect Data
    epochs, evoked = meg.get_processed_meg_data(subj, folder_dict, meg_dir)
    src, bem = srcl.get_processed_mri_data(subj, source_localization_dir)
    cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)
    fwd = srcl.make_forward_sol(evoked, src, bem, "%s/%s-trans.fif" % (meg_dir, subj))

    # Generate figures for the evoked object
    if should_save_evoked_figs:
        inv_op = srcl.make_inverse_operator(evoked, fwd, cov)
        stc, residual = srcl.apply_inverse(evoked, inv_op)
        stc_fsaverage = srcl.morph_to_fsaverage(stc, subj)
        save_evoked_figs(should_save_evoked_figs, stc_fsaverage, subj, residual)

    # Train Model with Epoch data
    if should_train_model:
        inv_op_epoch = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, loose=0.2, depth=0.8)
        stc_epoch = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op_epoch, 0.11, return_generator=True)

    return 0


if __name__ == "__main__":
    main()