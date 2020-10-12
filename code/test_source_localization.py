import mne
import re
import os
import numpy as np
import source_localization as srcl
import MEG_analysis as meg


def main():
    home = '../../../../MEG_raw01/'
    home_folders = os.listdir(home)
    home_folders = [os.path.join(home, entry) for entry in home_folders if re.findall(r"^\w+", entry)]
    folder_dict = {}
    for folder in home_folders:
        entries = os.listdir(folder)
        folder_dict[folder] = [os.path.join(folder, entry) for entry in entries if re.findall(r"^\w+SD_\w*raw.fif", entry)]

    # TODO: take these in as cmd line inputs
    subj = 'MK-aligned'
    meg_dir = '../../../../MEG_raw01/170731_kawaguchi_SD'
    load = True
    source_localization_dir = "/usr/local/freesurfer/subjects"

    epochs = []
    evoked = []
    src = []
    bem = []
    info = []

    print("reading in data")
    if load:
        epochs = mne.read_epochs("../Data/Epochs/%s-epo.fif" % subj)
        evoked = mne.read_evokeds("../Data/Evoked/%s-ave.fif" % subj)[0]
        src = srcl.get_source_space("../Data/SourceSpaces/%s-oct6-src.fif" % subj)
        bem = srcl.read_bem("../Data/BEM/%s-bem-sol.fif" % subj)
    else:
        evoked, epochs, info = meg.process_data(folder_dict[meg_dir], 0)
        evoked.save('../Data/Evoked/%s-ave.fif' % subj)
        epochs.save('../Data/Epochs/%s-epo.fif' % subj)
        src = srcl.create_source_space(subj, source_localization_dir, save=True)
        bem = srcl.make_bem(subj, source_localization_dir, save=True)
    
    cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)
    fwd = srcl.make_forward_sol(evoked, src, bem, "%s/%s-trans.fif" % (meg_dir, subj))
    inv_op = srcl.make_inverse_operator(evoked, fwd, cov)
    stc, residual = srcl.apply_inverse(evoked, inv_op)
    residual.plot_topo(title='Residual Plot', show=False).savefig('../Figures/%s_residual_erf.pdf' % subj)
    stc_fsaverage = srcl.morph_to_fsaverage(stc, subj)
    #stc_fsaverage = stc
    srcl.save_movie(stc_fsaverage, subj)

    for i in [0, 0.1, 0.15, 0.192, 0.25, 0.3]:
        srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="dorsal", hemi="both")
        srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="caudal", hemi="both")
        srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="lateral", hemi="rh")
        srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="lateral", hemi="lh")


    return 0


if __name__ == "__main__":
    main()