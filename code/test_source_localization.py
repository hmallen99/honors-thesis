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

    print("reading in data")
    #evoked, epochs = meg.process_data(folder_dict['../../../../MEG_raw01/170808_noguchi_SD'], 0)
    #evoked.save('../Data/Evoked/NNo-ave.fif')
    #epochs.save('../Data/Epochs/NNo-epo.fif')
    epochs = mne.read_epochs("../Data/Epochs/NNo-epo.fif")
    evoked = mne.read_evokeds("../Data/Evoked/NNo-ave.fif")[0]
    print("computing covariance")
    cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)

    print("Starting source localization")
    subj = 'NNo'
    source_localization_dir = "/usr/local/freesurfer/subjects"
    print("creating source space")
    #src = srcl.create_source_space(subj, source_localization_dir, save=True)
    src = srcl.get_source_space("../Data/SourceSpaces/NNo-oct6-src.fif")
    print("Bem")
    #bem = srcl.make_bem(subj, source_localization_dir, save=True)
    bem = srcl.read_bem("../Data/BEM/NNo-bem-sol.fif")
    print("forward solution")
    fwd = srcl.make_forward_sol(evoked, src, bem)
    print("inv_op")
    inv_op = srcl.make_inverse_operator(evoked, fwd, cov)
    print("source estimate")
    stc, residual = srcl.apply_inverse(evoked, inv_op)
    residual.plot_topo(title='Residual Plot', show=False).savefig('../Figures/%s_residual_erf.pdf' % subj)
    for i in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        srcl.plot_source(stc, subject=subj, initial_time=i, views="dorsal", hemi="both")
        srcl.plot_source(stc, subject=subj, initial_time=i, views="caudal", hemi="both")
        srcl.plot_source(stc, subject=subj, initial_time=i, views="lateral", hemi="rh")
        srcl.plot_source(stc, subject=subj, initial_time=i, views="lateral", hemi="lh")


    return 0


if __name__ == "__main__":
    main()