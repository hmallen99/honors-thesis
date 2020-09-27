import mne
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

    evoked, epochs = meg.process_data(folder_dict['../../../../MEG_raw01/170131_fujita_SD'], 0)
    cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)

    subj = 'MF'
    source_localization_dir = "/usr/local/freesurfer/subjects"
    src = srcl.create_source_space(subj, source_localization_dir)
    bem = srcl.make_bem(subj, source_localization_dir)
    fwd = srcl.make_forward_sol(evoked, src, bem)
    inv_op = srcl.make_inverse_operator(evoked, fwd, cov)
    stc, residual = srcl.apply_inverse(evoked, inv_op)


    return 0


if __name__ == "__main__":
    main()