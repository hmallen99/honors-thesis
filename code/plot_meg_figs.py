import numpy as np
import matplotlib.pyplot as plt
import mne
import MEG_analysis as meg
import source_localization as srcl
from file_lists import aligned_dir, meg_locations, meg_subj_lst, ch_picks

def get_meg_data(subj):
    folder_dict = meg.get_folder_dict()
    subj_aligned = aligned_dir[subj]
    meg_dir = meg_locations[subj]

    epochs, evoked = meg.get_processed_meg_data(subj_aligned, folder_dict, meg_dir)
    return epochs, evoked


def get_source_data(subj):
    source_localization_dir = "/usr/local/freesurfer/subjects"
    subj_aligned = aligned_dir[subj]
    src, bem = srcl.get_processed_mri_data(subj_aligned, source_localization_dir)
    meg_dir = meg_locations[subj]

    epochs, evoked = get_meg_data(subj)
    cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=False)
    fwd = srcl.make_forward_sol(evoked, src, bem, "%s/%s-trans.fif" % (meg_dir, subj_aligned))

    inverse_op = srcl.make_inverse_operator(evoked, fwd, cov)
    stc, _ = srcl.apply_inverse(evoked, inverse_op)
    return stc

def plot_erfs():
    evoked_lst = []
    for subj in meg_subj_lst:
        _, evoked = get_meg_data(subj)
        #evoked.plot_topomap()
        #plt.savefig("../Figures/final_results/erfs/%s_erf.png" % subj)
        evoked_lst.append(evoked)

    evoked_all = mne.combine_evoked(evoked_lst, 'equal')
    #evoked_all.pick_types('grad').plot_topo(color='r', title="Average Evoked Response", show=False)
    plt.figure(figsize=(8, 6))
    evoked_all.plot(picks=ch_picks, show=False, spatial_colors=True)
    plt.savefig("../Figures/final_results/erfs/occ_erf.png", dpi=400)

def plot_source():
    stc_list = []
    vertices = []
    tstep = 0.01
    for subj in ["KA", "AK"]:
        stc = get_source_data(subj)
        stc = srcl.morph_to_fsaverage(stc, aligned_dir[subj])
        #srcl.plot_source(stc, subject=subj)
        stc_list.append(stc.data)
        vertices = stc.vertices
        tstep = stc.times[1] - stc.times[0]

    stc_all_data = np.mean(np.array(stc_list), axis=0)
    stc_all = mne.SourceEstimate(stc_all_data, vertices, -0.5, tstep, subject="fsaverage")
    srcl.plot_source(stc_all, subject="fsaverage", views="lateral", hemi="lh")
    for i in np.arange(0, 0.38, 0.025):
        srcl.plot_source(stc_all, initial_time=i, subject="fsaverage", views="lateral", hemi="lh")




def main():
    plot_erfs()
    #plot_source()


if __name__ == "__main__":
    main()