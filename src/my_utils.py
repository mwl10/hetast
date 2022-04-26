import pandas as pd
import numpy as np


# light curves are all modified in place...

'''read tsv to numpy array for each light curve, return them as a python list

what kind of exceptions do we want to catch down the line?
'''


def file_to_np(*args):
    max_len = 0
    light_curves = []
    for file in args:
        with open(file, 'r') as f:
            light_curve = pd.read_csv(file, sep='\t').to_numpy()
            # visually check there are three columns for each file
            print(f"dims of {file}:\t{light_curve.shape}")
            if light_curve.shape[0] > max_len:
                max_len = light_curve.shape[0]
            light_curves.append(light_curve)

    return light_curves, max_len


'''if individual light curves have more than one of the same time point, get rid of row 
-----average them -> later :)
'''


def handle_dups(lcs):
    for index, lc in enumerate(lcs):
        unique, i = np.unique(lc[:, 0], return_index=True)
        lcs[index] = lc[i]
    return lcs


'''

'''


def zero_start(lcs):
    starts = np.zeros(len(lcs))
    max_len = 0
    for i, lc in enumerate(lcs):
        starts[i] = lc[0, 0]
        lc[:, 0] = lc[:, 0] - starts[i]
        if len(lc) > max_len:
            max_len = len(lc)
    return lcs, starts, max_len


'''get all the time points across the list of light curves
---------------------------------------------------
input
---------------------------------------------------
lcs-------> python list of np light curves
---------------------------------------------------
return
---------------------------------------------------
union_tp--> 1-d numpy array of of all the time points across the light curves
'''


def union_timepoints(lcs):
    lc_lengths = np.array([len(lc) for lc in lcs])
    union_tp = np.zeros((sum(lc_lengths)))
    acc = 0
    for i, lc in enumerate(lcs):
        #lc[:,0] = lc[:,0] - lc[0,0]
        union_tp[acc:acc + len(lc)] = lc[:, 0]
        acc += len(lc)
    union_tp = np.unique(union_tp)
    union_tp.sort()

    return union_tp

    # list of all time values for every light curve


'''reorient the light curves to include all of the time points, setting 0 for the flux & error values of the new tps
---------------------------------------------------
input
---------------------------------------------------
lcs -----> python list of np light curves 
union_tp-> np 1-d array of all the time points across the light curves
---------------------------------------------------
return
---------------------------------------------------
new_lcs--> new light curves with consistent time points dims are (len(lcs), len(union_tp), 3)

'''


def include_union_tp(lcs, max_len):
    for i, lc in enumerate(lcs):
        length = len(lc)
        need_to_append = max_len - length
        lc = np.append(lc, np.zeros((need_to_append, 3)), axis=0)
        lcs[i] = lc
    return np.array(lcs)


# function for data augmentation using noise properties
# sample new light curve by adding N(0,error_bars) to flux values


def resample_lc(lc, num_samples=1):
    fluxes = lc[:, 1]
    error_bars = lc[:, 2]
    lc[:, 1] = fluxes + np.random.normal(0, error_bars)
    return lc

# def resample_lcs(lcs):
#     fluxes = lcs[:, :, 1]
#     error_bars = lcs[:, :, 2]
#     lcs[:, :, 1] = fluxes + np.random.normal(0, error_bars)
#     return lcs


# make the masks beforehand...
def make_masks(lcs, frac=0.7):
    # will depend on dimensions later
    subsampled_mask = np.zeros_like(lcs[:, :, 1])
    recon_mask = np.zeros_like(lcs[:, :, 1])
    for i, lc in enumerate(lcs):
        indexes = lc[:, 1].nonzero()[0]
        # this should vary to some extent
        length = int(np.round(len(indexes) * frac))
        obs_points = np.sort(np.random.choice(
            indexes, size=length, replace=False))
        subsampled_mask[i, obs_points] = 1
        recon_mask[i] = np.logical_xor(lc[:, 1], subsampled_mask[i])

    #recon_mask = np.split(recon_mask, len(subsampled_mask) / batch_size)
    #subsampled_mask = np.split(subsampled_mask, len(subsampled_mask) / batch_size)
    return subsampled_mask, recon_mask


def main():
    pass


if __name__ == "__main__":
    main()
