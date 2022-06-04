import pandas as pd
import numpy as np


def make_masks(lcs, frac=0.7):
  # frac taken should depend on amount of points we have... ?
    # will depend on dimensions later
    subsampled_mask = np.zeros_like(lcs[:,:,1])
    recon_mask = np.zeros_like(lcs[:,:,1])
    for i,lc in enumerate(lcs):
        indexes = lc[:,1].nonzero()[0]
        # this should vary to some extent
        length = int(np.round(len(indexes) * frac))
        obs_points = np.sort(np.random.choice(indexes, size=length, replace=False))
        subsampled_mask[i, obs_points] = 1
        recon_mask[i] = np.logical_xor(lc[:,1], subsampled_mask[i])

    return subsampled_mask, recon_mask


def main():
    pass
    
    
if __name__ == "__main__":
    main()





