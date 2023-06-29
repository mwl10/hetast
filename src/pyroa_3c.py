import multiprocessing
import PyROA
import os
import pickle


filters = ["g", "r", "i"]
obj_name = "3C273"
psi_types = ["LogGaussian", "LogGaussian"]

default = [[0.0, 10.0], [0, 30.0], [0.00, 50.0], [0.01, 10.0], [0.0, 10.0]]
med = [[0.0, 10.0], [0, 30.0], [0.00, 50.0], [5, 15], [0.0, 10.0]]
init_deltas = [1, 10]

if not os.path.isdir('pyroa_fits'):
    os.mkdir('pyroa_fits')

segments = ['notebooks/intrps/3C273/gri_seg3_10_det',
            'datasets/ZTF_rm_segs/3C273/epoch3_det',
            'notebooks/intrps/3C273/gri_seg3_5_det']
segments = [os.path.join('../../',segment) for segment in segments]

def fit_and_save(segment, obj_name, filters, prior, psi_types, init_delta):
    fit = PyROA.Fit(segment + '/', obj_name, filters, prior, Nburnin=20000, Nsamples=50000, delay_dist=True,
                    add_var=True, psi_types=psi_types, init_delta=init_delta)
    save_fn = f"pyroa_fits/{obj_name}_{i}_{segment.split('/')[-1]}"
    with open(save_fn, 'wb') as f:
        pickle.dump(fit, f, pickle.HIGHEST_PROTOCOL)


# Create a pool of worker processes
pool = multiprocessing.Pool()
for segment in segments:
    for i, prior in enumerate([default, med]):
        # Use the pool of worker processes to parallelize the fits
        pool.apply_async(fit_and_save, args=(segment, obj_name, filters, prior, psi_types, init_deltas[i]))

# Close the pool and wait for all processes to finish
pool.close()