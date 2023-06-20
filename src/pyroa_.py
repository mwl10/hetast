import PyROA
import os
import pickle 


filters = ["g","r","i"] 
obj_name = "NGC5548"
psi_types = ["LogGaussian","LogGaussian"]


default = [[0.0, 10.0], [0, 30.0], [0.00, 50.0], [0.01, 10.0], [0.0, 10.0]]
med = [[0.0, 10.0], [0, 30.0], [0.00, 50.0], [5,15], [0.0, 10.0]]
high = [[0.0, 10.0], [0.0, 30.0], [0.00, 50.0], [10,30], [0.0, 10.0]]
highest = [[0.0, 10.0], [0.0, 30.0], [0.00, 50.0], [20,30], [0.0, 10.0]]
init_deltas = [1,10,20,25]

if not os.path.isdir('pyroa_fits'): 
    os.mkdir('pyroa_fits')
  
segments = ['notebooks/intrps/NGC5548/gri_seg3_10_det','notebooks/intrps/NGC5548/gri_seg34_10_det',
         'datasets/ZTF_rm_segs/NGC5548/epoch3_det','datasets/ZTF_rm_segs/NGC5548/epoch34_det']

     
for i, prior in enumerate([default,med,high,highest]):
    for segment in segments:
        fit = PyROA.Fit(segment + '/', obj_name, filters, prior, Nburnin=25000,Nsamples=35000, delay_dist=True,
                        add_var=True, psi_types=psi_types,init_delta=init_deltas[i])
        save_fn = f"{obj_name}_{i}_{segment.split('/')[-1]}"
        with open(f'pyroa_fits/{save_fn}', 'wb') as f:
            pickle.dump(fit, f, pickle.HIGHEST_PROTOCOL)
    
    