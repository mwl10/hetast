import PyROA
import os
import pickle 


filters = ["g","r","i"] 
objName = "MCG+08-11-011"
psi_types = ["LogGaussian"]

## need the dirs to run this, 
## pkl the fit

# for every 500 obs from the interpolation? move over every 250? 


default = [[0.0, 20.0],[0.0, 100.0], [-50.0, 50.0], [0.01, 10.0], [0.0, 10.0]]
            # rms      # mean       # lag        # width of window  # extra error 
priors = [[0.0, 20.0], [0.0, 100.0], [-50.0, 50.0], [0.01, 10.0], [0.0, 10.0]]

inc_window = [[0.0, 20.0], [0.0, 100.0], [-50.0, 50.0], [10,30], [0.0, 10.0]]

if not os.path.isdir('pyroa_fits'): os.mkdir('pyroa_fits')
  
 
# segment 1: 
#     58300
#     58600
    

# segment 2:   
#     58700
#     59000

seg1 = 'notebooks/intrps/MCG+08-11-011/gri_finet_seg1/'
seg2 = 'notebooks/intrps/MCG+08-11-011/gri_finet_seg2/'


fit1 = PyROA.Fit(seg1, objName, filters, default, Nburnin=15000,Nsamples=20000, delay_dist=True, add_var=True, psi_types=psi_types,)# init_delta=20.0)

fit2 = PyROA.Fit(seg2, objName, filters, default, Nburnin=15000,Nsamples=20000, delay_dist=True, add_var=True, psi_types=psi_types,)# init_delta=20.0)


  
with open('pyroa_fits/fit1.pkl', 'wb') as f:
    pickle.dump(fit1, f, pickle.HIGHEST_PROTOCOL)
    
with open('pyroa_fits/fit2.pkl', 'wb') as f:
    pickle.dump(fit2, f, pickle.HIGHEST_PROTOCOL)