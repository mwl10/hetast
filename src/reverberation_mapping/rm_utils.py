import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

# saves detrended files into epoch_folder + '_det'
def linear_detrend(epoch_folder, save=False):
    if save:
        det_folder = os.path.split(epoch_folder)[0]+'_det'
        if not os.path.isdir(det_folder): os.mkdir(det_folder)
        print('saving to:', det_folder)
    for file in glob.glob(epoch_folder+'/*'): 
        data = np.loadtxt(file)
        mn = np.min(data[:,0])
        t = data[:, 0] - mn
        y = data[:, 1]
        yerr = data[:,2]
        trend = np.polyfit(t, y, deg=1)
        detrended = y - (t*trend[0])
        if save:
            fn = os.path.join(det_folder, os.path.split(file)[1])
            print('saving: ',fn)
            np.savetxt(fn, np.column_stack((t+mn, detrended, yerr)), delimiter=' ')
            
            

def ZDCF(lcf1, lcf2, outfile, acf=False, mcmc=10000, uniform=False, omit_zero_lag=True, min_ppb=0, fortran_dir='/Users/mattlowery/Desktop/Desko/code/astro/hetvae/src/reverberation_mapping/fortran_dir'):
    """
    This function calls the ZDCF program compiled in fortran using the CL 
    params: 
        lcf1
        lcf2
        -------optional--------
        fortran_dir    (str) 'fortran_dir'
        mcmc           (int)   100 -- > number of monte carlo runs for error est
        outfile        (str) out--> which file to save the results in
        uniform        (bool)        True--> are the lcs uniformly sampled?
        omit_zero_lag  (bool)  True --> omit zero lag points?
        min_ppb        (int) 0 --> min number of points per bin 
    """
    if os.path.isdir(fortran_dir): 
        os.chdir(fortran_dir) # change to dir with Fortran program
    else:
        raise Exception(f'{fortran_dir} is not a directory')
    
    uniform = 'y' if uniform else 'n'
    omit_zero_lag = 'y' if omit_zero_lag else 'n'
    if acf == True:
        params = f'1\n{outfile}\n{uniform}\n{min_ppb}\n{omit_zero_lag}\n{mcmc}\n{lcf1}'
    else:
        params = f'2\n{outfile}\n{uniform}\n{min_ppb}\n{omit_zero_lag}\n{mcmc}\n{lcf1}\n{lcf2}'
    os.system(f"printf '{params}' | ./zdcf")
    os.chdir('../')

    # Function that calls PLIKE from this notebook

def PLIKE(ccf_file, lower, upper, fortran_dir='/Users/mattlowery/Desktop/Desko/code/astro/hetvae/src/reverberation_mapping/fortran_dir'):
    '''
    This function calls the PLIKE program compiled in fortran using the CL
    params 
        ccf_file (str) Enter dcf file name
        lower (int) lower bound on peak location
        upper (int) upper bound on peak location
        ----optional-----
        fortran_dir (str) 'fortran_dir'
    p1 = input("Enter dcf file name:")
    p2 = input("Enter lower bound on peak location:")
    p3 = input("Enter upper bound on peak location:")
    '''
    if os.path.isdir(fortran_dir):
        os.chdir(fortran_dir)
    else:
        raise Exception(f'{fortran_dir} is not a directory')
    params = f'{ccf_file}\n{lower}\n{upper}'
    os.system(f"printf '{params}' | ./plike")
    os.chdir('../') # change back to current directory
    
    
def load_results_ZDCF(acf_file, ccf_file, plot=False, fortran_dir='/Users/mattlowery/Desktop/Desko/code/astro/hetvae/src/reverberation_mapping/fortran_dir'):
    path = os.path.join(os.getcwd(), fortran_dir)
    cols = ['tau', '-sig(tau)', '+sig(tau)', 'dcf', '-err(dcf)', '+err(dcf)', '#bin']
    ccf = pd.read_csv(os.path.join(path, ccf_file), sep=" ", header=None, skipinitialspace=True)
    acf = pd.read_csv(os.path.join(path, acf_file), sep=" ", header=None, skipinitialspace=True)
    ccf.columns = cols
    acf.columns = cols
    if plot:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.plot(acf['tau'], acf['dcf'], 'o--b', label='ACF', markersize=4)
        #ax.plot(-acf['tau'], acf['dcf'], 'o--b')
        ax.plot(ccf['tau'], ccf['dcf'], 's--r', label='CCF', markersize=4)
        ax.set_xlim(-120,120)
        #ax.set_ylim(-0.25, 1.1)
        ax.set_xlabel("Time", fontsize=15, labelpad=7)
        ax.set_ylabel("Correlation", fontsize=15, labelpad=7)
        ax.legend(fontsize=14)
        ax.tick_params(direction='in', pad = 5, labelsize=13)
        ax.set_title('CCF and ACF using ZDCF', fontsize=15)
        #ax.grid(which='major', axis='x', linestyle='--')
        plt.show()
    return acf, ccf