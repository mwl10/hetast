




def ZDCF(lcf1, lcf2, outfile, fortran_dir='fortran_dir', ccf=False, mcmc=100, uniform=False, omit_zero_lag=True, min_ppb=0):
    """
    This function calls the ZDCF program compiled in fortran using the CL 
    params: 
        lcf1
        lcf2
        -------optional--------
        fortran_dir    (str) 'fortran_dir'
        ccf            (bool)  True --> if false, compute acf for lcf1, otherwise compute ccf between lcf1 & lcf2
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
    
    if ccf:
        params = f'2\n{outfile}\n{uniform}\n{min_ppb}\n{omit_zero_lag}\n{mcmc}\n{lcf1}\n{lcf2}'
    else: # acf
        params = f'1\n{outfile}\n{uniform}\n{min_ppb}\n{omit_zero_lag}\n{mcmc}\n{lcf1}'
    
    os.system(f'{params} | ./zdcf')
    #!printf "{params}" | ./zdcf # saves to outfile
    os.chdir('../')

    # Function that calls PLIKE from this notebook

def PLIKE(ccf_file, lower, upper, fortran_dir='fortran_dir'):
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
                 
    os.chdir(path) # change to dir with Fortran program


    parameters = p1+'\n'+p2+'\n'+p3
    params = f'{ccf_file}\n{lower}\n{upper}'
    os.system(f'{params} | ./plike')
    os.chdir('../') # change back to current directory
    
    
def load_results_ZDCF(acf_file, ccf_file, fortran_dir='fortran_dir', plot=False):
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