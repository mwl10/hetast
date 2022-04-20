import pandas as pd
import numpy as np

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
      unique, i = np.unique(lc[:,0], return_index=True)
      lcs[index] = lc[i]
    return lcs



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
        union_tp[acc:acc + len(lc)] = lc[:,0]
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
    for i,lc in enumerate(lcs):
        length = len(lc)
        need_to_append = max_len - length
        lc = np.append(lc, np.zeros((need_to_append, 3)), axis=0)
        lcs[i] = lc
    return np.array(lcs)

 
  # new_lcs = np.zeros((len(lcs), len(union_tp), 3))
  # for i, lc in enumerate(lcs):
  #   # get all the time points that aren't already in this light curve
  #   new_tps = np.setdiff1d(union_tp, lc[:,0])
  #   new_tps = np.expand_dims(new_tps, axis=1)
  #   # add columns of zeros to the new time points for the flux vals and error
  #   new_tps = np.append(new_tps, np.zeros((len(new_tps), 2)), axis=1)
  #   lc = np.append(lc, new_tps, axis=0) # add the new rows of time points
  #   # sort by time along the rows
  #   sorted_indexes = lc[:,0].argsort()
  #   new_lcs[i] = lc[sorted_indexes]

    #return new_lcs



def main():
    pass
    
    
if __name__ == "__main__":
    main()





