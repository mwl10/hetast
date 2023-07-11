import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import copy

class DataSet:
    def __init__(self, folder):
        if os.path.isdir(folder):
            self.folder = folder
        else:
            raise Exception(f"{folder} is not a directory") 
        self.valid_files_df = pd.DataFrame() # keep track of the files across dimensions
        self.bands = []                      # array of band names
        self.dataset = []                    # python array w/ragged dataset, ultimately formatted to np array. 
        self.unnormalized_data = []          # keep unnormalized data for deprocessing
        self.union_tp = np.array([])         # union_tp for hetvae's intensity pathway
        self.target_x = np.array([])         # for making interpolations later on
        self.sigma_nxs = np.array([])        # normalized excess variance
        self.mean_mag = np.array([])         # mean magnitude
        self.med_cadence = np.array([])      # median cadence
     
    
    def files_to_df(self):
        band_folders = [os.path.join(self.folder,bf) for bf in os.listdir(self.folder)]
        if band_folders and all([os.path.isdir(bf) for bf in band_folders]):
            for bf in band_folders:
                band = os.path.basename(bf)
                files = [os.path.join(bf, file) for file in os.listdir(bf)]
                print(f'found {len(files)} for {band=}')
                if len(files) > 0: self.bands.append(band)    
                for file in files:
                    if file.find('_') > 0:
                        obj_name = os.path.basename(file).split('_')[0]
                    else:
                        # if no underscore to separate obj name, take filename as is 
                        obj_name = os.path.splitext(os.path.basename(file))[0]         
                    self.valid_files_df.loc[obj_name, band] = file
        else:
            raise Exception('Empty data directory or files found where band folders should be.')
    
    
    # just read and add zeros in place where necessary 
    def read(self, sep=','):
        """
        args
        sep: delimiter for files in the dataset
        """
        for i,object_files in enumerate(self.valid_files_df.values):
            object_lcs = []
            for j,band_file in enumerate(object_files): 
                try: # to read
                    lc = pd.read_csv(band_file, sep=sep).to_numpy()
                except Exception:
                    lc = np.zeros((1,3)) # placeholder 
                object_lcs.append(lc)
            self.dataset.append(object_lcs) 
            
    
    def prune(self,start_col=1, min_length=1, keep_missing=True, std_threshold=10):   
        replace = np.zeros((1,3)) # placeholder
        drops = []
        for i,object_lcs in enumerate(self.dataset):
            replace_count = 0
            for j,lc in enumerate(object_lcs):
                '''filter lcs w/ no points'''
                if not lc.any(): 
                    replace_count+=1 
                    self.dataset[i][j] = replace
                    continue 
                '''get columns for t,mag,magerr'''
                
                try:
                    lc = lc[:, start_col:start_col+3].astype(np.float32)
                    assert lc.shape[1] == 3
                except Exception:
                    raise Exception('double check start_col value')
                '''rm outliers'''
                y, y_mean, y_std = lc[:,1], np.mean(lc[:,1]), np.std(lc[:,1])
                outliers = np.where((y > (y_mean + y_std*std_threshold)) | \
                                           (y < (y_mean - y_std*std_threshold)))[0]
                lc = np.delete(lc, outliers, axis=0)
                if len(lc) == 0: 
                    replace_count+=1 
                    self.dataset[i][j] = replace
                    continue
                '''average duplicate points'''
                while len(lc) != len(np.unique(lc[:,0])):
                    lc = np.array([lc[np.where(i==lc[:,0])[0]].mean(0) for i in np.unique(lc[:,0])])
                '''min length filter'''
                if len(lc) < min_length:
                    replace_count+=1 
                    self.dataset[i][j] = replace
                    continue
                self.dataset[i][j] = lc
            if (replace_count >= len(self.bands)) or (not keep_missing and replace_count > 0): 
                drops.append(i)     
        self.valid_files_df.drop(self.valid_files_df.index[drops], inplace=True)
        self.dataset = [self.dataset[i] for i in range(len(self.dataset)) if i not in drops]
    
 
    def chop_lcs(self, std_threshold=1):
        """
        cut light curves longer than std_thresholds beyond the mean of lengths,
        which is important for training if dataset has very right skewed ranges 
        
        args
        ------
        std_threshold     (int)       threshold number of stds beyond mean of lengths in which 
                                      observations are lopped off
        
        """
        ranges = [np.ptp(lc[:,0]) for object_lcs in self.dataset for lc in object_lcs]
        std_range = np.std(ranges)
        mean_range = np.mean(ranges)
        for i, object_lcs in enumerate(self.dataset):
            for j, lc in enumerate(object_lcs):
                #num_splits = int(ranges[i*j] / (mean_range))#std_threshold * std_ranges))
                split_threshold = lc[:,0].min() + (mean_range + (std_threshold * std_range)) 
                split_pt = np.where(lc[:,0] > split_threshold)[0]
                if np.any(split_pt):
                    self.dataset[i][j] = lc[:split_pt[0]] # could add this as new example            
   
    
    def resample_lcs(self, num_resamples=10, seed=2):
        """
        sample from dist w/ mean 0, std as the light curves observational errors 
        and add these values to original light curves to create new examples 
        """
        np.random.seed(seed=seed)  
        torch.manual_seed(seed=seed)
        new_samples = []
        for _ in range(num_resamples):
            for i, object_lcs in enumerate(self.dataset):
                new_sample = []
                for j, lc in enumerate(object_lcs):
                    new_lc = np.array([lc[:,0], lc[:,1] + np.random.normal(0,lc[:,2]), lc[:,2]]).T
                    new_sample.append(new_lc)
                new_samples.append(new_sample)
        self.dataset.extend(new_samples)
    
    
    def normalize(self): 
        """
        make time start at 0,
        normalize y to have mean 0, std 1
        normalize yerr as yerr/std(y),
        skip if light curve is missing for a particular dimension  
        """
        self.unnormalized_data = copy.deepcopy(self.dataset)
        for object_lcs in self.dataset:
            ### subtract the earliest time value between all bands 
            min_t = float('inf')
            for lc in object_lcs:
                if lc[:,1].any():
                    if lc[0,0] < min_t:
                        min_t = lc[0,0]            
            for lc in object_lcs:
                if lc[:,1].any():
                    lc[:,0] = lc[:,0] - min_t
                    #lc[:,0] = lc[:,0] / 365
                    #lc[:,0] = lc[:,0] / np.max(lc[:,0])
                    lc[:,1] = lc[:,1] - np.mean(lc[:,1])
                    if np.std(lc[:,1]) != 0: lc[:,1] = lc[:,1] / np.std(lc[:,1])  
                    lc[:,2] = lc[:,2] / np.std(lc[:,1])
                    
                    
    def format_(self):
        union_tp_ex = [np.unique(np.hstack([lc[:,0] for lc in object_lcs])) for object_lcs in self.dataset]
        union_tp_max = np.max([len(utp) for utp in union_tp_ex]) # to zero fill w/ 
        for i, object_lcs in enumerate(self.dataset):
            for j, lc in enumerate(object_lcs):
                # need to reformat the observations relative to union_tp 
                new_y = np.zeros_like(union_tp_ex[i])
                new_yerr = np.zeros_like(union_tp_ex[i])
                mask = np.isin(union_tp_ex[i], lc[:,0]) # where are the timepoints relative to union_
                indexes = np.nonzero(mask)[0]
                new_y[indexes] = lc[:,1]
                new_yerr[indexes] = lc[:,2]
                # set the new time series that is correctly formatted 
                new_lc = np.array([union_tp_ex[i], new_y, new_yerr]).T
                new_lc = np.append(new_lc, np.zeros(((union_tp_max - new_lc.shape[0]),3)),axis=0)
                self.dataset[i][j] = new_lc
        self.dataset = np.array(self.dataset,dtype=np.float32)
   

    def set_union_tp(self, uniform=False, n=1000):
        """
        calcluates an array of the union of all the time points across the dataset unless uniform==True,
        in that case set union_tp to n uniformly spaced points between the maximum and minimum observed time
        across the dataset
        args:
            uniform       (boolean)   --> uniformly spread?
            n             (int)       --> number of uniformly spread pts
        side effects
            - sets self.union_tp 
            - prints length of union tp
        """
        self.union_tp = np.unique(self.dataset[:,:,:,0].flatten()) 
        print('max time: ', np.max(self.union_tp))
        if uniform: 
            step = np.ptp(self.union_tp) / n 
            self.union_tp = np.arange(np.min(self.union_tp), np.max(self.union_tp), step) 
        self.union_tp = self.union_tp.astype(np.float32)
        print(f'created union_tp attribute of length {len(self.union_tp)}')
      
    
    def set_target_x(self, n=40, uniform=False, r=1500):
        """
        sets the target time values we want to interpolate the light curve to. If uniform is false,
        we set n points between the min and max t for each light curve. 
        If its true, we set n evenly spaced points between 0 and r. 
        args:
        
            n       (int)      --> how many points do we want? 
            uniform (boolean)  --> should we do uniform?
            r       (int)      --> if uniform, whats the range of points? 
            
        side effects:
            - sets self.target_x as a numpy array with dimensions as (len(self.dataset), len(self.bands), n)
        """
        if uniform:
            target_x = np.arange(0,r,step=r/n)[:n]
            # format it... want (num exs x num bands x num tps)
            self.target_x = target_x[np.newaxis,len(self.bands)].repeat(self.dataset.shape[0],axis=0)    
        else:
            self.target_x = np.zeros((self.dataset.shape[0],self.dataset.shape[1],n))
            for i, obj_lcs in enumerate(self.dataset):
                for j,lc in enumerate(obj_lcs):
                    max_t = np.max(lc[:,0])
                    min_t = lc[0,0]
                    if max_t == 0:
                        max_t=1500              
                    tx = np.arange(min_t,max_t, ((max_t - min_t)/n))
                    self.target_x[i,j] = tx[:n]     
        self.target_x = self.target_x.astype(np.float32)
    
    
    def set_data_obj(self, batch_size=8, test_split=0.1, shuffle=False, seed=2):
        if shuffle: # keep a consistent shuffle for unprocessing
            np.random.seed(seed=seed) 
            torch.manual_seed(seed=seed)
            shuf = np.random.permutation(len(self.dataset)) 
            self.dataset = self.dataset[shuf]
            try: ## incase we don't normalize the data, unnorm_data wont be set
                self.unnormalized_data = [self.unnormalized_data[i] for i in shuf]
            except Exception:
                pass 
            self.valid_files_df = self.valid_files_df.reindex(self.valid_files_df.index[shuf])
        # val and train set can be the same because light curves are conditioned 
        # on differing subsamples
        splindex = int(np.floor((1-test_split)*len(self.dataset)))
        training, test = np.split(self.dataset, [splindex])
        valid_splindex = int(np.floor(0.8*len(training))) 
        _, valid = np.split(training, [valid_splindex])
        self.test_split_index = splindex # keep this for denormalizing later
        print(f'train size: {len(training)}, valid size: {len(valid)}, test size: {len(test)}')
        train_loader = torch.utils.data.DataLoader(training, batch_size=batch_size)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)
        union_tp = torch.Tensor(self.union_tp)  
        
        self.data_obj = {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "valid_loader": valid_loader,
            'union_tp': union_tp,
            "input_dim": len(self.bands),
        }
        
        
    def set_sigma_xs(self):
        """
        setting the intrinsic variance, as per the definition in Tachibana et al. 2020.
        side effects:
            sets self.sigma_xs as a numpy array with dimensions as (len(self.dataset), len(self.bands)) 
        """
        sigma_xs = np.zeros((len(self.dataset),len(self.bands)))
        for i, object_lcs in enumerate(self.dataset):
            for j, lc in enumerate(object_lcs):
                if lc.any() > 0:
                    sigma_xs[i,j] = ((1/(len(lc)-1))*(((lc[:,1] - np.mean(lc[:,1]))**2).sum()))-((1/len(lc))*(lc[:,2]**2).sum())
                    
#                     sigma_nxs[i,j] = ((1/(len(lc)-1)) * ((lc[:,1] - np.mean(lc[:,1]))**2).sum() - np.mean(lc[:,2]**2)) / np.mean(lc[:,1])**2
                else:
                    sigma_xs[i,j] = 0
        self.sigma_xs =  sigma_xs      
       
    def set_mean_mag(self):
        """
        setting the mean magnitude of the light curve
        side effects:
            sets self.mean_mag as a numpy array with dimensions as (len(self.dataset), len(self.bands)) 
        """
        mean_mag = [np.mean(lc[:,1]) for object_lcs in self.dataset for lc in object_lcs]
        self.mean_mag = np.array(mean_mag).reshape(-1,len(self.bands))
        
    def set_med_cadence(self):
        """
        setting the median cadence of the light curve
        side effects:
            sets self.med_cadence as a numpy array with dimensions as (len(self.dataset), len(self.bands)) 
        """
        fn = lambda lc: np.median(lc[1:,0]-lc[:-1,0])
        med_cadence = [fn(lc) for object_lcs in self.dataset for lc in object_lcs]
        self.med_cadence = np.array(med_cadence).reshape(-1,len(self.bands))

        
        
class ZtfDataSet(DataSet):
    
    def __init__(self, folder):
        super().__init__(folder)
        
    def prune(self,start_col=1, min_length=2, keep_missing=True, std_threshold=10):   
        replace = np.zeros((1,3)) # placeholder
        drops = []
        for i,object_lcs in enumerate(self.dataset):
            replace_count = 0
            for j,lc in enumerate(object_lcs):
                '''filter ztf error codes'''    
                try: 
                    lc = lc[np.where(lc[:,4] == 0)[0]]
                except IndexError:
                    pass
                '''filter lcs w/ no points'''
                if not lc.any(): 
                    replace_count+=1 
                    self.dataset[i][j] = replace
                    continue 
                '''get columns for t,mag,magerr'''
                try:
                    lc = lc[:, start_col:start_col+3].astype(np.float32)
                    assert lc.shape[1] == 3
                except Exception:
                    raise Exception('double check start_col value')
                '''rm outliers & ztf outliers'''
                y, y_mean, y_std = lc[:,1], np.mean(lc[:,1]), np.std(lc[:,1])
                outliers = np.where((y > (y_mean + y_std*std_threshold)) | \
                                           (y < (y_mean - y_std*std_threshold)))[0]
                ztf_outliers = np.where(lc[:,2] >=1)[0]
                lc = np.delete(lc, np.concatenate((outliers,ztf_outliers)), axis=0)
                
                if len(lc) == 0: 
                    replace_count+=1 
                    self.dataset[i][j] = replace
                    continue
                '''average duplicate points'''
                while len(lc) != len(np.unique(lc[:,0])):
                    lc = np.array([lc[np.where(i==lc[:,0])[0]].mean(0) for i in np.unique(lc[:,0])])
                '''min length filter'''
                if len(lc) < min_length:
                    replace_count+=1 
                    self.dataset[i][j] = replace
                    continue
                '''ztf filtering by limiting mag / saturated obs'''   
                mean_mag = np.mean(lc[:,1])
                if (self.bands[j] == 'r' and mean_mag > 20.4 and mean_mag < 13.5) \
                or (self.bands[j] == 'i' and mean_mag > 19.7 and mean_mag < 13.5) \
                or (self.bands[j] == 'g' and mean_mag > 20.6 and mean_mag < 13.5):
                    replace_count+=1 
                    self.dataset[i][j] = replace
                    continue
                self.dataset[i][j] = lc
            if (replace_count >= len(self.bands)) or \
            (not keep_missing and replace_count > 0): 
                drops.append(i)     
        self.valid_files_df.drop(self.valid_files_df.index[drops], inplace=True)
        self.dataset = [self.dataset[i] for i in range(len(self.dataset)) if i not in drops]
        