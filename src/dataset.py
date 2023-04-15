import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import copy

class DataSet:
    def __init__(self, name, start_col=1, sep=','):
        """
        params:
            start_col: first column where time exists, then mag, then magerr -> lots of odd bugs if left unset
            sep: delimiter for files in the dataset
        """
        ############################
        self.name = name
        self.start_col = start_col 
        self.sep = sep
        ############################
        self.valid_files_df = pd.DataFrame()
        self.bands = [] 
        
        
    def set_data_obj(self, batch_size=8, split=0.90, shuffle=False):
        #############################################################
        # keep a consistent shuffle for unprocessing the light curves
        #############################################################
        if shuffle==True:
            shuffle = np.random.permutation(len(self.dataset)) 
            self.dataset = self.dataset[shuffle]
            self.unnormalized_data = [self.unnormalized_data[i] for i in shuffle]
            self.valid_files_df = self.valid_files_df.reindex(self.valid_files_df.index[shuffle])
        #######################################################################################################
        # validation and training set can be the same because light curves are conditioned on differing subsamples 
        ########################################################################################################
        splindex = int(np.floor(split*len(self.dataset)))
        training, test = np.split(self.dataset, [splindex])
        valid_splindex = int(np.floor(0.8*len(training)))
        _, valid = np.split(training, [valid_splindex])
        self.test_split_index = splindex # keep this for denormalizing later
        self.valid_split_index = valid_splindex
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
     
    def correct_z(self, catalog='catalogs/sample_cat'):
        """
        get z from catalog files, 
        match to object, then t_rest = t_obs/(1+z)
        """
        rm_objs_z = {'Mrk876':0.1211,'Mrk817':0.15838,'3C273':0.158339,'H2106-099': 0.026515009927301107, \
                  '3C120': 0.03357276087555472, 'NGC5548': 0.01627, 'Mrk142': 0.04459, 'NGC2617': 0.014325776576655791, \
                  'MCG+08-11-011': 0.02004386648045696}
        catalog = os.path.join(os.path.dirname(self.name), catalog)
        if os.path.isfile(catalog) and self.name.lower().find('ztf') > 0:
            sample_df = pd.read_csv(catalog)
            sample_df = sample_df.set_index('SDSS')
            # want z array relative to order of dataset/valid_files_df
            for i,obj in enumerate(self.valid_files_df.index):
                try:
                    z = sample_df.loc[obj]['z']
                except Exception:
                    # obj is separate rm obj
                    z = rm_objs_z[obj]
                for lc in self.dataset[i]:
                    if lc[:,1].any():
                        lc[:,0] = lc[:,0] * (1 / (1+z))

                                   
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
            min_t = 1000000
            for lc in object_lcs:
                if lc[:,1].any():
                    if lc[0,0] < min_t:
                        min_t = lc[0,0]
            for lc in object_lcs:
                if lc[:,1].any():
                    lc[:,0] = lc[:,0] - min_t
                    #lc[:,0] = lc[:,0] / 365
                    #lc[:,0] = lc[:,0] / np.max(lc[:,0])
                    lc[:,1] = (lc[:,1] - np.mean(lc[:,1])) / np.std(lc[:,1])  
                    lc[:,2] = lc[:,2] / np.std(lc[:,1])
                 

                
    def prune_graham(self, med_filt=3, res_std=True, std_threshold=3, mag_threshold=0.25, plot=False, index=2):
        """
        prunes outliers by the given method in Graham et al. 2015
        
        that is, appling a (3 pt) median filter to the data to remove all outlier points that deviate 
        more than a given number std deviations (3) or a given magnitude threshold (0.25) from a quintic polynomial 
        fit to the light curve, and if more than ten percent are removed, iteratively increasing the clipping threshold 
        until a maximum of 10 percent are
        
        parameters:
            med_filt       (int)    --> number of points for the median filter 
            res_std        (bool)   --> if true, use the std_threshold not the mag_threshold
            std_threshold  (float)  --> sets the initial clipping threshold by a residual 
                                        std value of the quintic polynomial fit
            mag_threshold  (float)  --> sets the initial clipping threshold by a magnitude value
            plot           (bool)   --> if you want to plot the light curve & the outliers of its quintic polynomial fit 
            index          (int)    --> which light curve index you'd like to plot if plot==True
            
        """
        for i, object_lcs in enumerate(self.dataset):
            for j, lc in enumerate(object_lcs):
                lc[:,1] = signal.medfilt(lc[:,1], kernel_size=med_filt)
                quintic_fit = np.polyfit(lc[:,0], lc[:,1], deg=5)
                quintic_y = np.array([lc[:,0]**5, lc[:,0]**4, lc[:,0]**3, lc[:,0]**2, \
                                      lc[:,0], np.ones(len(lc))])
                quintic_y = np.matmul(quintic_y.T, quintic_fit)
                dev = np.abs(lc[:,1] - quintic_y)
                if res_std:
                    res_std = np.sqrt(np.mean(dev**2))
                    mag_threshold = std_threshold*res_std
                # increase mag_threshold of outliers if more than 10% are removed
                percentage=1.
                while(True):
                    outliers = np.where(dev >= mag_threshold)[0]
                    percentage = len(outliers) / len(lc)
                    if percentage > 0.1:
                        mag_threshold += 0.01
                    else:
                        break
                pruned_lc = np.delete(lc, outliers, axis=0)
                self.dataset[i][j] = pruned_lc
                
            if plot == True and i == index:
                plt.plot(lc[:,0], quintic_y)
                plt.scatter(lc[outliers,0], lc[outliers,1], c='r', marker='x')
                plt.scatter(pruned_lc[:,0], pruned_lc[:,1], c='b')    
                
                
    def resample_lcs(self, num_resamples=10):
        """
        sample from dist w/ mean 0, std as the light curves observational errors 
        and add these values to original light curves to create new examples 
        """
        new_samples = []
        for _ in range(num_resamples):
            for i, object_lcs in enumerate(self.dataset):
                new_sample = []
                for j, lc in enumerate(object_lcs):
                    new_lc = np.array([lc[:,0], lc[:,1] + np.random.normal(0,lc[:,2]), lc[:,2]]).T
                    new_sample.append(new_lc)
                new_samples.append(new_sample)
        self.dataset.extend(new_samples)
        
        
    def prune_outliers(self, std_threshold=10):
        """
        removes all points further than a given number of std deviations in the dataset
        """
        for i, object_lcs in enumerate(self.dataset):
            for j, lc in enumerate(object_lcs):
                y = lc[:,1]
                y_std = np.std(y)
                y_mean = np.mean(y)
                outliers = np.where((y > (y_mean + y_std*std_threshold)) | \
                                           (y < (y_mean - y_std*std_threshold)))[0]
                self.dataset[i][j] = np.delete(lc, outliers, axis=0)
                ## remove outliers with >= 1 mag error for ZTF
                if self.name.lower().find('ztf') > 0:
                    outliers = np.where(lc[:,2] >=1)[0]
                    self.dataset[i][j] = np.delete(lc, outliers, axis=0)
                    
                
    def add_band(self,band_folder): 
        """
        When we add a band via a folder filled with light curve files, a dataframe keeps track of all the 
        the new objects with their according files so that when another band 
        is added the same dataframe can be used. 
        
        parameters:
            band_folder
            
        side effects:
            --places file appropriately in self.valid_files_df
            --appends new band dimension to self.band
        """
    
        if os.path.isdir(band_folder):
            band = os.path.basename(band_folder)
            valid_counter = 0
            dataset = []
            files = [os.path.join(band_folder, file) for file in os.listdir(band_folder)]
            for file in files:
                if file.find('_') > 0:
                    obj_name = file.split('/')[-1].split('_')[0]
                else:
                    # if no underscore to separate obj name, take filename as is 
                    obj_name = "".join(file.split('/')[-1].split('.')[:-1])          
                valid_counter += 1
                self.valid_files_df.loc[obj_name, band] = file
            print(f'validated {valid_counter} files out of {len(files)} for {band=}')
            if valid_counter == 0:
                raise Exception(f"No readable files in {band_folder=}")
            else:
                self.bands.append(band)
         
        
    def chop_lcs(self, std_threshold=1):
        """
        cut light curves longer than std_thresholds beyond the mean of lengths,
        which is important for training if dataset has very right skewed ranges 
        
        parameters:
            std_threshold     (int)       threshold number of stds beyond mean of lengths in which observations are lopped off
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
                        
    
    def filter(self, min_length=1):
        """
        a bit ugly and hacky, but hopefully we've added zeros where a light curve is missing, 
        made sure that not all light curves across bands are empty
        and filtered light curves we do have by min length, excess var, and mean mag 
        
        for ZTF datasets remove lcs w/ 
            mean g band magnitude fainter than 20.6
            mean r fainter than 20.4,
            mean i fainter than 19.7,
            and brighter than 13.5 (to avoid saturated observations) 
        parameters   min_length     (int)      -->    minimum number of observations allowed 
            """
        self.valid_files_df = self.valid_files_df.dropna()
        dataset = []
        drops = []
        for i,object_files in enumerate(self.valid_files_df.values):
            object_lcs = []
            zero_count = 0 # i.e. light curve file w/ no observations
            for j,band_file in enumerate(object_files):
                try: 
                    lc = pd.read_csv(band_file, sep=self.sep).to_numpy()
                except Exception:
                    object_lcs.append(np.zeros((1,3)))
                    zero_count += 1
                    continue
                    
                ##### filtering ZTF error codes #########
                if self.name.lower().find('ztf') > 0:
                    try: # might be that its not original 8 columns from ztf api 
                        lc = lc[np.where(lc[:,4] == 0)[0]]
                    except Exception:
                        pass
                #########################################
                if len(lc) > min_length:
                    lc = lc[:, self.start_col:self.start_col+3]
                    lc = lc[lc[:,0].argsort()].astype(np.float32)
#                         excess_var = ((np.std(lc[:,1]) ** 2) - (np.mean(lc[:,2]) ** 2)) \
#                         / np.mean(lc[:,1])
                    mean_mag = np.mean(lc[:,1])
                    ### ZTF limiting mags: g = 20.8, r = 20.6, i = 19.9 mag
                    if self.name.lower().find('ztf') > 0: 
                        if self.bands[j] == 'r' and mean_mag <= 20.4 and mean_mag >= 13.5:   ###### filter r mags
                            object_lcs.append(lc)
                        elif self.bands[j] == 'i' and mean_mag <= 19.7 and mean_mag >= 13.5: ###### filter i mags
                            object_lcs.append(lc)
                        elif self.bands[j] == 'g' and mean_mag<= 20.6 and mean_mag >= 13.5:  ###### filter g mags 
                            object_lcs.append(lc)
                        else:
                            object_lcs.append(np.zeros((1,3)))
                            zero_count += 1
                    else:
                        object_lcs.append(lc)

                elif len(lc) == 0:
                    object_lcs.append(np.zeros((1,3)))
                    zero_count += 1      
                
            # make sure we have three bands, and not all of them are missing
            if (len(object_lcs) == len(self.bands)) and (zero_count != len(self.bands)):
                  dataset.append(object_lcs)
            else:
                drops.append(i)
                
        self.valid_files_df.drop(self.valid_files_df.index[drops], inplace=True)
        self.dataset = dataset
        
        
    
            
    def format(self):
        """
        
        
        """
        max_union_tp = []
        # iterating through each example
        for i, object_lcs in enumerate(self.dataset):
            # get all the time points across the dimensions
            union_tp = np.unique(np.hstack([lc[:,0] for lc in object_lcs]))
            # save the longest one to zero fill later on
            if len(union_tp) > len(max_union_tp):
                max_union_tp = union_tp
            # iterating through each dimension    
            for j, lc in enumerate(object_lcs):
                # removing duplicates 
                _, unique = np.unique(lc[:, 0], return_index=True)
                lc = lc[unique]
                # need to reformat the observations relative to union_tp 
                new_y = np.zeros_like(union_tp)
                new_yerr = np.zeros_like(union_tp)
                mask = np.isin(union_tp, lc[:,0])
                indexes = np.nonzero(mask)[0]
                new_y[indexes] = lc[:,1]
                new_yerr[indexes] = lc[:,2]
                # set the new time series that is correctly formatted 
                formatted_lc = np.array([union_tp, new_y, new_yerr]).T
                self.dataset[i][j] = formatted_lc         
        # set the no longer ragged multivariate dataset as one numpy array
        for i, object_lcs in enumerate(self.dataset):
            example = np.concatenate(([lc[np.newaxis] for lc in object_lcs]), axis=0)
            zs_to_append = len(max_union_tp) - example.shape[1]
            example = np.append(example, np.zeros((example.shape[0], zs_to_append, example.shape[2])), axis=1)
            self.dataset[i] = example
        self.dataset = np.array(self.dataset)
    
        self.dataset = self.dataset.astype(np.float32)
        
             
    def set_union_tp(self, uniform=False, n=1000):
        """
        calcluates an array of the union of all the time points across the dataset unless uniform==True,
        in that case set union_tp to n uniformly spaced points between the maximum and minimum observed time
        across the dataset
        
        parameters:
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
        self.union_tp = self.union_tp.astype('float32')
        print(f'created union_tp attribute of length {len(self.union_tp)}')
    
    
    def set_target_x(self, n=40, uniform=False, r=1500):
        """
        sets the target time values we want to interpolate the light curve to. If uniform is false,
        we set n points between the min and max t for each light curve. 
        If its true, we set n evenly spaced points between 0 and r. 
        
        parameters:
        
            n       (int)      --> how many points do we want? 
            uniform (boolean)  --> should we do uniform?
            r       (int)      --> if uniform, whats the range of points? 
            
        side effects:
            - sets self.target_x as a numpy array with dimensions as (len(self.dataset), len(self.bands), n)
        """
       
        if uniform==True:
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
    
    
    def set_sigma_nxs(self):
        """
        setting the normalized excess variance, as per the definition in Vaughan et al. 2003.
        
        side effects:
            sets self.sigma_nxs as a numpy array with dimensions as (len(self.dataset), len(self.bands)) 
        """
        sigma_nxs = np.zeros((len(self.dataset),len(self.bands)))
        for i, object_lcs in enumerate(self.dataset):
            for j, lc in enumerate(object_lcs):
                if len(lc) > 1:
                    sigma_nxs[i,j] = \
                    ((1/(len(lc)-1)) * ((lc[:,1] - np.mean(lc[:,1]))**2).sum() - np.mean(lc[:,2]**2)) \
                    / np.mean(lc[:,1])**2
                else:
                    sigma_nxs[i,j] = 0
        self.sigma_nxs =  sigma_nxs      
       
    
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
