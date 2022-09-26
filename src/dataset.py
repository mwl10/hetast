import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy import signal
import torch

class DataSet:
    def __init__(self, min_length=3, start_col=0, sep=','):
        """
        initializes some important aspects of the dataset
        
        parameters:
            min_length
            start_col
            sep
            
        """
        ############################
        self.min_length = min_length
        self.start_col = start_col
        self.sep = sep
        ############################
        
        self.valid_files_df = pd.DataFrame()
        self.bands = []
        
    def chop_lcs(self, std_threshold=1):
        ranges = [np.ptp(lc[:,0]) for object_lcs in self.dataset for lc in object_lcs]
        std_ranges = np.std(ranges)
        mean_ranges = np.mean(ranges)
        for i, object_lcs in enumerate(self.dataset):
            for j, lc in enumerate(object_lcs):
                num_splits = int(ranges[i*j] / (mean_ranges))#std_threshold * std_ranges))
                if num_splits <= 1:
                    continue
                else:
                    split_pt = np.where(lc[:,0] > (std_threshold * std_ranges))[0][0]
                    self.dataset[i][j] = lc[:split_pt]
                
    def preprocess(self):
        self.to_numpy()
        self.normalize()
        self.prune_graham()
        self.chop_lcs(std_threshold=1)
        self.formatting()
        self.set_union_tp()
        logging.info(self.dataset.shape)
     
    
    def to_numpy(self):
        
        dataset = []
        files_array = self.valid_files_df.to_numpy()
        for object_files in files_array:
            object_lcs = []
            for band_file in object_files:
                if not pd.isna(band_file):
                    lc = pd.read_csv(band_file, sep=self.sep)
                    lc = lc.to_numpy()[:, self.start_col:self.start_col+3]
                    lc = lc[lc[:,0].argsort()]
                    lc = lc.astype(np.float32)
                    
                else:
                    lc = np.zeros((self.min_length,3)).astype(np.float32)
                object_lcs.append(lc)
            dataset.append(object_lcs)
        self.dataset = dataset
        
        
    def normalize(self): 
        """
        This function does not so much normalize the dataset as shift it. Scaling the mag/flux values takes away from important amplitude information in differentiating
        objects as well as bands. The positional encoder would suffer if the time values were scaled in any way. Therefore we merely subtract the mean in both 
        dimensions of the light curve. Scaling values is usually to ensure that dataset features are in the same range, but we don't have different features here so long as all
        light curves in the dataset have the same units. 
        
        """
        self.unnormalized_data = self.dataset.copy()
        
        for object_lcs in self.dataset:
            for lc in object_lcs:
                if lc[:,1].sum() == 0: # means no light curve for this band/object
                    continue
               # lc[:,0] = lc[:,0] - np.mean(lc[:,0]) #lc[0,0] 
                lc[:,0] = lc[:,0] - lc[0,0] 
                lc[:,1] = lc[:,1] - np.mean(lc[:,1])
#                 mean = np.mean(lc[:,1])
#                 if y_by_range:
#                     range_y = np.max(lc[:,1]) - np.min(lc[:,1])
#                     lc[:,1] = (lc[:,1] - mean) / range_y
#                     lc[:,2] = lc[:,2] / y_mean_std[i,1] # normalize errors as well 
#                 else:
#                     std = np.std(lc[:,1])
#                     lc[:,1] = (lc[:,1] - mean) / std
#                     lc[:,2] = lc[:,2] / std # normalize errors as well    





    def prune_graham(self, med_filt=3, res_std=True, std_threshold=3, mag_threshold=0.25, plot=False, index=100):
        """
        
        prunes outliers by the given method in Graham et al. 2015
        
        that is, appling a (3 pt) median filter to the data to remove all outlier points that deviate 
        more than a given number std deviations (3) or a given magnitude threshold (0.25) from a quintic polynomial 
        fit to the light curve, and if more than ten percent are removed, iteratively increasing the clipping threshold 
        until a maximum of 10 percent are
        
        parameters:
            med_filt       (int)      --> number of points for the median filter 
            res_std        (boolean)  --> if true, use the std_threshold not the mag_threshold
            std_threshold  (float)    --> sets the initial clipping threshold by a residual std value of the quintic polynomial fit
            mag_threshold  (float)    --> sets the initial clipping threshold by a magnitude value
            plot           (boolean)  --> if you want to plot the light curve & the outliers of its quintic polynomial fit 
            index          (int)      --> which light curve index you'd like to plot if plot==True
            
        """
        
        for i, object_lcs in enumerate(self.dataset):
            for j, lc in enumerate(object_lcs):
                if lc[:,1].sum() == 0: ## skip if there's no light curve for this band
                    continue
                    
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
                
                
    def prune_outliers(self, std_threshold=100):
        """
        removes all points further than a given number of std deviations in the dataset
        
        """
        for i, object_lcs in enumerate(self.dataset):
            for j, lc in enumerate(object_lcs):
                if lc[:,1].sum() == 0: ## skip if theres no light curve for this band
                    break
                y = lc[:,1]
                y_std = np.std(y)
                y_mean = np.mean(y)
                # can do this with quantiles?
                
                outliers = np.where((y > (y_mean + y_std*std_threshold)) | \
                                           (y < (y_mean - y_std*std_threshold)))[0]
                
                self.dataset[i][j] = np.delete(lc, outliers, axis=0)
                
                
        
    def set_epoch_counts(self, sep_std=2, plot=False, index=1, figsize=(15,15)):
        ## count the epochs per lc
        epoch_counts = np.zeros((len(self.dataset), len(self.bands)))
        if plot==True:
            fig, ax = plt.subplots(len(self.bands),1, figsize=figsize, squeeze=False)
        for i, object_lcs in enumerate(self.dataset):
            for j, lc in enumerate(object_lcs):
                if lc[:,1].sum() == 0:
                    epochs.append(0)
                    continue
                dt = lc[1:,0] - lc[:-1,0]
                dt_std = np.std(dt)
                dt_mean = np.mean(dt)
                seps = np.where(dt > (dt_mean + dt_std*sep_std))[0]
                if plot == True and i == index:
                    ax[j][0].scatter(lc[:,0], lc[:,1], label='observed points')
                    for sep in seps:
                        ax[j][0].axvline(x=(lc[sep, 0] + lc[sep+1,0])/2, linestyle=':', label='epoch separation')
                num_epochs = len(seps) + 1
                epoch_counts[i,j] = num_epochs
        self.epoch_counts = epoch_counts
        if plot == True:
            lines_labels = ax[0][0].get_legend_handles_labels()
            lines,labels = lines_labels[0], lines_labels[1]
            fig.legend(lines[:2], labels[:2], bbox_to_anchor=(0.12, 0.92), loc='upper left')
    
    
    def set_data_obj(self, batch_size=8, split=0.8, shuffle=False):
        #############################################################
        # keep a consistent shuffle for unprocessing the light curves
        #############################################################
        
        if shuffle==True:
            shuffle = np.random.permutation(len(lcs.dataset)) 
            self.dataset = self.dataset[shuffle]
            self.unnormalized_data = [self.unnormalized_data[i] for i in shuffle]
            self.valid_files_df = self.valid_files_df.reindex(shuffle)
        
        #######################################################################################################
        # validation and training set can be the same because light curves are conditioned on differing subsamples 
        ########################################################################################################
        splindex = int(np.floor(split*len(self.dataset)))
        training, test = np.split(self.dataset, [splindex])
        _, valid = np.split(training, [splindex])

        self.split_index = splindex # keep this for deprocessing too
        
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
    
    
    
    
    def add_band(self, band, folder, sep=','): 
        """
        
        when we add a band via a folder filled with light_curve files, a dataframe keeps track of all the 
        
        indexes the new stellar objects 
        with their according files so that when another band is added the same dataframe can be used 
        
        """
        if os.path.isdir(folder):
            
            valid_counter = 0
            files = [os.path.join(folder, file) for file in os.listdir(folder)]
            for file in files:
                obj_name = file.split('/')[-1].split('_')[0]
                try:
                    lc = pd.read_csv(file, sep=sep)
                    if len(lc) > self.min_length:
                        valid_counter += 1
                        self.valid_files_df.loc[obj_name, band] = file
                except Exception:
                    pass
    
            logging.info(f'validated {valid_counter} files out of {len(files)} for {band=}')
            if valid_counter == 0:
                raise Exception(f"No readable files in {folder=}")
            else:
                self.bands.append(band)
    

    
    
    def formatting(self):
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
                
                ####################################
                # removing duplicates 
                ####################################
                _, unique = np.unique(lc[:, 0], return_index=True)
                lc = lc[unique]
                
                
                ########################################################
                # need to reformat the observations relative to union_tp 
                ########################################################
                new_y = np.zeros_like(union_tp)
                new_yerr = np.zeros_like(union_tp)
                subsampled_mask = np.zeros_like(union_tp)
                recon_mask = np.zeros_like(union_tp)
                
                mask = np.isin(union_tp, lc[:,0])
                indexes = np.nonzero(mask)[0]
                new_y[indexes] = lc[:,1]
                new_yerr[indexes] = lc[:,2]
                
                #####################################################
                # set the new time series that is correctly formatted 
                #####################################################
                formatted_lc = np.array([union_tp, new_y, new_yerr]).T
                self.dataset[i][j] = formatted_lc
                
        ##################################################################
        # set the no longer ragged multivariate dataset as one numpy array
        ##################################################################
        for i, object_lcs in enumerate(self.dataset):
            example = np.concatenate(([lc[np.newaxis] for lc in object_lcs]), axis=0)
            zs_to_append = len(max_union_tp) - example.shape[1]
            example = np.append(example, np.zeros((example.shape[0], zs_to_append, example.shape[2])), axis=1)
            self.dataset[i] = example
            
        self.dataset = np.array(self.dataset).astype(np.float32)

        

    def set_union_tp(self):
        """
        calcluates an array of the union of all the time points across the dataset &
        sets it as self.union_x
        """
        self.union_tp = np.unique(self.dataset[:,:,:,0].flatten()).astype('float32')
        logging.info(f'created union_tp attribute of length {len(self.union_tp)}')
    
    def set_target_x(self, num_points=40, forecast=False, forecast_frac=1.2):
        """
        sets the target time values we might want to interpolate the light curve to. 
        
        
        parameters:
            -----optional-----
            num_points     (int)      --> how many points do we want 
            forecast       (boolean)  --> do we want to forecast?
            forecast_frac  (float)    --> what percentage of the light curve relative to 
                                          its length should we forecast? 
        """
        time = self.dataset[:,:,:,0]
        zs_to_append = time.shape[2] - num_points
        self.target_x = np.zeros_like(time)
        
        for i, object_lcs_time in enumerate(time):
            for j, lc_time in enumerate(object_lcs_time):
                max_time = np.max(lc_time)
                min_time = lc_time[0]
                if forecast:
                    max_time = forecast_frac * max_time
                target_x = np.arange(min_time,max_time, (max_time - min_time)/num_points)
                target_x = np.append(target_x, np.zeros((zs_to_append)), axis=0)[:time.shape[2]]
                self.target_x[i,j] = target_x
            


    def set_intrinsic_vars(self):
        intrinsic_vars = np.zeros((len(self.dataset),len(self.bands)))
        for i, object_lcs in enumerate(self.dataset):
            for j, lc in enumerate(object_lcs):
                dev_from_mean = (lc[:,1] - np.mean(lc[:,1]))
                avg_sq_dev_from_mean = np.matmul(dev_from_mean, dev_from_mean) * (1 / (len(lc) - 1))
                avg_sq_err = np.matmul(lc[:,2], lc[:,2]) / len(lc)
                intrinsic_var = avg_sq_dev_from_mean - avg_sq_err
                intrinsic_vars[i,j] = intrinsic_var
        self.intrinsic_vars = intrinsic_vars
    
    def set_mean_mags(self):
        mean_mags = np.zeros((len(self.dataset), len(self.bands)))
        mean_mags = [np.mean(lc[:,1]) for object_lcs in self.dataset for lc in object_lcs]
        self.mean_mags = np.array(mean_mags)