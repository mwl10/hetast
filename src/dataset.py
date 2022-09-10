import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


############    each function mainly acts as a nice builder for the dataset instead of using a big constructor, and the order of their use matters

class DataSet:

    def add_files(self, files):
        self.files = files
        return self
    
    def flux_to_mag():
        pass
    
    def mag_to_flux():
        pass
    
    def files_to_numpy(self, sep='\t', start_col=0, minimum=50):
        """
        
        reads files from self.files and converts them to individual numpy arrays
        
        parameters:
            sep        (str)  --> sep for pd.read_csv(), i.e. '/t' or ','
            start_col  (int)  --> index of column containing the time values, the following two must be flux/mag & fluxerr/magerr
            minimum    (int)  --> a light curve won't get read into the dataset if it has less than this number of points
        
        """
        dataset = []
        print(f'read {len(self.files)} files')
        read_err_counter = 0
        length_err_counter = 0
        good_counter = 0
        bad_files = []
        for i, file in enumerate(self.files):
            try:
                example = pd.read_csv(file, sep=sep)  
            except:
                read_err_counter += 1
                bad_files.append(file)
                continue    
            example = example.to_numpy()
            example = example[:,start_col:start_col+3].astype(np.float32)
            example = example[example[:,0].argsort()]
            if (len(example) <= minimum): #or  (len(example) >= maximum):
                length_err_counter += 1
                bad_files.append(file)
                continue
            good_counter += 1
            dataset.append(example)
        self.dataset = dataset
        self.files = [file for file in self.files if file not in bad_files]
        print(f"{read_err_counter} files couldn't be read by pd, {length_err_counter} were too short, {good_counter} were ok")
        return self


    def handle_dups(self):
        """
        
        removes duplicate observations in the dataset, should average the duplicate values instead of dropping them in the future
        
        """
        for index, example in enumerate(self.dataset):
            unique, i = np.unique(example[:, 0], return_index=True)
            self.dataset[index] = example[i]
        return self


    def prune_outliers(self, std_threshold=3):
        """
        
        removes all points further than a given number of std deviations in the dataset
        
        """
        
        for i, example in enumerate(self.dataset):
            y = example[:,1]
            y_std = np.std(y)
            y_mean = np.mean(y)
            outlier_indexes = np.where((y > (y_mean + y_std*std_threshold)) | (y < (y_mean - y_std*std_threshold)))[0]
            #print(f'indexes of outliers to be pruned, if any: {outlier_indexes}')
            self.dataset[i] = np.delete(example, outlier_indexes, axis=0)
        return self


    def prune_graham(self, med_filt=3, res_std=True, std_threshold=3, mag_threshold=0.25, plot=False, index=100,):
        """
        
        prunes outliers by the given method in Graham et al. 2015
        
        that is, appling a (3 pt) median filter to the data to remove all outlier points that deviate more than a given number std deviations (3) 
        or a given magnitude threshold (0.25) from a quintic polynomial fit to the light curve, and if more than ten percent are removed, 
        iteratively increasing the clipping threshold until a maximum of 10 percent are
        
        parameters:
            med_filt       (int)      --> number of points for the median filter 
            res_std        (boolean)  --> if true, use the std_threshold not the mag_threshold
            std_threshold  (float)    --> sets the initial clipping threshold by a residual std value of the quintic polynomial fit
            mag_threshold  (float)    --> sets the initial clipping threshold by a magnitude value
            plot           (boolean)  --> if you want to plot the light curve & the outliers of its quintic polynomial fit 
            index          (int)      --> which light curve index you'd like to plot if plot==True
            
        """
        
        for i, example in enumerate(self.dataset):
            example[:,1] = signal.medfilt(example[:,1], kernel_size=med_filt)
            quintic_fit = np.polyfit(example[:,0], example[:,1], deg=5)
            quintic_y = np.array([example[:,0]**5, example[:,0] ** 4, example[:,0] ** 3, example[:,0] ** 2 , example[:,0], np.ones(len(example))])
            quintic_y = np.matmul(quintic_y.T, quintic_fit)
            quintic_y_std = np.std(quintic_y)
            dev = np.abs(example[:,1] - quintic_y)
            # residual std error 
            if res_std:
                res_std = np.sqrt(np.mean(dev**2))
                mag_threshold = std_threshold*res_std
               
            # increase mag_threshold of outliers if more than 10 percent are removed
            percentage = 1.
            while(True):
                outliers = np.where(dev >= mag_threshold)[0]
                percentage = len(outliers)/ len(example) # none past 10 percent 
                if percentage > .1:
                    mag_threshold += 0.01
                else:
                    break
            pruned_example = np.delete(example, outliers, axis=0)
            self.dataset[i] = pruned_example
            if plot==True and i == index:
                plt.plot(example[:,0], quintic_y)
                plt.scatter(example[outliers,0], example[outliers,1], c='r', marker='x')
                plt.scatter(pruned_example[:,0], pruned_example[:,1], c='b')
                plt.xlabel('MJD')
                plt.ylabel('mag')
        return self 


    def normalize(self, y_by_range=False): 
        """
        
        this function normalizes the dataset, particularly the flux/mag values. If we want to normalize the y values by range instead of std that is also possible
        by setting y_by_range==True. The time values only get normalized by setting their starting values to 0. We save the original unnormalized data to 
        self.unnormalized_data. 
        
        
        """
        self.unnormalized_data = self.dataset.copy()
        dataset = self.dataset
        for i,example in enumerate(dataset):
            example[:,0] = example[:,0] - example[0,0] # start light curves at 0
            mean = np.mean(example[:,1])
            if y_by_range:
                range_y = np.max(example[:,1]) - np.min(example[:,1])
                example[:,1] = (example[:,1] - mean) / range_y
                example[:,2] = example[:,2] / y_mean_std[i,1] # normalize errors as well 
            else:
                std = np.std(example[:,1])
                example[:,1] = (example[:,1] - mean) / std
                example[:,2] = example[:,2] / std # normalize errors as well 
        return self

    def reorder(self):
        """
        
        order the time points in the dataset correctly if we need to
        
        """
        for example in self.dataset:
            example[:] = example[example[:,0].argsort()]
        return self

    def set_union_x(self):
        """
        
        calcluates an array of the union of all the time points across the dataset &
        sets it as self.union_x
        
        """
        
        union_x = np.unique(np.hstack([example[:,0] for example in self.dataset]))
        self.union_x = union_x.astype(np.float32)
        print(f'created union_x attribute of length {len(self.union_x)}')
        return self

    def set_target_x(self, num_points=40, forecast=False, forecast_frac=1.2):
        
        """
        sets the target time values we might want to interpolate the light curve to. 
        
        
        parameters:
            num_points     (int)      --> how many points do we want 
            forecast       (boolean)  --> do we want to forecast?
            forecast_frac  (float)    --> what percentage of the light curve relative to its length should we forecast? 
        
        """
        time = self.dataset[:,:,0]
        to_append = time.shape[1] - num_points
        self.target_x = np.zeros_like(time)
        for i,example in enumerate(time):

            max_time = np.max(example)
            if forecast:
                max_time =  forecast_frac * max_time
            target_x = np.arange(0,max_time, max_time/num_points)
            target_x = np.append(target_x, np.zeros((to_append)), axis=0)[:time.shape[1]]
            self.target_x[i] = target_x
        self.target_x = self.target_x.astype(np.float32)
        return self 
    
    def zero_fill(self):
        """
        
        for the sake of formatting, we have to zero fill up to the longest example given
        
        """
        max_len = 0
        for example in self.dataset:
            if len(example) > max_len:
                max_len = len(example)
        for i, example in enumerate(self.dataset):
            length = len(example)
            need_to_append = max_len - length
            example = np.append(example, np.zeros((need_to_append, 3)), axis=0)
            self.dataset[i] = example
        self.dataset = np.array(self.dataset).astype(np.float32)
        print(f'zero fill all the examples up to the length of longest one given, dataset is also now a numpy array w shape: {self.dataset.shape}, instead of a list of numpy arrays')
        return self

    @staticmethod
    def resample_example(example, num_samples=1):
        new_samples = []
        for _ in range(num_samples):
            y = example[:, 1]
            y_err = example[:, 2]
            new_sample = np.array([example[:,0], y + np.random.normal(0,y_err), y_err]).T
            new_samples.append(new_sample)

        return new_samples

    def resample_dataset(self, num_samples=1):
        
        """
        following the resampling method outlined in Naul et al. 2017
        
        that is, sample from the std err, and add it to the flux/mag values
        
        parameters:
            num_samples (int)  --> how many times we resample each light curve from the errors
            
        
        """
        print(f'generating {num_samples} new sample of each example in the dataset & appending them \n old dataset length: {len(self.dataset)}')
        new_samples = []
        for _ in range(num_samples):
            for example in self.dataset:
                y = example[:,1]
                y_err = example[:,2]
                new_sample = np.array([example[:,0], y + np.random.normal(0,y_err), y_err]).T
                new_samples.append(new_sample)
        self.dataset.extend(new_samples)
        print(f'new dataset length: {len(self.dataset)}')
        return self

    def make_masks(self, frac=0.5):
        """
        
        to learn the dataset with hetvae, we take an unsupervised approach whereby the network will reconstruct the time series given only a
        subsampled fraction of the original time points.
        
        parameters:
            frac  (float)  --> fraction of time points in we subsample
        
        """
        
        dataset = self.dataset
        # will depend on dimensions later
        subsampled_mask = np.zeros_like(dataset[:, :, 1])
        recon_mask = np.zeros_like(dataset[:, :, 1])
        for i, example in enumerate(dataset):
            indexes = example[:, 1].nonzero()[0]
            # this should vary to some extent
            length = int(np.round(len(indexes) * frac))
            obs_points = np.sort(np.random.choice(
                indexes, size=length, replace=False))
            subsampled_mask[i, obs_points] = 1
            recon_mask[i] = np.logical_xor(example[:, 1], subsampled_mask[i])
        self.subsampled_mask = subsampled_mask
        self.recon_mask = recon_mask
        print('created subsampled_mask & recon_mask instance attributes')
        return self

    
    def error_to_sample_weight(self):
        
        """
        
        reciprocate the error to weights to be used in the objective function
        
        
        """
        self.dataset[:,:,2] = 1. / self.dataset[:,:,2]
        self.dataset[:,:,2][np.isinf(self.dataset[:,:,2])] = 0.0
        return self