import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# all operations are in place


class DataSet:
    def add_files(self, files):
        self.files = files
        return self
    
    def files_to_numpy(self, minimum=50, maximum=300):
        dataset = []
        print(len(self.files))
        for i, file in enumerate(self.files):
            with open(file, 'r') as f:
                example = pd.read_csv(file, sep='\t').to_numpy()

            print(f'dims of {file}:\t{example.shape}')
            example = example[example[:,0].argsort()]
            if (len(example) <= minimum): #or  (len(example) >= maximum):
                del self.files[i]
                continue
            dataset.append(example)
        self.dataset = dataset

        return self


    def handle_dups(self):
        for index, example in enumerate(self.dataset):
            unique, i = np.unique(example[:, 0], return_index=True)
            self.dataset[index] = example[i]
        return self


#  **************************************************
#     prune_outliers()
#     **************************************************
#         parameters: std_threshold (default: 3)

#         std_threshold sets how many stds from the mean y values we will remove outliers


    def prune_outliers(self, std_threshold=3):
        for i, example in enumerate(self.dataset):
            y = example[:,1]
            y_std = np.std(y)
            y_mean = np.mean(y)
            outlier_indexes = np.where((y > (y_mean + y_std*std_threshold)) | (y < (y_mean - y_std*std_threshold)))[0]
            print(f'indexes of outliers to be pruned, if any: {outlier_indexes}')
            self.dataset[i] = np.delete(example, outlier_indexes, axis=0)


        return self


# three point median filter

# clipping of all points that deviated significantly from a quintic polynomial fit to the data 

# clipping threshold was initially set to 0.25 mag and then iteratively increased (if necessary) until no more 
# than 10 percent of the points were rejected

    def prune_graham(self, plot=False, index=100, res_std=True, std_threshold=3, mag_threshold=0.25):
        for i, example in enumerate(self.dataset):
            example[:,1] = signal.medfilt(example[:,1], kernel_size=3)
            print(len(example))
            quintic_fit = np.polyfit(example[:,0], example[:,1], deg=5)
            quintic_y = np.array([example[:,0]**5, example[:,0] ** 4, example[:,0] ** 3, example[:,0] ** 2 , example[:,0], np.ones(len(example))])
            quintic_y = np.matmul(quintic_y.T, quintic_fit)
            quintic_y_std = np.std(quintic_y)
            #print(np.std(quintic_y))
            
            dev = np.abs(example[:,1] - quintic_y)
            # residual std error 
            if res_std:
                res_std = np.sqrt(np.mean(dev**2))
                mag_threshold = std_threshold*res_std
                print(mag_threshold)

            #print(np.std(dev))
            
            # increase mag_threshold of outliers if more than 10 percent are removed
            percentage = 1.
            while(True):
                outliers = np.where(dev >= mag_threshold)[0]
                percentage = len(outliers)/ len(example) # none past 10 percent 
                #print(percentage)
                if percentage > .1:
                    #print('need to increase outlier mag threshold')
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


#    **************************************************
#     normalize()
#     **************************************************

#         parameters: normalize_y (default: individual)
#                     normalize_x (default: none)

#         normalize_y sets how we normalize the y values, & can be
#             'all'         to normalize across the dataset 
#             'individual'  to normalize per example

#         normalize_x sets how we normalize x values (time) & can be
#          
#             'all'         to normalize across the dataset
#             'individual'  to normalize per example

#     ->set_union_x() sets instance attribute (union_x) of the dataset object union_x for the network to use in caexampleulating 'intensity' 
#     ->zero_fill(), make_masks(), are formating that the network needs
#     ->error_to_sample_weight() changes the errors column to sample weights which are used in the loss function 
#             i.e. MSE = (y_pred - y)**2 * sample_weights 
    
# '''

    def normalize(self, normalize_y='individual', y_by_range=False): 
        dataset = self.dataset
        
        union_y = np.hstack([example[:,1] for example in dataset])
        range_y = np.max(union_y) - np.min(union_y)
        std_y = np.std(union_y)
        mean_y = np.mean(union_y)
        min_y = np.min(union_y)
        y_mean_std = np.zeros((len(dataset), 2)) # keep this for denormalization purposes in prediction
        x_min= np.zeros((len(dataset), 1)) # keep this for denormalization purposes in prediction

        for i,example in enumerate(dataset):
             
            x_min[i] = example[0,0]
            example[:,0] = example[:,0] - x_min[i] # start light curves at 0

            if normalize_y == 'all':
                if y_by_range:
                    y_mean_std[i] = (min_y, range_y)
                    example[:,1] = (example[:,1] - min_y) /range_y
                    example[:,2] = example[:,2] / range_y # normalize errors as well 
                else:
                    y_mean_std[i] = (min_y, std_y)
                    example[:,1] = (example[:,1] - min_y) / std_y
                    example[:,2] = example[:,2] / std_y # normalize errors as well 

            elif normalize_y == 'individual':
                if y_by_range:
                    range_y = np.max(example[:,1]) - np.min(example[:,1])
                    y_mean_std[i] = (np.min(example[:,1]), range_y)
                    example[:,1] = (example[:,1] - y_mean_std[i,0]) / y_mean_std[i,1]
                    example[:,2] = example[:,2] / y_mean_std[i,1] # normalize errors as well 

                else:
                    y_mean_std[i] = (np.min(example[:,1]), np.std(example[:,1]))
                    example[:,1] = (example[:,1] - y_mean_std[i,0]) / y_mean_std[i,1]
                    example[:,2] = example[:,2] / y_mean_std[i,1] # normalize errors as well 

        self.y_mean_std = y_mean_std
        self.x_min = x_min
            

        return self


    
    def denormalize(self):
        denormalized = self.dataset.copy()

        for i, example in enumerate(denormalized):
            y_mean, y_std = self.y_mean_std[i]
            x_mean, x_std = self.x_mean_std[i]

            example[:,0] = (example[:,0] * x_std) + x_mean
            example[:,1] = (example[:,1] * y_std) + y_mean

            example[:,2] = example[:,2] * y_std
        return denormalized


    def reorder(self):
        for example in self.dataset:
            example[:] = example[example[:,0].argsort()]
        return self

    def set_union_x(self):
        union_x = np.unique(np.hstack([example[:,0] for example in self.dataset]))

        self.union_x = union_x.astype(np.float32)
        print(f'created union_x attribute of length {len(self.union_x)}')
        
        return self


    def set_target_x(self, num_points=40, forecast=False, forecast_frac=1.2):
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




#    **************************************************
#     resample_dataset()
#     **************************************************

#         desc: sampling from the errors and adding to the flux value to create more data variants
        
#         parameters: num_samples (default: 1)

#         num_samples sets how many sets of new samples we will make from the dataset

    def resample_dataset(self, num_samples=1):
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


    # make the masks beforehand...
    def make_masks(self, frac=0.7):
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

        self.dataset[:,:,2] = 1. / self.dataset[:,:,2]
        self.dataset[:,:,2][np.isinf(self.dataset[:,:,2])] = 0.0


        return self

    