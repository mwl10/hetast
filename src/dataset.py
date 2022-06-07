import pandas as pd
import numpy as np
from regex import D



class DataSet:
    def add_files(self, files):
        self.files = files
        return self
    
    def files_to_numpy(self):
        dataset = []
        for file in self.files:
            with open(file, 'r') as f:
                example = pd.read_csv(file, sep='\t').to_numpy()
            print(f'dims of {file}:\t{example.shape}')
            example = example[example[:,0].argsort()]

            dataset.append(example)
        self.dataset = dataset

        return self

    def handle_dups(self):
        for index, example in enumerate(self.dataset):
            unique, i = np.unique(example[:, 0], return_index=True)
            self.dataset[index] = example[i]
        return self


    # function to purge low variability light curves from the dataset? 


    def prune_outliers(self, std_threshold=3):
        for i, example in enumerate(self.dataset):
            y = example[:,1]
            y_std = np.std(y)
            y_mean = np.mean(y)
            outlier_indexes = np.where((y > (y_mean + y_std*std_threshold)) | (y < (y_mean - y_std*std_threshold)))[0]
            print(f'indexes of outliers to be pruned, if any: {outlier_indexes}')
            self.dataset[i] = np.delete(example, outlier_indexes, axis=0)


        return self




    # this feels ugly
    def normalize(self, normalize_time=False): 
        dataset = self.dataset
        starts = np.zeros(len(dataset))

        # # time to lag
        # for i,example in enumerate(dataset):
        #     time = example[:,0]
        #     starts[i] = time[0]
        #     dt = np.insert(time[1:] - time[:-1], 0, 0)
        #     if normalize_time:
        #         dt = dt / np.std(dt)
        #     example[:,0] = dt

        # normalizing y
        union_y = np.hstack([example[:,1] for example in dataset])
        std_y = np.std(union_y)
        mean_y = np.mean(union_y)
        for example in dataset:
            example[:,1] = (example[:,1] - mean_y) / std_y

        return self

        # normalize ys across the dataset
        
        
    def denormalize(self):
        # for i, d in enumerate(dt):
        # dt[i] = d+start+dt[:i].sum()
        return self

    def reorder(self):
        for example in self.dataset:
            example[:] = example[example[:,0].argsort()]
        return self


    def set_union_x(self):

        example_lengths = np.array([len(example) for example in self.dataset])
        union_x = np.zeros((sum(example_lengths)))
        acc = 0
        for i, example in enumerate(self.dataset):

            union_x[acc:acc + len(example)] = example[:, 0]
            acc += len(example)
        union_x = np.unique(union_x)
        union_x.sort()

        self.union_x = union_x.astype(np.float32)
        print(f'created union_x attribute of length {len(self.union_x)}')
        
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

    