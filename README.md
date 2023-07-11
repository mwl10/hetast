
This repository inherits code from https://github.com/reml-lab/hetvae, the official implementation of Heteroscedastic Temporal Variational Autoencoder for Irregularly Sampled Time Series (https://arxiv.org/pdf/2107.11350.pdf). 

It's been modified to use for Time Domain Astronomy, more speficially for modeling AGN light curves in my MRes work at Durham University. 

## Requirements

    numpy, pytorch, matplotlib, pandas, eztao, scipy, optuna, sklearn
 

## Usage

Data folders must follow the following directory scheme

```bash
ZTF_DR_data/

    ├── g 
         
         └── 
              000018.77+191232.9_DR_gband.csv
              000111.81-092508.2_DR_gband.csv
              
    ├── i
    
         └──  
              000018.77+191232.9_DR_iband.csv
              000111.81-092508.2_DR_iband.csv
```
              
Where the object name must be separated by an underscore from the rest of the filename so the objects can be matched properly to handle the multivariate case and the lightcurves must be in subdirectories labeled by the name of the band, i.e. g, i, b, r. In the univariate case, you still must place your light curves in a band folder, but the naming convention doens't matter. 

Additionally, the light curve files should be CSV with time, mag/flux, magerr/fluxerr in the first 3 columns respectively. If your csvs don't have the mag/flux starting in the first column you can provide a --start-col (the default is 0).

There is an extensive list of hyperparameters for this nextwork which can be found in conf/config.yaml, all of which can be altered in the training from their defaults like:

```bash
python3 train.py data_folder=datasets/ZTF_g \
                         dataset.start_col=1 \
                         dataset.shuffle=True \
                         device=mps \
                         print_at=1 \
                         training.optimizer.lr=0.0001\
                         save_at=100 \
                         filter.keep_missing=true\
                         filter.min_length=1 \
                         dataset.sep=comma
```







