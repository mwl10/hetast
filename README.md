
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

i.e. calling this script returns something akin to


```bash

Namespace(data_folder='/home2/fggr82/astr/hetast/src/datasets/ZTF_g', start_col=1, checkpoint=None, seed=2, device='cuda', net='HeTVAE', mixing='concat', n_union_tp=3500, embed_time=128, num_heads=8, latent_dim=64, num_ref_points=16, rec_hidden=128, width=512, niters=6000, patience=10000, batch_size=2, k_iwae=1, lr=0.0001, beta1=0.9, beta2=0.999, scheduler=True, warmup=10, factor=0.9, lr_patience=35, threshold=0.01, dropout=0.1, inc_errors=False, frac=0.5, mse_weight=5.0, kl_annealing=True, kl_itrs=6000, n_cycles=32, start=0.0, stop=0.8, ratio=0.5, keep_missing=False, min_length=25, print_at=1, save_at=30, kl_zero=False, const_var=False, var_per_dim=False, num_resamples=0, is_bounded=True) 5796
found 3407 for band='g'
max time:  1687.1367
created union_tp attribute of length 3500
dataset created, lcs.dataset.shape=(2831, 1, 1962, 3)
train size: 2547, valid size: 510, test size: 284
model_size=349378
1,	Iter: 1, train loss: 6.4002, avg nll: 1.4171, avg wnll: 1.4625, avg kl: 0.0448, mse: 0.996535, wmse: 1.430211, mae: 0.788138, val nll: 1.4170, val mse 0.9989, lr 0.000100000
2,	Iter: 2, train loss: 6.3838, avg nll: 1.4155, avg wnll: 1.4587, avg kl: 0.0512, mse: 0.993470, wmse: 1.429660, mae: 0.786717, val nll: 1.4082, val mse 0.9876, lr 0.000100000
3,	Iter: 3, train loss: 6.3208, avg nll: 1.4087, avg wnll: 1.4795, avg kl: 0.0878, mse: 0.981975, wmse: 1.412823, mae: 0.782373, val nll: 1.3970, val mse 0.9624, lr 0.000100000
4,	Iter: 4, train loss: 5.9265, avg nll: 1.3716, avg wnll: 1.4446, avg kl: 0.3094, mse: 0.908870, wmse: 1.287067, mae: 0.743421, val nll: 1.3491, val mse 0.8737, lr 0.000100000
5,	Iter: 5, train loss: 5.5905, avg nll: 1.3335, avg wnll: 1.4169, avg kl: 0.3320, mse: 0.848568, wmse: 1.177278, mae: 0.713572, val nll: 1.3203, val mse 0.8245, lr 0.000100000
6,	Iter: 6, train loss: 5.4810, avg nll: 1.3201, avg wnll: 1.4209, avg kl: 0.2858, mse: 0.829249, wmse: 1.146539, mae: 0.703656, val nll: 1.3132, val mse 0.8097, lr 0.000100000
```

