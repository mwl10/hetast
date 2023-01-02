
This repository inherits code from https://github.com/reml-lab/hetvae, the official implementation of Heteroscedastic Temporal Variational Autoencoder for Irregularly Sampled Time Series (https://arxiv.org/pdf/2107.11350.pdf). 

It's been modified to use for Time Domain Astronomy, more speficially for modeling AGN light curves in my MRes work at Durham University. 

Clearly a work in progress


## Requirements

    numpy, pytorch, matplotlib, pandas, eztao, scipy, optuna, sklearn
 

## Training and Evaluation


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

There is an extensive list of hyperparameters for this nextwork which can be found in main() of train.py, all of which can be altered in the training from their defaults like:

```bash
python3 train.py --folder './ZTF_DR_data' --niters 100 --device cuda --checkpoint './'
```

some that might be of particular convienience are

```bash
--checkpoint
--niters
--early-stopping
--patience
--device
--batch-size
--dropout
--niters
--dropout
--latent-dim
--embed-time
--enc-num-heads
--lr
--num-ref-point
--patience
--save-at

```

the default to save the network happens either when it stops improving for a number of epochs, which is set by the --patience argument and defaults to 50, or we can just saveat a given epoch by setting --save-at 50

<!-- 
if you've trained the network and want to glance at some of the results, take a peak at

[science.py](./src/science.ipynb) -->

<!-- where you can make predictions on the network, visualize the latent space, the attention, etc. -->







