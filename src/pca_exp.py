
import utils
import sys
import os

import sys
import os
import torch
import numpy as np
import utils
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle 
%load_ext autoreload
np.random.seed(2) 
torch.manual_seed(2)

# pca is just a text file bin gets a folder w/ 3 light curves, doing it w/ gri 

# '../checkpoints/gr/ZTF_gr0.9532953500747681.h5
# lcs = utils.get_data('datasets/ZTF_gr', test_split=0.0,keep_missing=False, min_length=25)

lcs = utils.get_data('datasets/ZTF_gri', test_split=0.0,keep_missing=False, min_length=25)

net,optimizer,scheduler,lrs,args,epoch,losses = utils.load_checkpoint('datasets/ZTF_gri0.8933992385864258.h5', lcs.data_obj,device='cuda')

qzs,disc_path = utils.encode(lcs.data_obj['train_loader'], net, device='cuda')

num_ref_points = 16 #args.num_ref_points
latent_dim = 64     #args.latent_dim
l = len(lcs.dataset)
n_samples = 10
zs = (np.random.randn(n_samples, qzs.shape[0], qzs.shape[2],qzs.shape[3]) * qzs[:,1,:,:] + qzs[:,0,:,:]).mean(0)
np.savetxt('zs_gri.dat',zs.reshape(zs.shape[0],-1))


RS = 20150101
pca = PCA(random_state=RS)
pca3d = pca.fit_transform(zs.reshape(zs.shape[0],-1)) 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pca3d[:,0],pca3d[:,1],pca3d[:,2])


# bins for different pcs for 3 dims 
bins = 8
interpols = []
pca_comps = 10
for i in range(pca_comps): # first 4 pcs
    r1,r2 = np.min(pca3d[:,i]), np.max(pca3d[:,i])
    
    bins1 = np.arange(r1,r2, step=(r2-r1)/bins)
    bin_i = np.digitize(pca3d[:,i],bins=bins1) # which light curve belongs to which bin? 
    
    # average light curves across bins 
    avgs = [np.concatenate((zs[bin_i==i].mean(0)[np.newaxis],disc_path[bin_i==i].mean(0)[np.newaxis]), axis=0) \
     for i in range(1,bins+1)]
    
    avgs = np.array(avgs,dtype=np.float32)
    #print(avgs.shape)
    target_tp = np.arange(0,1500,step=2.5, dtype=np.float32)
    target_tp = target_tp[np.newaxis].repeat(len(lcs.bands),axis=0)[np.newaxis].repeat(len(bins1),axis=0)
    interps = utils.decode(net,zs=avgs[:,0],disc_path=avgs[:,1],target_x=target_tp,device='cuda', batch_size=2)
    interpols.append(interps)

    
with open('interpols_gri.pkl', 'wb') as f:  # Overwrites any existing file.
    pickle.dump(interpols, f, pickle.HIGHEST_PROTOCOL)
    
    
#### blending  
bins = 7
interpols = []

r1_1,r2_1 = np.min(pca3d[:,0]), np.max(pca3d[:,0])
r1_2,r2_2 = np.min(pca3d[:,1]), np.max(pca3d[:,1])

bins1 = np.arange(r1_1,r2_1, step=(r2_1-r1_1)/bins)
bins2 = np.arange(r1_2,r2_2, step=(r2_2-r1_2)/bins)

in_which_1 = np.digitize(pca3d[:,0],bins=bins1) - 1
in_which_2 = np.digitize(pca3d[:,1],bins=bins2) - 1 
avgs= []
for i in range(bins):
    for j in range(bins):
        #print(np.logical_and(in_which_1==i,in_which_2==j).sum())
        avg = np.concatenate((zs[np.logical_and(in_which_1==i,in_which_2==j)].mean(0)[np.newaxis], \
                                disc_path[np.logical_and(in_which_1==i,in_which_2==j)].mean(0)[np.newaxis]),\
                               axis=0)
        avgs.append(avg)
        
avgs = np.array(avgs,dtype=np.float32)
target_tp = np.arange(0,1500,step=2.5, dtype=np.float32)
target_tp = target_tp[np.newaxis].repeat(len(lcs.bands),axis=0)[np.newaxis].repeat(bins**2,axis=0)
interps = utils.decode(net,zs=avgs[:,0],disc_path=avgs[:,1],target_x=target_tp,device='cuda', batch_size=2)


with open('interpols_gri_blend.pkl', 'wb') as f:  # Overwrites any existing file.
    pickle.dump(interps, f, pickle.HIGHEST_PROTOCOL)