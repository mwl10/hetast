import numpy as np
import torch
import torch.optim as optim
import argparse
from random import SystemRandom
from argparse import Namespace
from model import load_network
import utils
import optuna
from optuna.trial import TrialState
import sys
import logging
import warnings
from torch.optim import lr_scheduler

warnings.simplefilter('ignore', np.RankWarning) # set warning for polynomial fitting
LCS = utils.get_data('../../datasets/ZTF_g_test', start_col=1)

### trials is first command line arg, niters per trial is second  ####

def define_model_args(trial):
    args = Namespace(
        niters=int(sys.argv[2]),  ### num iters per trial 
        enc_num_heads=trial.suggest_categorical("enc_num_heads",[1,2,4,8,16,32]),
        embed_time = trial.suggest_categorical("embed_time",[32,64,128,256]),
        width=trial.suggest_categorical('width', [128,256,512,1024]),
        num_ref_points=trial.suggest_categorical('num_ref_points', [8,16,24,32,48]),
        rec_hidden=trial.suggest_categorical("rec_hidden",[32,64,128,256]),
        latent_dim= trial.suggest_categorical("latent_dim",[32,64,128,256]),
        dropout =0.0,#trial.suggest_float("dropout", 0.0,0.5,step=0.1),
        lr=0.0003,
        frac = 0.5,
        mixing='concat',
        mse_weight=5,#trial.suggest_int("mse_weight",1,20),
        data_folder = 'ZTF_gband_test',
        batch_size = 2,
        patience = 10000,
        scheduler = False,
        warmup = 4000,#trial.suggest_int('warmup', 3000,5000),
        k_iwae=1,
        kl_annealing=True,
        kl_zero=False, 
        net='hetvae', 
        norm=True, 
        save=False, 
        seed=0, 
        std=0.1, 
        device='mps',
        checkpoint = '',
        save_at = 10000000, 
        inc_errors = False,
        nodet=False,
        print_at=1
    ) 
    return args


def train(trial, args, lcs):
    experiment_id = int(SystemRandom().random() * 10000000)
    #print(args, experiment_id)
    ##################################
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    ##################################
    device = torch.device(args.device)
    
    data_obj = lcs.data_obj
    N_union_tp = trial.suggest_categorical('N_union_tp', [500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
    lcs.set_union_tp(uniform=True, n=N_union_tp)
    train_loader = data_obj["train_loader"]
    test_loader = data_obj["test_loader"]
    val_loader = data_obj["valid_loader"]
    dim = data_obj["input_dim"]
    union_tp = torch.tensor(lcs.union_tp)
    
    net = load_network(args, dim, union_tp)
    params = list(net.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    epoch = 1
    loss = 1000000
    train_losses = []
    test_losses = []
    val_losses = []
    lrs = []
    
    model_size = utils.count_parameters(net) 
    print(f'{model_size=}')
    ############### have patience ##########
    best_loss = loss
    patience_counter = 0
    ######################################## 
    if args.kl_annealing:
        kl_coefs = utils.frange_cycle_linear(6000, n_cycle=16)
    ##################
    for itr in range(epoch, epoch+args.niters):
        print(f'{itr},', end='', flush=True)
    
        train_loss = 0
        train_n = 0
        avg_loglik, avg_wloglik, avg_kl, mse, wmse, mae = 0, 0, 0, 0, 0, 0 
        if args.kl_annealing:
            kl_coef = kl_coefs[itr] # need global number of epochs to continue on based on schedule
        elif args.kl_zero:
            kl_coef = 0
        else:
            kl_coef = 1
        ###################################################################  
        for train_batch in train_loader:
            batch_len = train_batch.shape[0]
            
            ### including obs error ##############################
            errorbars = torch.swapaxes(train_batch[:,:,:,2], 2,1)
            weights = errorbars.clone()
            weights[weights!=0] = 1 / weights[weights!=0]
            errorbars[errorbars!=0] = torch.log(errorbars[errorbars!=0]**2)
            logerr = errorbars.to(device) # log variance on observations 
            weights = weights.to(device)
            
            ############################################################
            subsampled_mask = utils.make_masks(train_batch, frac=args.frac)
            train_batch = train_batch.to(device)
            subsampled_mask = subsampled_mask.to(device)
            recon_mask = torch.logical_xor(subsampled_mask, train_batch[:,:,:,1])
            context_y = torch.cat((
              train_batch[:,:,:,1] * subsampled_mask, subsampled_mask), 1).transpose(2,1)
            recon_context_y = torch.cat((
                train_batch[:,:,:,1] * recon_mask, recon_mask), 1).transpose(2,1)
            #############################################################
            loss_info = net.compute_unsupervised_loss(
                train_batch[:, 0, :,0],
                context_y,
                train_batch[:, 0, :,0],
                recon_context_y,
                logerr,
                weights,
                num_samples=args.k_iwae,
                beta=kl_coef,
            )
            optimizer.zero_grad()
            
            if args.inc_errors:
                loss_info.weighted_comp_loss.backward()
            else:
                loss_info.composite_loss.backward()
                
            optimizer.step()
            
            ########### train loss ##################################
            train_loss += loss_info.composite_loss.item() * batch_len
            avg_loglik += loss_info.loglik * batch_len
            avg_wloglik += loss_info.wloglik * batch_len
            avg_kl += loss_info.kl * batch_len
            mse += loss_info.mse * batch_len
            wmse += loss_info.wmse * batch_len
            mae += loss_info.mae * batch_len
            train_n += batch_len
            #########################################################
        
    
        ####### nan, stop training #############
        if np.isnan(train_loss / train_n):
            print('nan in loss,,,,,,,,, stopping')
            break
        
        #### validation loss
        val_nll, val_mse, val_indiv_nlls = utils.evaluate_hetvae(
            net,
            dim,
            val_loader, 
            0.5,
            device=args.device
        )
        
        lrs.append(optimizer.param_groups[0]['lr'])
        if scheduler:
            scheduler.step(val_nll) 
        
        ######## print train / valid losses ########
        if itr % args.print_at == 0:
            print(
                '\tIter: {}, train loss: {:.4f}, avg nll: {:.4f}, avg wnll: {:.4f}, avg kl: {:.4f}, '
                'mse: {:.6f}, wmse: {:.6f}, mae: {:.6f}, val nll: {:.4f}, val mse {:.4f}'.format(
                    itr,
                    train_loss / train_n,
                    -avg_loglik / train_n,
                    -avg_wloglik / train_n,
                    avg_kl / train_n,
                    mse / train_n,
                    wmse / train_n,
                    mae / train_n,
                    val_nll,
                    val_mse))
                
        ####### test loss every 10 itrs, save losses too #############
        if itr % 10 == 0:
#             test_nll, test_mse, indiv_nlls = utils.evaluate_hetvae(
#                 net,
#                 dim,
#                 test_loader, 
#                 0.5,
#                 device=args.device
#                 )
            
            train_losses.append([(-avg_loglik / train_n).item(),(mse / train_n).item(),(avg_kl / train_n).item()])
#             test_losses.append([test_nll.item(),test_mse.item()])
            val_losses.append([val_nll.item(),val_mse.item()])
#             print('test nll: {:.4f}, test mse: {:.4f}'.format(test_nll,test_mse))
            

        ###### optuna stuff #######################        
        trial.report(val_nll, itr)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_nll



def objective(trial):
    args = define_model_args(trial)
    loss = train(trial,args, LCS)
    return loss


def main():
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(sys.argv[1]), timeout=10000)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    

if __name__ == '__main__':
    main()
        
