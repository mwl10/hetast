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

warnings.simplefilter('ignore', np.RankWarning) # set warning for polynomial fitting
LCS = utils.get_data('synth_test_data', seed = 0, start_col=0)

### trials is first CL arg, niters per trial is second 

'''
considering:
    batch_size
    frac
    enc_num_heads
    mse_weight
    width
    rec_hidden
    latent_dim
    num_ref_points

'''

def define_model_args(trial):
    
    args = Namespace(
        frac = 0.8,
        data_folder = 'synth_test_data',
        batch_size = 8, #trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
        dropout =0.0,# trial.suggest_float("dropout", 0.0,0.5),
        elbo_weight = 1,#trial.suggest_float("elbo_weight", 0.0, 5.0),
        embed_time = 16,#,trial.suggest_categorical("embed_time", [16,32,64,128]),
        enc_num_heads=trial.suggest_categorical("enc_num_heads", [1,2,4,8,16]),
        width=128,#trial.suggest_categorical("width", [8,16,32,64,128]),
        num_ref_points=16,#trial.suggest_categorical("num_ref_points", [8,16,32,64,128]),
        rec_hidden=8,#trial.suggest_categorical("rec_hidden", [8,16,32,64,128]),
        latent_dim=32,#,trial.suggest_categorical("latent_dim", [8,16,32,64,128]),
        warmup = 4000,#trial.suggest_int('warmup', 3000,5000),
        mse_weight=trial.suggest_float("mse_weight",1,10),
        k_iwae=1,
        kl_annealing=True,
        kl_zero=False, 
        lr=0.001,
        mixing='concat',
        net='hetvae', 
        niters=int(sys.argv[2]), 
        norm=True, 
        save=True, 
        seed=0, 
        std=0.1, 
        device='mps',
        checkpoint = '',
        early_stopping = False,
        patience = False,
        save_at = 10000000, 
        scheduler = False,
        inc_errors = False,
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
#     if args.data_folder == 'synth':
#         data_obj = utils.get_synthetic_data(seed=seed, uniform=True)
#     else:
#         lcs = utils.get_data(seed = seed, folder=args.data_folder, start_col=args.start_col)
#         
        
    train_loader = data_obj["train_loader"]
    test_loader = data_obj["test_loader"]
    val_loader = data_obj["valid_loader"]
    dim = data_obj["input_dim"]
    union_tp = data_obj['union_tp']
    
    if args.checkpoint:
        net, optimizer, args, epoch, loss = utils.load_checkpoint(args.checkpoint, data_obj)
        print(f'loaded checkpoint with loss: {loss}')
    else:
        net = load_network(args, dim, union_tp)
        params = list(net.parameters())
        optimizer = optim.Adam(params, lr=args.lr)
        epoch = 1
        loss = 1000000000
        
    model_size = utils.count_parameters(net) 
    #print('model size {model_size}')
    ############### have patience ##########
    best_loss = loss 
    patience_counter = 0
    ########################################
    for itr in range(epoch, epoch+args.niters):
        ##############
        train_loss = 0
        train_n = 0
        avg_loglik, avg_wloglik, avg_kl, mse, wmse, mae = 0, 0, 0, 0, 0, 0
        ###########set learning rate based on our scheduler ###############
        if args.scheduler == True: 
            args.lr = utils.update_lr(model_size, epoch, args.warmup)
            for g in optimizer.param_groups:
                g['lr'] = args.lr 
        ##################################################################    
        if args.kl_annealing:
            wait_until_kl_inc = 10000
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.999999 ** (itr - wait_until_kl_inc))
        elif args.kl_zero:
            kl_coef = 0
        else:
            kl_coef = 1
        ###################################################################  
        for train_batch in train_loader:
            batch_len = train_batch.shape[0]
            ### errorbar stuff (do on cpu) ##############################
            errorbars = torch.swapaxes(train_batch[:,:,:,2], 2,1)
            weights = errorbars.clone()
            weights[weights!=0] = 1 / weights[weights!=0]
            errorbars[errorbars!=0] = torch.log(errorbars[errorbars!=0])
            logerr = errorbars.to(device)
            weights = weights.to(device)
            ############################################################
            subsampled_mask = utils.make_masks(train_batch, frac=args.frac)
            train_batch = train_batch.to(device)
            subsampled_mask = subsampled_mask.to(device)
            recon_mask = torch.logical_xor(subsampled_mask, train_batch[:,:,:,1])
            context_y = torch.cat((
              train_batch[:,:,:,1] * subsampled_mask, subsampled_mask
            ), 1).transpose(2,1)
            recon_context_y = torch.cat((
                train_batch[:,:,:,1] * recon_mask, recon_mask
            ), 1).transpose(2,1)
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
            ##################################################
            train_loss += loss_info.composite_loss.item() * batch_len
            avg_loglik += loss_info.loglik * batch_len
            avg_wloglik += loss_info.wloglik * batch_len
            avg_kl += loss_info.kl * batch_len
            mse += loss_info.mse * batch_len
            wmse += loss_info.wmse * batch_len
            mae += loss_info.mae * batch_len
            train_n += batch_len
            ##################################################
                
        print(f'{itr},', end='', flush=True)
        if itr % 10 == 0:
            print(
                '\tIter: {}, train loss: {:.4f}, avg nll: {:.4f}, avg wnll: {:.4f}, avg kl: {:.4f}, '
                'mse: {:.6f}, wmse: {:.6f}, mae: {:.6f}'.format(
                    itr,
                    train_loss / train_n,
                    -avg_loglik / train_n,
                    -avg_wloglik / train_n,
                    avg_kl / train_n,
                    mse / train_n,
                    wmse / train_n,
                    mae / train_n))
            
        valid_nll_loss, _ = utils.evaluate_hetvae(
            net,
            dim,
            val_loader, # should be val_loader
            0.5,
            k_iwae=args.k_iwae,
            device=args.device
            )
        ###########################################
        if itr % args.save_at == 0 and args.save:
            #print('saving.................')
            torch.save({
                'args': args,
                'epoch': itr,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / train_n,
            }, 'synth' + '_' + str(experiment_id) + '.h5')
            #print('done')
        ############################################
        if args.early_stopping:
            if best_loss > (train_loss / train_n): 
                patience_counter += 1
            else:
                best_loss = (train_loss / train_n) 
                patience_counter == 0
            
            if patience_counter >= args.patience:
                print(f'training has not improved for {args.patience} epochs')
                torch.save({
                    'args': args,
                    'epoch': itr,
                    'state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss / train_n,
                }, 'synth' + '_' + str(experiment_id) + '.h5')
                break

                
        ###### optuna stuff #######################        
        trial.report(valid_nll_loss, itr)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return valid_nll_loss

def objective(trial):
    args = define_model_args(trial)
    loss = train(trial,args, LCS)
    return loss



def main():
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(sys.argv[1]), timeout=600)
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
        
