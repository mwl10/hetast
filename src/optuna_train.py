import numpy as np
import torch
import torch.optim as optim
import my_utils
import argparse
from argparse import Namespace
from random import SystemRandom
import models
import utils
import optuna
from optuna.trial import TrialState



def main(trial, args):
    experiment_id = int(SystemRandom().random() * 10000000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device(args.device)
    
    if args.data_folder == 'synth': # we can make the synthetic data, or we can pass the data_obj 
        data_obj = my_utils.get_synthetic_data(seed=seed, uniform=True)
    else:
         _, data_obj = my_utils.get_data(seed=seed, folder=args.data_folder, start_col=args.start_col)
        
     
    train_loader = data_obj["train_loader"]
    test_loader = data_obj["test_loader"]
    val_loader = data_obj["valid_loader"]
    dim = data_obj["input_dim"]
    union_tp = data_obj['union_tp']
    
    if args.checkpoint:
        net, optimizer, args, epoch, loss = my_utils.load_checkpoint(args.checkpoint, data_obj)
        print(f'loaded checkpoint with loss: {loss}')
    else:
        net = models.load_network(args, dim, union_tp)
        params = list(net.parameters())
        optimizer = optim.Adam(params, lr=args.lr)
        epoch = 1
        loss = 10000
        
    model_size = utils.count_parameters(net) 
    
    if args.scheduler == True: 
        args.lr = my_utils.update_lr(model_size, epoch, args.warmup)
        for g in optimizer.param_groups:
            g['lr'] = args.lr
            
        
    # stop if loss doesn't decrease for how many epochs? 
    best_loss = loss 
    counter = 0
    
    for itr in range(epoch, epoch+args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_loglik, avg_kl, mse, mae = 0, 0, 0, 0
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
        for train_batch in train_loader:
            batch_len = train_batch.shape[0]
            train_batch = train_batch.to(device)
          
            subsampled_mask = train_batch[:,:,:,-2]
            recon_mask = train_batch[:,:,:,-1]

            context_y = torch.cat((
              train_batch[:,:,:,1] * subsampled_mask, subsampled_mask
            ), 1).transpose(2,1)
        
            recon_context_y = torch.cat((
                train_batch[:,:,:,1] * recon_mask, recon_mask
            ), 1).transpose(2,1)
            
            loss_info = net.compute_unsupervised_loss(
                train_batch[:, 0, :,0],
                context_y,
                train_batch[:, 0, :,0],
                recon_context_y,
                num_samples=args.k_iwae,
                beta=kl_coef,
            )
            optimizer.zero_grad()
            loss_info.composite_loss.backward()
            optimizer.step()
            train_loss += loss_info.composite_loss.item() * batch_len
            avg_loglik += loss_info.loglik * batch_len
            avg_kl += loss_info.kl * batch_len
            mse += loss_info.mse * batch_len
            mae += loss_info.mae * batch_len
            train_n += batch_len
            
        valid_nll_loss = my_utils.evaluate_hetvae(
            net,
            dim,
            val_loader,
            0.5,
            k_iwae=args.k_iwae,
            device=args.device)
        
        if args.scheduler == True: 
            args.lr = my_utils.update_lr(model_size, itr, args.warmup)
            for g in optimizer.param_groups:
                g['lr'] = args.lr
        
        
        trial.report(valid_nll_loss, itr)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return valid_nll_loss 
                

if __name__ == '__main__':

    
    def define_model_args(trial):

        args = Namespace(
            batch_size = trial.suggest_categorical("batch_size", [8,16,32]),
            bound_variance = True,
            const_var = False,
            dropout =0.0,# trial.suggest_float("dropout", 0.0,0.5),
            elbo_weight = 1,#trial.suggest_float("elbo_weight", 0.0, 5.0),
            embed_time = trial.suggest_categorical("embed_time", [16,32,64,128]),
            enc_num_heads=1,#,trial.suggest_categorical("enc_num_heads", [1,2,4,8,16]),
            intensity=True,
            k_iwae=1,
            kl_annealing=True,#trial.suggest_categorical("kl_annealing",False),
            kl_zero=False, 
            latent_dim=32,#,trial.suggest_categorical("latent_dim", [8,16,32,64,128]),
            lr=0.001,
            mixing='concat',#trial.suggest_categorical("mixing", ["concat", "concat_and_mix"]),#"separate", "interp_only", "na"]),
            mse_weight=trial.suggest_float("mse_weight",1,10),
            net='hetvae', 
            niters=10, 
            norm=True, 
            normalize_input='znorm', 
            num_ref_points=16,#trial.suggest_categorical("num_ref_points", [8,16,32,64,128]),
            rec_hidden=8,#trial.suggest_categorical("rec_hidden", [8,16,32,64,128]),
            save=True, 
            seed=0, 
            shuffle=True, 
            std=0.1, 
            var_per_dim=False, 
            width=128,#trial.suggest_categorical("width", [8,16,32,64,128]),
            device='mps',
            data_obj = '',# if we have a dataset to start with
            checkpoint = '',
            early_stopping = False,
            patience = False,
            save_at = 10000000,
            scheduler = True,
            warmup = trial.suggest_int('warmup', 3000,5000),
            data_folder = './ZTF_DR_data/i_band',
            start_col = 1
        )

        return args
    
    def objective(trial):

        args = define_model_args(trial)
        loss = main(trial,args)
        
        return loss

    
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, timeout=600)
    
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
        
    