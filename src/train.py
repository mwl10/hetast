import numpy as np
import torch
import torch.optim as optim
import argparse
from random import SystemRandom
from model import load_network
import utils
import warnings


def train(args):
    experiment_id = int(SystemRandom().random() * 10000000)
    print(args, experiment_id)
    ##################################
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    ##################################
    device = torch.device(args.device)
    lcs = utils.get_data(folder=args.data_folder, start_col=args.start_col, n_union_tp=args.n_union_tp, num_resamples=args.num_resamples, batch_size=args.batch_size)
    data_obj = lcs.data_obj
    train_loader = data_obj["train_loader"]
    test_loader = data_obj["test_loader"]
    val_loader = data_obj["valid_loader"]
    dim = data_obj["input_dim"]
    # can just make it instead here! 
    union_tp = data_obj['union_tp']
    # what needs to be the same in args for continued run
    if args.checkpoint:
        net, optimizer, _, epoch, loss, train_losses, test_losses = utils.load_checkpoint(args.checkpoint, data_obj, device=args.device)
        epoch+=1
        for g in optimizer.param_groups:
                ## update learning rate for checkpoint 
                g['lr'] = args.lr    

        print(f'loaded checkpoint w/ {loss=}')
    else:
        net = load_network(args, dim, union_tp)
        params = list(net.parameters())
        optimizer = optim.Adam(params, lr=args.lr)
        epoch = 1
        loss = 1000000
        train_losses = []
        test_losses = []
        
    model_size = utils.count_parameters(net) 
    print(f'{model_size=}')
    ############### have patience ##########
    best_loss = loss
    patience_counter = 0
    ######################################## 
    if args.kl_annealing:
        kl_coefs = utils.frange_cycle_linear(args.niters)
    ##################

    for itr in range(epoch, epoch+args.niters):
        train_loss = 0
        train_n = 0
        avg_loglik, avg_wloglik, avg_kl, mse, wmse, mae = 0, 0, 0, 0, 0, 0
        ###########set learning rate based on our scheduler###############
        if args.scheduler == True: 
            args.lr = utils.update_lr(model_size, itr, args.warmup)
            for g in optimizer.param_groups:
                g['lr'] = args.lr 
        ##################################################################    
        if args.kl_annealing:
            kl_coef = kl_coefs[itr-epoch]
        elif args.kl_zero:
            kl_coef = 0
        else:
            kl_coef = 1
        ###################################################################  
        for train_batch in train_loader:
            batch_len = train_batch.shape[0]
            ### errorbar stuff, to do on cpu ##############################
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
            #########################################################
            train_loss += loss_info.composite_loss.item() * batch_len
            avg_loglik += loss_info.loglik * batch_len
            avg_wloglik += loss_info.wloglik * batch_len
            avg_kl += loss_info.kl * batch_len
            mse += loss_info.mse * batch_len
            wmse += loss_info.wmse * batch_len
            mae += loss_info.mae * batch_len
            train_n += batch_len
            #########################################################
            
        print(f'{itr},', end='', flush=True)
        
        ####### nnnnannnnnn #############
        if np.isnan(train_loss / train_n):
            print('nan in loss,,,,,,,,, stopping')
            break
        
        if itr % args.print_at == 0:
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
            
        if itr % 10 == 0:
            test_nll, test_mse, indiv_nlls = utils.evaluate_hetvae(
                net,
                dim,
                test_loader, 
                0.5,
                device=args.device
                )
            
            train_losses.append([(-avg_loglik / train_n).item(),(mse / train_n).item()])
            test_losses.append([test_nll.item(),test_mse.item()])
        ###########################################
        if itr % args.save_at == 0 and args.save:
            print('saving.................')
            torch.save({
                'args': args,
                'epoch': itr,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss' : train_loss / train_n,
                'train_losses': train_losses,
                'test_losses':test_losses,
            }, lcs.name + str((-avg_loglik / train_n).item()) + '.h5')
            print('done')
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
                    'loss' : train_loss / train_n,
                    'train_losses': train_losses,
                    'test_losses':test_losses,
                }, lcs.name + str((-avg_loglik / train_n).item()) + '.h5')
                break

    
def main():
    
    warnings.simplefilter('ignore', np.RankWarning) # set warning for polynomial fitting
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n-union-tp', type=int, default='3500')
    
    ## dataset stuff
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--start-col', type=int, default='0')
    parser.add_argument('--inc-errors', action='store_true')
    parser.add_argument('--print-at', type=int, default='100')
    
    
    ##### model architecture hypers 
    parser.add_argument('--embed-time', type=int, default=128)  
    parser.add_argument('--enc-num-heads', type=int, default=4) 
    parser.add_argument('--latent-dim', type=int, default=128)  
    parser.add_argument('--mixing', type=str, default='concat') 
    parser.add_argument('--num-ref-points', type=int, default=16) 
    parser.add_argument('--rec-hidden', type=int, default=128) 
    parser.add_argument('--width', type=int, default=512)
    
    ##### training hypers
    parser.add_argument('--save-at', type=int, default='1000000')
    parser.add_argument('--patience', type=int, default='100')
    parser.add_argument('--early-stopping', action='store_true')
    parser.add_argument('--niters', type=int, default=10)
    parser.add_argument('--frac', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=128) 
    parser.add_argument('--mse-weight', type=float, default=5.0)  
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num-resamples', type=int, default=0)
    ## learning rate
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--warmup', type=int, default='4000')
    
    ## no need to mess with 
    parser.add_argument('--kl-zero', action='store_true')
    parser.add_argument('--kl-annealing', action='store_true')
    parser.add_argument('--net', type=str, default='hetvae')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--const-var', action='store_true') 
    parser.add_argument('--var-per-dim', action='store_true')
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=3) 
    parser.add_argument('--save', action='store_false')
    parser.add_argument('--k-iwae', type=int, default=1) 
    

    args = parser.parse_args()
    train(args)
    return args

if __name__ == '__main__':
    main()
    
    