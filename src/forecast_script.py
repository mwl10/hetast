import glob
import utils
import sys

#### plotting forecasts, and getting forecast losses

for i, obj_folder in enumerate(glob.glob('./datasets/obj_dirs/*')):
    obj_name = obj_folder.split('/')[-1].split('_')[1]
    print(f'{i},', end='', flush=True)
    print(obj_folder)
    
    try:
        lcs = utils.get_data(obj_folder, sep=',', start_col=1, batch_size=1, min_length=40, \
                             n_union_tp=3500, num_resamples=111, shuffle=False, chop==False)
        train = lcs.data_obj['train_loader']

    except Exception as e:
        print(e)
        continue
        
    net, optimizer, args, epoch, loss = utils.load_checkpoint('./ZTFgr_small0.7880523800849915.h5', lcs.data_obj, device='cuda')
    examples, z, recons = utils.predict(train, net, device='cuda', subsample=False, target_x=None, forecast=True)
    utils.save_recon(examples, recons,z,obj_name, bands=lcs.bands, save_folder='forecast_plots')

    train_nll, mse, individual_nlls = utils.evaluate_hetvae(
                net,
                2,
                train,
                0.5,
                k_iwae=2,
                device='cuda',
                forecast=True
                )

    with open('forecast_losses.txt', 'a') as f:
        f.write(f'{obj_name},{train_nll},{mse}\n')
        
        
    
    
    