# on hpc

import glob
import utils


for i, obj_folder in enumerate(glob.glob('./datasets/obj_dirs/*')[2:]):
    obj_name = obj_folder.split('/')[-1].split('_')[1]
    print(f'{i},', end='', flush=True)
    try:
        lcs = utils.get_data(obj_folder, sep=',', start_col=1, batch_size=1, min_length=40, n_union_tp=3500, num_resamples=5,shuffle=True, extend=5000)
        
    except Exception:
        print('faillllll')
        continue
    lcs.set_target_x(5000, forecast=True, forecast_frac=1.25)
    net, optimizer, args, epoch, loss = utils.load_checkpoint('./ZTFgr_small0.7880523800849915.h5', lcs.data_obj)
    examples, z, recons = utils.predict(lcs.data_obj['train_loader'], net, device='mps', subsample=False, target_x=lcs.target_x)
    utils.save_recon(examples, recons,z,obj_name, bands=lcs.bands, save_folder='test')
    
    break