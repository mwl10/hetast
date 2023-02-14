import utils


for i, obj_folder in enumerate(glob.glob('./datasets/obj_dirs/*')):
    obj_name = obj_folder.split('/')[-1].split('_')[1]
    try:
        lcs = utils.get_data(f'../datasets/obj_dirs/ZTFgr_{obj_name}', sep=',', start_col=1, batch_size=1, min_length=40, n_union_tp=3500, num_resamples=111)
    except Exception:
        continue
        
    net, optimizer, args, epoch, loss = utils.load_checkpoint('./ZTFgr_small0.7880523800849915.h5', lcs.data_obj)

    train = lcs.data_obj['train_loader']

    train_nll, mse, individual_nlls = utils.evaluate_hetvae(
                    net,
                    2,
                    train,
                    0.5,
                    k_iwae=1,
                    device='mps'
                    )
    
    with open('LOSSSES.txt', 'a') as f:
        f.write(f'{obj_name},{train_nll},{mse}\n')