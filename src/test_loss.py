

import utils
import sys

folder = sys.argv[1]
checkpoint = sys.argv[2]
num_samples = sys.argv[3]

lcs = utils.get_data(folder, sep=',', start_col=1, batch_size=128, min_length=40, n_union_tp=3500, num_resamples=0)
net, optimizer, args, epoch, loss = utils.load_checkpoint(checkpoint, lcs.data_obj,device='cuda')
test = lcs.data_obj['test_loader']

test_nll, mse = utils.evaluate_hetvae(
                net,
                2,
                test,
                0.5,
                k_iwae=_k_iwae,
                device='cuda'
                )

print(test_nll,mse)

# with open('test_loss.txt', 'a') as f:
#     f.write(f'{test_nll},{mse}\n")