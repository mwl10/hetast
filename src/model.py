# pylint: disable=E1101
from vae_model import HeTVAE

def load_network(args, dim, union_tp=None):
    net = HeTVAE(
        input_dim=dim,
        nhidden=args.rec_hidden,
        latent_dim=args.latent_dim,
        embed_time=args.embed_time,
        num_heads=args.enc_num_heads,
        intensity=True,
        union_tp=union_tp,
        width=args.width,
        num_ref_points=args.num_ref_points,
        std=args.std,
        is_constant=False, #args.constant_var
        is_bounded=False, 
        is_constant_per_dim=False,#args.var_per_dim, 
        elbo_weight=1,
        mse_weight=args.mse_weight,
        norm=True,
        mixing=args.mixing,
        device=args.device,
    ).to(args.device)
    return net

