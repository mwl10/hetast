# pylint: disable=E1101
from vae_model import HeTVAE, HeTVAE_nodet

def load_network(args, dim, union_tp=None):
    if args.net == 'HeTVAE_nodet':
        net = HeTVAE_nodet(
            input_dim=dim,
            nhidden=args.rec_hidden,
            latent_dim=args.latent_dim,
            embed_time=args.embed_time,
            num_heads=args.num_heads,
            intensity=True,
            union_tp=union_tp,
            width=args.width,
            num_ref_points=args.num_ref_points,
            std=args.std,
            is_constant=False, #args.constant_var
            is_bounded=True, 
            is_constant_per_dim=False,#args.var_per_dim, 
            elbo_weight=1.0,
            mse_weight=args.mse_weight,
            norm=True,
            mixing=args.mixing,
            device=args.device,
            dropout=args.dropout,
        ).to(args.device)
        
    elif args.net == 'HeTVAE': 
        net = HeTVAE(
            input_dim=dim,
            nhidden=args.rec_hidden,
            latent_dim=args.latent_dim,
            embed_time=args.embed_time,
            num_heads=args.num_heads,
            intensity=True,
            union_tp=union_tp,
            width=args.width,
            num_ref_points=args.num_ref_points,
            std=0.1,
            is_constant=False, #args.constant_var
            is_bounded=args.is_bounded, 
            is_constant_per_dim=False,#args.var_per_dim, 
            elbo_weight=1.0,
            mse_weight=args.mse_weight,
            norm=True,
            mixing=args.mixing,
            device=args.device,
            dropout=args.dropout,
        ).to(args.device)
        
    return net

