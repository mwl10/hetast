# pylint: disable=E1101
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from layers import UnTAN

class Gaussian:
    mean = None
    logvar = None

class LossInfo:
    px = None
    loglik = None
    elbo = None
    kl = None
    mse = None
    mae = None
    mean_mse = None
    mean_mae = None
    mogloglik = None
    composite_loss = None

class HeTVAE(nn.Module):
    '''Heteroscedastic Temporal Variational Autoencoder'''
    def __init__(
        self,
        input_dim,
        nhidden,
        latent_dim,
        num_heads,
        width,
        num_ref_points,
        embed_time, 
        intensity, 
        union_tp, 
        std=0.1,
        is_constant=False,
        is_bounded=True,
        is_constant_per_dim=False,
        elbo_weight=1.0,
        mse_weight=5.0,
        norm=True,
        mixing='concat',
        device='cuda',
        dropout=0.0,
    ):
        super().__init__()
        self.dim = input_dim
        self.nhidden = nhidden
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.width = width
        self.num_ref_points = num_ref_points
        self.query = torch.linspace(0, 1, num_ref_points)
        self.std = std
        self.is_constant = is_constant
        self.is_bounded = is_bounded
        self.is_constant_per_dim = is_constant_per_dim
        self.elbo_weight = elbo_weight
        self.mse_weight = mse_weight
        self.norm = norm
        self.mixing = mixing
        self.device = device
        self.embed_time = embed_time
        self.intensity = intensity
        self.union_tp = union_tp
        self.dropout = dropout
        k = 2 if self.mixing == 'concat' else 1
        if self.mixing == 'concat_and_mix':
            self.h2z = nn.Sequential(
                nn.Linear(2 * self.num_heads * self.dim, self.width),
                nn.ReLU(),
                nn.Linear(self.width, 2 * self.latent_dim))
        else:
            self.h2z_mean = nn.Sequential( 
                nn.Linear(k * self.num_heads * self.dim, self.width),
                nn.ReLU(),
                nn.Linear(self.width, self.latent_dim))
            self.h2z_var = nn.Sequential(
                nn.Linear(k * self.num_heads * self.dim, self.width),
                nn.ReLU(),
                nn.Linear(self.width, self.latent_dim))
        self.z2o = nn.Sequential(
            nn.Linear(self.nhidden, self.width),
            nn.ReLU(),
            nn.Linear(self.width, 2 * self.dim),
        )
        
        if self.mixing == 'interp_only':
            self.proj = nn.Linear(self.num_heads, self.latent_dim)
        else:
            self.proj = nn.Linear(2 * self.dim * self.num_heads, self.latent_dim)

        self.encoder = UnTAN(
            input_dim=self.dim,
            nhidden=self.nhidden,
            embed_time=self.embed_time,
            num_heads=self.num_heads,
            intensity=self.intensity,
            union_tp=self.union_tp,
            no_mix=True,
            device=self.device,
            dropout=self.dropout,
        )
        self.decoder = UnTAN(
            input_dim= 2 * self.latent_dim, # self.latent_dim
            nhidden=self.nhidden,
            embed_time=self.embed_time,
            num_heads=self.num_heads,
            device=self.device,
            dropout=self.dropout,
        )
        
        

    def h2z_mixing(self, hidden):
        qz = Gaussian()
        if self.mixing == 'concat_and_mix':
            hidden = torch.cat((hidden[:, :, :, 0], hidden[:, :, :, 1]), -1)
            q = self.h2z(hidden)
            qz.mean, qz.logvar = q[:, :,:self.latent_dim], q[:, :, self.latent_dim:]
        elif self.mixing == 'concat':
            hidden = torch.cat((hidden[:, :, :, 0], hidden[:, :, :, 1]), -1)
            qz.mean = self.h2z_mean(hidden)
            qz.logvar = self.h2z_var(hidden)
        elif self.mixing == 'separate':
            qz.mean = self.h2z_mean(hidden[:, :, :, 0])
            qz.logvar = self.h2z_var(hidden[:, :, :, 1])
        elif self.mixing == 'interp_only':
            qz.mean = self.h2z_mean(hidden[:, :, :, 0])
            qz.logvar = self.h2z_var(hidden[:, :, :, 0])
        elif self.mixing == 'na':
            qz.mean = self.h2z_mean(hidden)
            qz.logvar = self.h2z_var(hidden)
        return qz
    
    def encode(self, context_x, context_y):
        mask = context_y[:, :, self.dim:]
        value = context_y[:, :, :self.dim]
        hidden = self.encoder(self.query, context_x, value, mask)
        return self.h2z_mixing(hidden), hidden

    def decode(self, z, target_x):
        px = Gaussian()
        num_sample, batch, seqlen, dim = z.size()
        z = z.view(num_sample * batch, seqlen, dim)
        if target_x.ndim == 1:
            target_x = target_x[None, :].repeat(batch, 1)
        target_x = target_x[None, :, :].repeat(
            num_sample, 1, 1).view(-1, target_x.shape[1])
        hidden = self.decoder(target_x, self.query, z)
        hidden = self.z2o(hidden)
        pred_mean = hidden[:, :, :self.dim]
        pred_logvar = self.compute_logvar(hidden[:, :, self.dim:])
        px.mean = pred_mean.view(
            num_sample, -1, pred_mean.size(1), pred_mean.size(2))
        px.logvar = pred_logvar.view(
            num_sample, -1, pred_logvar.size(1), pred_logvar.size(2))
        return px

    def sample(self, dist, num_sample=1):
        mean, logvar = dist.mean, dist.logvar
        epsilon = torch.randn(
            num_sample, mean.size(0), mean.size(1), mean.size(2)
        ).to(self.device)
        return epsilon * torch.exp(.5 * logvar) + mean

    def generate(self, t, num_samples=1):
        z = torch.randn(num_samples, self.num_ref_points, self.latent_dim)
        z = z.to(self.device)
        return self.decode(z, t)

    def get_reconstruction(self, context_x, context_y, target_x, num_samples=1,qz_mean=False):
        qz, hidden = self.encode(context_x, context_y)
        z = self.sample(qz, num_samples)
        if self.mixing == 'interp_only':
            hidden = self.proj(hidden[:, :, :, 0])
        else:
            hidden = torch.cat((hidden[:, :, :, 0], hidden[:, :, :, 1]), -1)
            hidden = self.proj(hidden)
    
        hidden = hidden.unsqueeze(0).repeat_interleave(num_samples, dim=0)
        
        if qz_mean:
            z = qz.mean.unsqueeze(0).repeat_interleave(num_samples, dim=0)
            z = torch.cat((z, hidden), -1)
        else:
            z = torch.cat((z, hidden), -1)
            
        px = self.decode(z, target_x)
        return px, qz

    def compute_logvar(self, sigma):
        if self.is_constant:
            sigma = torch.zeros(sigma.size()) + self.std
        elif self.is_bounded:
            sigma = 0.01 + F.softplus(sigma)
        elif self.is_constant_per_dim:
            sigma = 0.01 + F.softplus(sigma)
        else:
            return sigma
        return 2 * torch.log(sigma).to(self.device)

    def kl_div(self, qz, mask=None, norm=True):
        pz_mean = pz_logvar = torch.zeros(qz.mean.size()).to(self.device)
        kl = utils.normal_kl(qz.mean, qz.logvar, pz_mean,
                             pz_logvar).sum(-1).sum(-1)
        if norm:
            return kl / mask.sum(-1).sum(-1)
        return kl

    def compute_loglik(self, target_y, px, norm=True, logerr=0.):
        target, mask = target_y[:, :, :self.dim], target_y[:, :, self.dim:]
        log_p = utils.log_normal_pdf(
            target, px.mean, px.logvar, mask, logerr=logerr)
        log_p = log_p.sum(-1).sum(-1)
        if norm:
            return log_p / mask.sum(-1).sum(-1)
        return log_p

    def compute_mog_loglik(self, target_y, px):
        target, mask = target_y[:, :, :self.dim], target_y[:, :, self.dim:]
        loglik = utils.mog_log_pdf(target, px.mean, px.logvar, mask)
        return loglik.sum() / mask.sum()

    def compute_mse(self, target_y, pred, weights=1.):
        target, mask = target_y[:, :, :self.dim], target_y[:, :, self.dim:]
        return utils.mean_squared_error(target, pred, mask, weights=weights) / pred.size(0)

    def compute_mae(self, target_y, pred):
        target, mask = target_y[:, :, :self.dim], target_y[:, :, self.dim:]
        return utils.mean_absolute_error(target, pred, mask) / pred.size(0)

    def compute_mean_mse(self, target_y, pred, weights):
        target, mask = target_y[:, :, :self.dim], target_y[:, :, self.dim:]
        return utils.mean_squared_error(target, pred.mean(0), mask, weights)

    def compute_mean_mae(self, target_y, pred):
        target, mask = target_y[:, :, :self.dim], target_y[:, :, self.dim:]
        return utils.mean_absolute_error(target, pred.mean(0), mask)

    def compute_unsupervised_loss(
        self, context_x, context_y, target_x, target_y, logerr, weights, num_samples=1, beta=1.,qz_mean=False, f1=100.0,f2=10.0,
    ):
        loss_info = LossInfo()
        px, qz = self.get_reconstruction(
            context_x, context_y, target_x, num_samples,qz_mean=qz_mean)
        mask = target_y[:, :, self.dim:]
        loglik = self.compute_loglik(target_y, px, self.norm)
        wloglik = self.compute_loglik(target_y, px, self.norm, logerr=logerr)
        
        kl = self.kl_div(qz, mask, self.norm)
        loss_info.elbo = -(
            torch.logsumexp(loglik- beta * kl, dim=0).mean(0) - np.log(num_samples))
        
        loss_info.welbo = -(
            torch.logsumexp((wloglik * (1/f1)) - beta * kl, dim=0).mean(0) - np.log(num_samples))
        loss_info.kl = kl.mean()
        loss_info.loglik = loglik.mean()
        loss_info.loglik_per_ex = loglik
        loss_info.wloglik = wloglik.mean() * (1/f1)
        loss_info.mse = self.compute_mse(target_y, px.mean, 1.)
        loss_info.wmse = (self.compute_mse(target_y, px.mean, weights=weights)) * 1/f2
        loss_info.mae = self.compute_mae(target_y, px.mean)
        loss_info.mean_mse = self.compute_mean_mse(
            target_y, px.mean, weights)
        loss_info.mean_mae = self.compute_mean_mae(target_y, px.mean)
        loss_info.mogloglik = self.compute_mog_loglik(target_y, px)
        loss_info.composite_loss = self.elbo_weight * loss_info.elbo \
            + self.mse_weight * loss_info.mse
        loss_info.weighted_comp_loss = self.elbo_weight * loss_info.welbo \
            + self.mse_weight * loss_info.wmse 
        return loss_info

    
    
class HeTVAE_nodet(HeTVAE):
    '''Heteroscedastic Temporal Variational Autoencoder, no det pathway'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.decoder = UnTAN(
            input_dim=self.latent_dim,
            nhidden=self.nhidden,
            embed_time=self.embed_time,
            num_heads=self.num_heads,
            dropout=self.dropout,
            device=self.device
        )
 
    def encode(self, context_x, context_y):
        mask = context_y[:, :, self.dim:]
        value = context_y[:, :, :self.dim]
        hidden = self.encoder(self.query, context_x, value, mask)
        return self.h2z_mixing(hidden)

    def get_reconstruction(self, context_x, context_y, target_x, num_samples=1, qz_mean=False):
        qz = self.encode(context_x, context_y)
        z = self.sample(qz, num_samples)
        px = self.decode(z, target_x)
        return px, qz
