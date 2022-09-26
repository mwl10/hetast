# pylint: disable=E1101
import torch
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_normal_pdf(x, mean, logvar, mask, logerr=0.):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    logerr = logerr * mask
    logvar = logvar + logerr
    return -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar)) * mask
   
def mog_log_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    const2 = torch.from_numpy(np.array([mean.size(0)])).float().to(x.device)
    loglik = -0.5 * (const + logvar + (x - mean) **
                     2.0 / torch.exp(logvar)) * mask

    return torch.logsumexp(loglik - torch.log(const2), 0)

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.0
    lstd2 = lv2 / 2.0
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2)) - 0.5
    return kl

def mean_squared_error(orig, pred, mask, weights=1.):
    weights = weights * mask
    error = ((orig - pred) ** 2) * weights
    error = error * mask
    return error.sum() / mask.sum()

def mean_absolute_error(orig, pred, mask):
    error = torch.abs(orig - pred)
    error = error * mask
    return error.sum() / mask.sum()

