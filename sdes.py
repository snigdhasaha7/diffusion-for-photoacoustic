import numpy as np
import torch

#----------------------------------------------
#          Variance Exploding (VE)
#----------------------------------------------
class VarianceExploding():
    def __init__(self):
        pass
    
    def marginal_prob_std(self, t, sigma_min, sigma_max):
        # def of sigma from Yang Song's repo score_sde
        sigma = sigma_min * (sigma_max / sigma_min)**t
        #mean = x
        var = sigma
        return var

    def drift_coeff(self, x, t, sigma_min, sigma_max):
        drift = torch.zeros_like(torch.tensor(x, device=t.device))
        return drift

    def diffusion_coeff(self, t, sigma_min, sigma_max):
        # def of sigma from Yang Song's repo score_sde
        sigma = sigma_min * (sigma_max / sigma_min)**t
        diffusion = sigma * torch.sqrt(torch.tensor(2*(np.log(sigma_max) - np.log(sigma_min)), device=t.device))
        return diffusion


#----------------------------------------------
#          Variance Preserving (VP)
#----------------------------------------------
class VariancePreserving():
    def __init__(self):
        pass
    
    def marginal_prob_std(t,sigma):
        t = torch.tensor(t, device=device)
        var = torch.sqrt((sigma**(2 * t) - 1.) / (2. * np.log(sigma)))
        return var 

    def diffusion_coeff(t, sigma): 
        return torch.tensor(sigma**t, device=device)

    def loss_fn(model, x, marginal_prob_std, eps=1e-5):
        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
        z = torch.randn_like(x)
        std = marginal_prob_std(t=random_t)
        perturbed_x = x + z * std[:, None, None, None]
        score = model(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        return loss


#----------------------------------------------
#          Sub Variance Preserving (subVP)
#----------------------------------------------
class SubVariancePreserving():
    def __init__(self):
        pass
    
    def marginal_prob_std(self, t, sigma_min, sigma_max):
        var = None
        return var

    def drift_coeff(self, x, t, sigma_min, sigma_max):
        drift = None
        return drift

    def diffusion_coeff(self, t, sigma_min, sigma_max):
        diffusion = None
        return diffusion