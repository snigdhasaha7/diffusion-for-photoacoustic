import numpy as np
import torch


#----------------------------------------------
#          Classic SDE (Classic)
#----------------------------------------------
class Classic():
    def __init__(self):
        pass

    # TODO
    def marginal_prob_std(self, x, t, sigma):
        var = torch.sqrt((sigma**(2 * t) - 1.) / (2. * np.log(sigma)))
        mean = torch.zeros_like(x)
        return mean, var

    def drift_coeff(self, x, t, sigma):
        drift = torch.zeros_like(torch.tensor(x, device=t.device))
        return drift

    def diffusion_coeff(self, t, sigma):
        diffusion = torch.tensor(sigma**t, device=t.device)
        return diffusion


#----------------------------------------------
#          Variance Exploding (VE)
#----------------------------------------------
class VarianceExploding():
    def __init__(self):
        pass
    
    # TODO
    def marginal_prob_std(self, x, t, sigma_min, sigma_max):
        # def of sigma from Yang Song's repo score_sde
        sigma = sigma_min * (sigma_max / sigma_min)**t
        mean = x
        var = sigma
        return mean, var

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
    
    # TODO
    def marginal_prob_std(self, x, t, beta_min, beta_max):
        coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
        mean = torch.exp(coeff[:, None, None, None]) * x
        var = torch.sqrt(1 - torch.exp(2. * coeff))
        return mean, var

    def drift_coeff(self, x, t, beta_min, beta_max):
        # def of beta_t from Yang Song's repo score_sde
        beta_t = beta_min + t * (beta_max - beta_min)
        if (beta_t.dim() == 1) and (x.dim() > 1):
            beta_t = beta_t[:, None, None, None]
        drift = -.5 * beta_t * x
        return drift

    def diffusion_coeff(self, t, beta_min, beta_max):
        # def of beta_t from Yang Song's repo score_sde
        beta_t = beta_min + t * (beta_max - beta_min)
        diffusion = torch.sqrt(beta_t)
        return diffusion


#----------------------------------------------
#          Sub Variance Preserving (subVP)
#----------------------------------------------
class SubVariancePreserving():
    def __init__(self):
        pass
    
    # TODO
    def marginal_prob_std(self, x, t, beta_min, beta_max):
        coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
        # mean = torch.exp(coeff[:, None, None, None]) * x
        std = 1 - torch.exp(2. * coeff)
        if len(x.shape) == 4:
            mean = torch.exp(coeff)[:, None, None, None] * x
        else:
            mean = torch.exp(coeff)[:, None] * x
            
        return mean, std

    def coeff_and_std(self, t, beta_min, beta_max):
        coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
        std = 1 - torch.exp(2. * coeff)
        return coeff, std

    def drift_coeff(self, x, t, beta_min, beta_max):
        # def of beta_t from Yang Song's repo score_sde
        beta_t = beta_min + t * (beta_max - beta_min)
        if (beta_t.dim() == 1) and (x.dim() > 1):
            beta_t = beta_t[:, None, None, None]
        drift = -.5 * beta_t * x
        return drift

    def diffusion_coeff(self, t, beta_min, beta_max):
        # def of beta_t from Yang Song's repo score_sde
        beta_t = beta_min + t * (beta_max - beta_min)
        discount = 1. - torch.exp(-2 * beta_min * t - (beta_max - beta_min) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return diffusion



#----------------------------------------------
#          Snigdha's SDE class 
#----------------------------------------------

# class SDE(): 
#     def __init__(self,device): 
#         self.device = device
    
#     def simp_marginal_prob_std(t,sigma):
#         t = torch.tensor(t, device=self.device)
#         std = torch.sqrt((sigma**(2 * t) - 1.) / (2. * np.log(sigma)))
#         return std
    
#     def simp_diffusion_coeff(t,sigma):
#         return torch.tensor(sigma**t, self.device)
    
#     # TODO
#     def ve_marginal_prob_std(t,sigma_min,sigma_max):
#         std = sigma_min * (sigma_max / sigma_min) ** t
#         return std

#     def ve_diffusion_coeff(t,sigma_min,sigma_max): 
#         sigma = sigma_min * (sigma_max / sigma_min) ** t
#         diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)),
#                                                     device=t.device))
#         return diffusion
    
#     # TODO
#     def subvp_marginal_prob_std(t, beta_min, beta_max):
#         log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
#         std = 1 - torch.exp(2. * log_mean_coeff)
#         return std

#     def subvp_diffusion_coeff(t, beta_min, beta_max):
#         beta_t = beta_min + t * (beta_max - beta_min)
#         discount = 1. - torch.exp(-2 * beta_min * t - (beta_max - beta_min) * t ** 2)
#         diffusion = torch.sqrt(beta_t * discount)
#         return diffusion

#     def subvp_drift_coeff(t, x, beta_min, beta_max):
#         beta_t = beta_min + t * (beta_max - beta_min)
#         if beta_t.dim() == 0:
#             drift = -0.5 * beta_t * x
#         else:
#             drift = -0.5 * beta_t[:, None, None, None] * x
#         return drift
    
#     # TODO
#     def vp_marginal_prob_std(t, beta_min, beta_max):
#         log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
#         std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
#         return std

#     def vp_diffusion_coeff(t, beta_min, beta_max):
#         beta_t = beta_min + t * (beta_max - beta_min)
#         diffusion = torch.sqrt(beta_t)
#         return diffusion

#     def vp_drift_coeff(t, x, beta_min, beta_max):
#         beta_t = beta_min + t * (beta_max - beta_min)
#         if beta_t.dim() == 0:
#             drift = -0.5 * beta_t * x
#         else:
#             drift = -0.5 * beta_t[:, None, None, None] * x
#         return drift