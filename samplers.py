import torch
import tqdm
import numpy as np
from scipy import integrate
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy.stats import norm

#----------------------------------------------
#          Utils 
#----------------------------------------------

def plot_images(images):
    #denoised = denoised.clamp(0.0, 1.0)
    sample_grid = make_grid(images, nrow=int(np.sqrt(images.shape[0])))
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(sample_grid.cpu().permute(1, 2, 0).squeeze())
    plt.show()

def get_y_t(images, t, marginal_prob_std):
    # vector of t
    ts = t * torch.ones(images.shape[0], device=images.device)
    ts = ts[:, None, None, None]
    
    # sample some noise
    z = torch.randn_like(images)

    # perturb at level t
    std = marginal_prob_std(t=ts)
    perturbed_images = images + z * std

    return perturbed_images

def lbda_scheduler(t, lbda, schedule="constant", param=1):
    if schedule == "constant":
        lbda = lbda
    elif schedule == "linear":
        param = torch.tensor(param)
        f_t = param*t
        lbda = lbda * f_t
    elif schedule == "exponential":
        param = torch.tensor(param)
        f_t = (torch.exp(param*t) - 1) / (torch.exp(param) - 1)
        lbda = lbda * f_t
    elif schedule == "relu":
        slope, pivot = param
        lbda = max(0, slope * (t - pivot) * lbda)
    elif schedule == "sigmoid":
        slope, pivot = param
        lbda = lbda / (1 + torch.exp(-slope  * (t - pivot)))
    elif schedule == "bell":
        mean, std = param
        lbda = norm(mean, std).pdf(t) * lbda
    
    return lbda

def pseudo_right_inverse(A):
    # A^-1 = A^T(AA^T)^-1
    return torch.matmul(torch.t(A), torch.inverse(torch.matmul(A, A.T)))

def pseudo_left_inverse(A):
    # A^-1 = (A^TA)^-1 A^T
    return torch.matmul(torch.inverse(torch.matmul(A.T, A)), A.T)

def condition_on_y(raw_images, x_t, t, marginal_prob_std, lbda=.5, lbda_param=1, lbda_schedule='constant'):
    y_t = get_y_t(raw_images, t, marginal_prob_std)
    lbda = lbda_scheduler(t, lbda, schedule=lbda_schedule, param=lbda_param)
    x_t_prime = lbda * y_t + (1 - lbda) * x_t
    return x_t_prime

def condition_on_inpainted_y(raw_images, x_t, t, marginal_prob_std, subsampling_L, lbda=.5, lbda_param=1, lbda_schedule='constant'):
    y_t = get_y_t(raw_images, t, marginal_prob_std)
    lbda = lbda_scheduler(t, lbda, schedule=lbda_schedule, param=lbda_param)
    P, T = [torch.eye(raw_images.shape[-2] * raw_images.shape[-1],device=x_t.device)] * 2
    L = subsampling_L
    # turn images into column vectors
    flat_y_t = torch.unsqueeze(torch.flatten(y_t, start_dim=1), dim=2)
    flat_x_t = torch.unsqueeze(torch.flatten(x_t, start_dim=1), dim=2)
    # x_prime is a weighted function of x and y
    y_influence = lbda * torch.matmul(L, torch.matmul(torch.inverse(P), flat_y_t))
    x_influence = (1 - lbda) * torch.matmul(L, torch.matmul(T, flat_x_t)) + \
                  torch.matmul(torch.eye(L.shape[0], device=L.device) - L,
                               torch.matmul(T, flat_x_t))
    x_t_prime = torch.reshape(y_influence + x_influence, x_t.shape)
    return x_t_prime

def condition_on_gauss_sub_y(raw_images, x_t, t, marginal_prob_std, operator_P, subsampling_L, transformation_T, lbda=.5, lbda_param=1, lbda_schedule='constant'):
    y_t = get_y_t(raw_images, t, marginal_prob_std)
    lbda = lbda_scheduler(t, lbda, schedule=lbda_schedule, param=lbda_param)
    P, L, T = operator_P, subsampling_L, transformation_T
    # turn images into column vectors
    flat_y_t = torch.unsqueeze(torch.flatten(y_t, start_dim=1), dim=2)
    flat_x_t = torch.unsqueeze(torch.flatten(x_t, start_dim=1), dim=2)
    # x_prime is a weighted function of x and y
    y_influence = lbda * torch.matmul(L, torch.matmul(pseudo_right_inverse(P), flat_y_t))
    x_influence = (1 - lbda) * torch.matmul(L, torch.matmul(T, flat_x_t)) + \
                  torch.matmul(torch.eye(L.shape[0], device=L.device) - L,
                               torch.matmul(T, flat_x_t))
    x_t_prime = torch.matmul(torch.inverse(T), y_influence + x_influence)
    x_t_prime = torch.reshape(x_t_prime, x_t.shape)
    return x_t_prime

def condition_on_pat_y(raw_images, x_t, t, marginal_prob_std, operator_P, subsampling_L, transformation_T, lbda=.5, lbda_param=1, lbda_schedule='constant'):
    y_t = get_y_t(raw_images, t, marginal_prob_std)
    lbda = lbda_scheduler(t, lbda, schedule=lbda_schedule, param=lbda_param)
    P, L, T = operator_P, subsampling_L, transformation_T
    # turn images into column vectors
    flat_y_t = torch.unsqueeze(torch.flatten(y_t, start_dim=1), dim=2)
    flat_x_t = torch.unsqueeze(torch.flatten(x_t, start_dim=1), dim=2)
    # x_prime is a weighted function of x and y
    y_influence = lbda * torch.matmul(L, torch.matmul(P, flat_y_t))
    x_influence = (1 - lbda) * torch.matmul(L, torch.matmul(T, flat_x_t)) + \
                  torch.matmul(torch.eye(L.shape[0], device=L.device) - L,
                               torch.matmul(T, flat_x_t))
    x_t_prime = torch.matmul(pseudo_left_inverse(T), y_influence + x_influence)
    x_t_prime = torch.reshape(x_t_prime, x_t.shape)
    return x_t_prime

def psnr(clean, noisy):
    # our range of values is [0.,1.]
    eps = 1e-8
    mse = torch.mean((clean - noisy) ** 2)
    return - 10 * torch.log10(mse + eps)

#----------------------------------------------
#          Samplers for denoising 
#----------------------------------------------

def pc_denoiser(raw_images,
               score_model,
               lbda, 
               marginal_prob_std,
               diffusion_coeff,
               drift_coeff=None,
               task='denoise',
               lbda_schedule='constant',
               lbda_param=1,
               operator_P=None,
               subsampling_L=None,
               transformation_T=None,
               num_steps=500,
               report_PSNR=False,
               ipython=False,
               clean_images=None,
               snr=0.16,                
               device='cuda',
               eps=1e-3):
    num_images = len(raw_images)
    t = torch.ones(num_images, device=device)
    init_x = torch.randn(num_images, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    if report_PSNR == True: metrics=[]
    if ipython == True: 
        plot_time_track = 0
        plot_step = num_steps / 10 
    with torch.no_grad():
        for time_step in tqdm.notebook.tqdm(time_steps):      
            batch_time_step = torch.ones(num_images, device=device) * time_step
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            if drift_coeff == None:
                x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            else:
                f = drift_coeff(x, batch_time_step)
                x_mean = x + ( -1 * f + ((g**2)[:, None, None, None] * score_model(x, batch_time_step)) ) * step_size

            # Condition on y_t
            if task == 'denoise':
                x_mean = condition_on_y(raw_images, x_mean, time_step, marginal_prob_std, lbda, lbda_param, lbda_schedule)
            elif task == 'depaint':
                x_mean = condition_on_inpainted_y(raw_images, x_mean, time_step, marginal_prob_std, subsampling_L, lbda, lbda_param, lbda_schedule)
            elif task == 'degaussub':
                x_mean = condition_on_gauss_sub_y(raw_images, x_mean, time_step, marginal_prob_std, operator_P, subsampling_L, transformation_T, lbda, lbda_param, lbda_schedule)
            elif task == 'depat':
                x_mean = condition_on_pat_y(raw_images, x_mean, time_step, marginal_prob_std, operator_P, subsampling_L, transformation_T, lbda, lbda_param, lbda_schedule)

            # Compute metrics
            if report_PSNR == True:
                
                metrics.append([psnr(torch.squeeze(clean), torch.squeeze(noisy)).item() for (clean, noisy) in zip(clean_images,x_mean)])
            
            if ipython == True:
                if plot_time_track % plot_step == 0 or plot_time_track >= (num_steps - 10):
                    ipd.clear_output(wait=True)
                    fig = plt.imshow(x_mean.cpu()[0].squeeze())
                    plt.title(f'Step: {time_step}')
                    plt.colorbar()
                    plt.show()
                plot_time_track += 1

            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)

    # The last step does not include any noise
    if report_PSNR == True:
        return (x_mean, metrics)
    return x_mean


def Euler_Maruyama_denoiser(raw_images,
                           score_model,
                           lbda, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           drift_coeff=None,
                           lbda_schedule='constant',
                           lbda_param=1,
                           operator_P=None,
                           subsampling_L=None,
                           transformation_T=None,
                           num_steps=500,
                           report_PSNR=False,
                           ipython=False,
                           device='cuda', 
                           eps=1e-3):
    num_images = len(raw_images)
    t = torch.ones(num_images, device=device)
    init_x = torch.randn(num_images, 1, 28, 28, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    if ipython == True: 
        plot_time_track = 0
        plot_step = num_steps / 10 
    with torch.no_grad():
        for time_step in tqdm.notebook.tqdm(time_steps):      
            batch_time_step = torch.ones(num_images, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            if drift_coeff == None:
                mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            else:
                f = drift_coeff(x, batch_time_step)
                mean_x = x + ( -1 * f + ((g**2)[:, None, None, None] * score_model(x, batch_time_step)) ) * step_size
            
            # Condition on y_t
            if task == 'denoise':
                x_mean = condition_on_y(raw_images, x_mean, time_step, marginal_prob_std, lbda, lbda_param, lbda_schedule)
            elif task == 'depaint':
                x_mean = condition_on_inpainted_y(raw_images, x_mean, time_step, marginal_prob_std, subsampling_L, lbda, lbda_param, lbda_schedule)
            elif task == 'degaussub':
                x_mean = condition_on_gauss_sub_y(raw_images, x_mean, time_step, marginal_prob_std, operator_P, subsampling_L, transformation_T, lbda, lbda_param, lbda_schedule)
            elif task == 'depat':
                x_mean = condition_on_pat_y(raw_images, x_mean, time_step, marginal_prob_std, operator_P, subsampling_L, transformation_T, lbda, lbda_param, lbda_schedule)

            if ipython == True:
                if plot_time_track % plot_step == 0 or plot_time_track >= (num_steps - 10):
                    ipd.clear_output(wait=True)
                    fig = plt.imshow(x_mean.cpu()[0].squeeze())
                    plt.title(f'Step: {time_step}')
                    plt.colorbar()
                    plt.show()
                plot_time_track += 1
            
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
    # Do not include any noise in the last sampling step.
    return mean_x


#----------------------------------------------
#          Samplers for image generation 
#----------------------------------------------

def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           drift_coeff=None,
                           batch_size=64, 
                           num_steps=500, 
                           device='cuda', 
                           eps=1e-3):
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm.notebook.tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            if drift_coeff == None:
                mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            else:
                f = drift_coeff(x, batch_time_step)
                mean_x = x + ( -1 * f + ((g**2)[:, None, None, None] * score_model(x, batch_time_step)) ) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
    # Do not include any noise in the last sampling step.
    return mean_x

def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               drift_coeff=None,
               batch_size=64, 
               num_steps=500, 
               snr=0.16,                
               device='cuda',
               eps=1e-3):
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm.notebook.tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

            # Predictor step (Euler-Maruyama)
            # TODO revisit this for VP
            g = diffusion_coeff(batch_time_step)
            if drift_coeff == None:
                x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            else:
                f = drift_coeff(x, batch_time_step)
                x_mean = x + ( -1 * f + ((g**2)[:, None, None, None] * score_model(x, batch_time_step)) ) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

    # The last step does not include any noise
    return x_mean

def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                drift_coeff=None,
                batch_size=64, 
                atol=1e-5, 
                rtol=1e-5, 
                device='cuda', 
                z=None,
                eps=1e-3):
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if z is None:
        init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
          * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z
    shape = init_x.shape
    
    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
        with torch.no_grad():    
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):        
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t    
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        f = drift_coeff(x, torch.tensor(t)).cpu().numpy()
        return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps) + f

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
    return x
