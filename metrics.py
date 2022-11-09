import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# lower --> more similar images
def pixel_to_pixel_MSE(original, noisy): 
    return np.mean((original - noisy)**2)

# higher --> more similar images
def PSNR(original, noisy):
    return psnr(original, noisy)

# higher --> more similar images
def SSIM(original, noisy):
    return ssim(original, noisy)
