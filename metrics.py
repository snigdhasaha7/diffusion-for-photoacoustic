import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
# lower --> more similar images
def pixel_to_pixel_MSE(original, noisy): 
    return np.mean((original - noisy)**2)

# higher --> more similar images
def PSNR(original, noisy):
    mse = pixel_to_pixel_MSE(original, noisy)
    if mse == 0:
        return 100 
    max_pixel = 255.0
    return 20 * log10(max_pixel / sqrt(mse))

# higher --> more similar images
def SSIM(original, noisy):
    return ssim(original, noisy)
