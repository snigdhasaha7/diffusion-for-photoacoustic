import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# lower --> more similar images
def pixel_to_pixel_MSE(original, noisy): 
    return np.mean((original - noisy)**2)

# higher --> more similar images
def PSNR(original, noisy):
    if not isinstance(original, np.ndarray):
      original = original.cpu().numpy()
    if not isinstance(noisy, np.ndarray):
      noisy = noisy.cpu().numpy()
    return psnr(original, noisy)

def avg_PSNR(GT, preds):
  # GT and preds must have the same shape
  psnrs = []
  for i in range(len(GT)):
    psnrs.append(PSNR(GT[i], preds[i]))
  return sum(psnrs) / len(psnrs)

# higher --> more similar images
def SSIM(original, noisy):
    return ssim(original, noisy)
