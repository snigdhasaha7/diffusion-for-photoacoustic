# utility functions

import numpy as np
import matplotlib.pyplot as plt 
from metrics import PSNR 


def plot_before_after(clean_images, imgs_before, imgs_after, title=""):
    assert(imgs_before.shape[0] == imgs_after.shape[0])
    fig, axs = plt.subplots(2, imgs_before.shape[0], figsize=(16, 5))
    for i, image in enumerate(imgs_before):
        im = axs[0][i].imshow(image.cpu().permute(1, 2, 0).squeeze())
        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
    for i, image in enumerate(imgs_after):
        im = axs[1][i].imshow(image.cpu().permute(1, 2, 0).squeeze())
        axs[1][i].set_xticks([])
        axs[1][i].set_yticks([])
        plt.colorbar(im, ax=axs[1,i])
        if clean_images is not None:
            clean = clean_images[i].cpu().permute(1,2,0).squeeze()
            noisy = image.cpu().permute(1,2,0).squeeze()
            psnr_val = PSNR(clean, noisy).item()
            axs[1][i].set_title('PSNR: {:.3f}'.format(psnr_val), y=-0.2)
    fig.suptitle(title, size=20)


def awgn(x, snr=30):
  # Adding white gaussian noise
  # snr in dB
  signal_power = np.mean(x ** 2)
  noise_power = signal_power / (10 ** (snr / 10.0))
  noise = np.random.normal(scale=np.sqrt(noise_power), size=x.shape)
  x_with_noise = x + noise
  return x_with_noise


def limited_view_rmd(N_transducer, N_keep):
  # the parity of N_transducer and N_keep must match
  edge = (N_transducer - N_keep) / 2
  return [i for i in range(N_transducer) if not (i < edge or i >= (N_transducer - edge))]


def spatial_alias_rmd(N_transducer, skip):
  return [i for i in range(0, N_transducer, skip)]