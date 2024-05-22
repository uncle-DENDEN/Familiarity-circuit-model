import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
import random
import seaborn as sns
import scipy.signal as ss
from scipy.special import i0
from scipy.ndimage import convolve1d
import scipy.io as spio
from matplotlib import rc as matrc
import torch.nn as nn
import torch.nn.functional as f
import torch.distributions as td
import torch
import cv2
import os, sys
import math
from einops import rearrange


matrc('animation', html='jshtml', embed_limit=20971520 * 2)
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.pardir))

##################################################################################################################
## model functions
##################################################################################################################
def pooling(mat, ksize, method='max', pad=False):
    """Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    """

    m, n = mat.shape[:2]
    ky, kx = ksize

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m, ky)
        nx = _ceil(n, kx)
        size = (ny * ky, nx * kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        ny = m // ky
        nx = n // kx
        mat_pad = mat[:ny * ky, :nx * kx, ...]

    new_shape = (ny, ky, nx, kx) + mat.shape[2:]

    if method == 'max':
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
    else:
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))

    return result


# Constructing the Gabor filters
def gabor2D_Lee(size, orientation, sigma_deno, spat_freq):
    num_orient = len(orientation)

    # Generate 2d grid of a Gabor wavelet
    x_arr = np.arange(size[0]) - (size[0] + 1) / 2 + 1
    y_arr = np.arange(size[1]) - (size[1] + 1) / 2 + 1

    # Generate a 4D Tensor to hold all the Gabor filters
    x_arr = np.arange(size[0]) - (size[0] + 1) / 2 + 1
    y_arr = np.arange(size[1]) - (size[1] + 1) / 2 + 1
    x_arr = np.tile(x_arr, size[0])  # horizontal steps
    y_arr = np.repeat(y_arr, size[1])  # vertical steps
    gabor_arr_cos = np.zeros((size[0], size[0], num_orient))
    gabor_arr_sin = np.zeros((size[0], size[0], num_orient))

    # Generate Gabor wavelets
    for i in range(num_orient):
        gabor_arr_cos[:, :, i] = np.reshape(gaborFunc_Lee(x_arr, y_arr, size, orientation[i], sigma_deno, spat_freq)[0],
                                            (size[0], size[1]))
        gabor_arr_sin[:, :, i] = np.reshape(gaborFunc_Lee(x_arr, y_arr, size, orientation[i], sigma_deno, spat_freq)[1],
                                            (size[0], size[1]))

    return [gabor_arr_cos, gabor_arr_sin]


def gaborFunc_Lee(x, y, size, theta, sigma_deno, spat_freq):
    # Gaussian envelope of Gabor function
    sigma = size[0] / sigma_deno

    gaussEnv = 4 * (x * np.cos(theta) + y * np.sin(theta)) ** 2 + (
            -x * np.sin(theta) + y * np.cos(theta)) ** 2  # 4*a_i2 + a_j2
    gaussEnv = np.exp(- 1 / (sigma ** 2 * 8.0) * gaussEnv)

    # Sinusoidal part of Gabor function
    cplxPart = spat_freq * (x * np.cos(theta) + y * np.sin(theta))
    cplxPart_cos = np.cos(cplxPart)
    cplxPart_sin = np.sin(cplxPart)

    z_cos = gaussEnv * cplxPart_cos
    z_sin = gaussEnv * cplxPart_sin

    return [z_cos, z_sin]


def choosing_neighbors(i, num_kernel, num_row, num_col, radius):
    # (kernel, row, column) -> flatten
    N_e = num_kernel * num_row * num_col
    kernel_num = int(i / (num_row * num_col))
    row_num = int((i - kernel_num * num_row * num_col) / num_col)
    col_num = int((i - kernel_num * num_row * num_col - row_num * num_col))
    same_hypercolumn = np.arange(col_num + row_num * num_col, N_e, num_row * num_col)
    result = np.array([], dtype=int)

    for col in range(int(2 * radius + 1)):
        if 0 <= col_num - radius + col <= num_col - 1:
            for row in range(int(2 * radius + 1)):
                if 0 <= row_num - radius + row <= num_row - 1:
                    result = np.append(result,
                                       np.arange(col_num - radius + col + (row_num - radius + row) * num_col, N_e,
                                                 num_row * num_col, dtype=int))
    return result


def choosing_inhibition_neighbors(i, num_kernel, num_row, num_col, radius_inhibition):
    # (kernel, row, column) -> flatten
    # including neuron i itself
    N_e = num_kernel * num_row * num_col
    kernel_num = int(i / (num_row * num_col))
    row_num = int((i - kernel_num * num_row * num_col) / num_col)
    col_num = int((i - kernel_num * num_row * num_col - row_num * num_col))
    same_hypercolumn = np.arange(col_num + row_num * num_col, N_e, num_row * num_col)
    result = np.array([], dtype=int)

    # vertical norm between different kernel
    result = np.append(result, np.arange(col_num + row_num * num_col, N_e, num_row * num_col, dtype=int))
    
    # horizontal norm between different hypercoloumn (isoorientation suppression)
    for col in range(int(2 * radius_inhibition + 1)):
        if 0 <= col_num - radius_inhibition + col <= num_col - 1:
            for row in range(int(2 * radius_inhibition + 1)):
                if 0 <= row_num - radius_inhibition + row <= num_row - 1:
                    result = np.append(result, kernel_num * num_row * num_col + col_num - radius_inhibition + col + \
                                       (row_num - radius_inhibition + row) * num_col)
    return result


def take_center(y, num_kernel, num_row, num_col, center = 3):
    # only take neurons in the center 3-by-3 hypercolumns
    # (kernel, row, column) -> flatten
    assert len(y) == num_kernel * num_row * num_col
    yr = np.reshape(y, (num_kernel, num_row * num_col))
    no_hyper = np.array([], dtype = int)
    for i in range(center):
        no_hyper = np.append(no_hyper, (num_row - center) // 2 * num_col + (num_col - center) // 2 + \
                             np.arange(center) + i * num_col)
    return yr[:, no_hyper].flatten()


def initialization(num_layers, N_e, N_i):
    ue = []
    ui = []
    re = []
    ri = []
    duedt = []
    duidt = []
    dwrpdt = []

    for i in range(num_layers):
        ue.append(np.zeros(N_e[i]))
        re.append(np.zeros(N_e[i]))
        ui.append(np.zeros(N_i[i]))
        ri.append(np.zeros(N_i[i]))
        duedt.append(np.zeros(N_e[i]))
        duidt.append(np.zeros(N_i[i]))
        dwrpdt.append(np.zeros(N_e[i] * N_e[i]))

    return [ue, ui, re, ri, duedt, duidt, dwrpdt]


def initialization_between(num_layers, N_e):
    dwffdt = []
    dwfbdt = []

    for i in range(num_layers - 1):
        dwffdt.append(np.zeros(N_e[i] * N_e[i + 1]))
        dwfbdt.append(np.zeros(N_e[i] * N_e[i + 1]))

    return [dwffdt, dwfbdt]


##################################################################################################################
## Noise experiment functions
##################################################################################################################
def insertTeachingSig(r_in, p=2):
    """
    Insert teaching signal (pure image) after every p-1 noise corrupts sample. Specific for the noise exp
    
    r_in: Input sparse code or image, with shape (n_img, nl, np, ...)
    """
    imgs, nl, n = r_in.shape[0:3]
    u_shape = r_in.shape[3:]
    
    teaching_sig = r_in[:, 0, [0]]  # 20, 1, 64, 8, 8
    nl -= 1
    # noisec = r_in[:, 1:].reshape(imgs, nl*n, nc, h, w)  # 20, 90, 64, 8, 8
    noisec = rearrange(r_in[:, 1:], "i nl np a b c -> i (nl np) a b c")

    # interweaving teaching sig and noise
    c = np.empty((imgs, nl*n + nl*n//(p-1), *u_shape))
    teaching_ind = np.arange(p-1, nl*n + nl*n//(p-1), p)
    noise_ind = np.setdiff1d(np.arange(0, nl*n + nl*n//(p-1)), teaching_ind)
    assert len(noise_ind) == nl*n
    c[:, noise_ind] = noisec
    c[:, teaching_ind] = teaching_sig
    
    return c, teaching_ind, noise_ind

def insertTeachingSig_v2(r_in, p=2):
    """
    Insert teaching signal (pure image) after every p-1 noise corrupts sample. Specific for the noise exp.
    v2: for each image, add clear image before noisy-clear pairs.
    
    r_in: Input sparse code or image, with shape (n_img, nl, np, ...)
    """
    imgs, nl, n = r_in.shape[0:3]
    u_shape = r_in.shape[3:]
    
    teaching_sig = r_in[:, 0, [0]]  # 20, 1, 64, 8, 8
    nl -= 1
    # noisec = r_in[:, 1:].reshape(imgs, nl*n, nc, h, w)  # 20, 90, 64, 8, 8
    noisec = rearrange(r_in[:, 1:], "i nl np a b c -> i (nl np) a b c")

    # interweaving teaching sig and noise
    c = np.empty((imgs, nl*n + nl*n//(p-1) + p, *u_shape))
    teaching_ind = np.concatenate([np.arange(p), np.arange(2*p-1, nl*n + nl*n//(p-1) + p, p)]) 
    noise_ind = np.setdiff1d(np.arange(0, nl*n + nl*n//(p-1) + p), teaching_ind)
    assert len(noise_ind) == nl*n
    c[:, noise_ind] = noisec
    c[:, teaching_ind] = teaching_sig

    return c, teaching_ind, noise_ind

def seqUnmix(ys, t_ind, n_ind, n_imgs, npa, temporal_seq=True, flatten_imgs=True):
    """Extract pure image from the mixed training sequence. Specific for the noise exp"""
    # ys = ys.reshape(n_imgs, -1, ts, N)  # image, insert_ind, time, neurons
    if temporal_seq:
        ys = rearrange(ys, '(i x) ts n -> i x ts n', i=n_imgs)
    else:
        ys = rearrange(ys, '(i x) n -> i x n', i=n_imgs)
    
    ys_image = np.expand_dims(np.repeat(ys[:, t_ind[[0]]], npa, axis=1), 1)  # imgs, 1, 10, time, neurons
    # ys_noise = ys[:, n_ind].reshape(n_imgs, nl-1, npa, ts, N)
    if temporal_seq:
        ys_noise = rearrange(ys[:, n_ind], 'i (nl np) ts n -> i nl np ts n', np=npa)  # imgs, nl-1, np, time, neurons
    else:
        ys_noise = rearrange(ys[:, n_ind], 'i (nl np) n -> i nl np n', np=npa)  # imgs, nl-1, np, time, neurons
    
    ys = np.concatenate([ys_image, ys_noise], 1)  # imgs, nl, np, time, neurons
    
    if flatten_imgs:
        ys = rearrange(ys, 'i nl np ts n -> (i nl np) ts n') if temporal_seq else rearrange(ys, 'i nl np n -> (i nl np) n')
    
    return ys
