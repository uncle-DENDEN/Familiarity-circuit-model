from utils import *
from Model_gpu import ProtoRNN

from itertools import combinations
from itertools import chain as ch
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
from numpy.linalg import norm
import scipy.linalg as linalg
import numpy as np
import pandas as pd
import joblib
import os
import re

import torch
from einops import rearrange, repeat

# params
input_dir = 'SMT-BCM_input_correct/r_in_cifar_all_noise_abs_5_4_10_64_8_8.npy'  # input used to get t_ind
ff_dir = 'SMT-BCM_input_correct/r_in_cifar_all_noise_abs_5_4_10_64_8_8.npy'  # feedforward activity, would be different from input if analyzing noise
u_pre_dir = 'SMT-BCM_response_correct/r_pre_rx1_1000ms_mean_5_4x10_50_64_8_8_mix50%_all.npy'
Jac_pre_dir = 'SMT-BCM_Jac_v3/pre/edc_jac_img0_clear_sample.pkl'
u_trained_dir = 'SMT-BCM_response_correct/r_epoch0_tx30_rx1_1000ms_mean_5_4x10_50_64_8_8_mix50%_2e9pos1e7_all.npy'
Jac_trained_dir = 'SMT-BCM_Jac_v3/epoch0/edc_jac_img0_clear_sample.pkl'
out_dir = 'SMT_BCM_proj_stats_v3_slow_nm=1-200_p'

postfix = {0: 'clear', 1: '10%', 2: '30%', 3: '50%'}
num_imgs = 5
# modes_ns = [250, 250, 300, 350, 250]
# modes_ns = [300, 250, 300, 500, 250]
modes_ns = [200, 200, 200, 200, 200]

# make dir
os.makedirs(out_dir, exist_ok=True)

# model
num_kernel = 64
num_row = 8
num_col = 8
tau_stim = 150
delta_t = 1 
save_every_stim = 15
f = ProtoRNN(num_kernel, num_row, num_col, tau_stim, delta_t, save_every_stim)
device = torch.device('cpu')
f.to(device)

# get t_ind for unmixing
mix_coef = 2
r_in = np.load(input_dir)
r_in, t_ind, n_ind = insertTeachingSig(r_in, p=mix_coef)


# path funcs
def get_my_epoch(input_string, e):
    epoch_pos = input_string.find('epoch')
    end_pos = epoch_pos + 5  # Start from the character after 'epoch'
    
    while end_pos < len(input_string) and input_string[end_pos].isdigit():
        end_pos += 1

    # Replace the number after 'epoch' with e
    new_string = input_string[:epoch_pos + 5] + str(e) + input_string[end_pos:]

    return new_string

def get_my_epoch_img_nl(path, img, nl, epoch=None):
    # Replace the number after 'epoch'
    if epoch is not None:
        path = re.sub(r'(epoch)\d+', rf'\g<1>{epoch}', path)
    
    # Replace the number after 'img'
    path = re.sub(r'(img)\d+', rf'\g<1>{img}', path)
    
    # Replace the percentage part
    path = re.sub(r'\d+%', f'{nl}', path)
    
    return path

# ------------------------------------------------------------------------------------------------------------------
# manifold functions
# ------------------------------------------------------------------------------------------------------------------

def analyze_eigenmodes_pre_v3_noise(img, num_modes=None):
    '''
    Better derivative measures 10%-0% at 0%, 30%-10% at 10%, 50%-30% at 30%. 
    Distance scale measured through noise
    '''
    ff = np.load(ff_dir)  # trial, n_img*nl*np, 16, Ne+Ni
    ff = ff[:, :, -1]  # temporally-uncorrelated, trial, n_img*nl*np, Ne+Ni

    u = np.load(u_pre_dir) 
    # u = seqUnmix(u, t_ind, n_ind, n_imgs=num_imgs, npa=10)  # n_img*nl*np, 16, Ne+Ni
    r = f.r_numpy(u)   # trial, n_img*nl*np, 76, Ne+Ni
    r_eq = r[:, :, -2:].mean(-2)  # trial, n_img*nl*np, Ne+Ni

    r_eq = rearrange(r_eq, 't (n l p) k -> t n l p k', n=num_imgs, l=4)
    ff = rearrange(ff, 't (n l p) k -> t n l p k', n=num_imgs, l=4)
    # take the corresponding image and noise pattern where linearization performed
    r_eq = r_eq[:, img, :, 0]  # t, l, Ne+Ni
    ff = ff[:, img, :, 0]  # t, l, Ne

    proj_output_all, proj_input_all = [], []
    # image specific eigenspectra
    for nl in [0, 1, 2]:
        Jac_path = get_my_epoch_img_nl(Jac_pre_dir, img, postfix[nl])
        eigd = joblib.load(Jac_path)
        # eigspec = eigd['val']
        vl = eigd['vl'][:f.N_e]
        vl_prime = eigd['vl_prime'][:f.N_e]

        # get effective dimension
        vl_prime_norm = np.linalg.norm(vl_prime, axis=0)
        eff_dim = np.where(vl_prime_norm == 0)[0][0]
        N = num_modes if num_modes is not None else eff_dim

        # slowest decaying mode
        mode_s = vl[:, :N]  # Ne+Ni, 20
        # mode_s = mode_s / norm(mode_s, axis=0, keepdims=True)
        # change of variable modes
        mode_s_prime = vl_prime[:, :N]  # Ne+Ni, 20
        # mode_s_prime = mode_s_prime / norm(mode_s_prime, axis=0, keepdims=True)

        # projecting output of adjacent noise level to the mode
        r_eq_exc = rearrange(r_eq[:, [nl, nl+1], :f.N_e], 't l k -> (t l) k')  # t*nl, Ne
        proj_output = mode_s.T @ r_eq_exc.T  # 20, t*nl

        # projecting input of adjacent noise level to the change of variable mode
        ff_ = rearrange(ff[:, [nl, nl+1], :], 't l k -> (t l) k')  # t*nl, Ne
        proj_input = mode_s_prime.T @ ff_.T  # 20, t*nl

        proj_output_all.append(proj_output)
        proj_input_all.append(proj_input)
        
    return proj_output_all, proj_input_all


def analyze_eigenmodes_trained_v3_noise(epoch, img, num_modes=None):
    '''
    Better derivative measures 10%-0% at 0%, 30%-10% at 10%, 50%-30% at 30%. 
    Distance scale measured through noise
    '''
    ff = np.load(ff_dir) # trial, n_img*nl*np, 16, Ne+Ni
    ff = ff[:, :, -1] # temporally-uncorrelated, trial, n_img*nl*np, Ne+Ni
    
    u_path = get_my_epoch(u_trained_dir, epoch)
    u = np.load(u_path)
    # u = seqUnmix(u, t_ind, n_ind, n_imgs=num_imgs, npa=10)  # n_img*nl*np, 16, Ne+Ni
    r = f.r_numpy(u)   # trial, n_img*nl*np, 16, Ne+Ni
    r_eq = r[:, :, -2:].mean(-2)  # trial, n_img*nl*np, Ne+Ni

    r_eq = rearrange(r_eq, 't (n l p) k -> t n l p k', n=num_imgs, l=4)
    ff = rearrange(ff, 't (n l p) k -> t n l p k', n=num_imgs, l=4)
    # take the corresponding image and noise pattern where linearization performed
    r_eq = r_eq[:, img, :, 0]  # t, l, Ne+Ni
    ff = ff[:, img, :, 0]  # t, l, Ne

    proj_output_all, proj_input_all = [], []
    # image specific eigenspectra
    for nl in [0, 1, 2]:
        Jac_path = get_my_epoch_img_nl(Jac_trained_dir, img, postfix[nl], epoch=epoch)
        eigd = joblib.load(Jac_path)
        # eigspec = eigd['val']
        vl = eigd['vl'][:f.N_e]
        vl_prime = eigd['vl_prime'][:f.N_e]

        # get effective dimension
        vl_prime_norm = np.linalg.norm(vl_prime, axis=0)
        eff_dim = np.where(vl_prime_norm == 0)[0][0]
        N = num_modes if num_modes is not None else eff_dim

        # slowest decaying mode
        mode_s = vl[:, :N]  # Ne+Ni, 20
        # mode_s = mode_s / norm(mode_s, axis=0, keepdims=True)
        # change of variable modes
        mode_s_prime = vl_prime[:, :N]  # Ne+Ni, 20
        # mode_s_prime = mode_s_prime / norm(mode_s_prime, axis=0, keepdims=True)

        # projecting output of adjacent noise level to the mode
        r_eq_exc = rearrange(r_eq[:, [nl, nl+1], :f.N_e], 't l k -> (t l) k')  # t*nl, Ne
        proj_output = mode_s.T @ r_eq_exc.T  # 20, t*nl

        # projecting input of adjacent noise level to the change of variable mode
        ff_ = rearrange(ff[:, [nl, nl+1], :], 't l k -> (t l) k')  # t*nl, Ne
        proj_input = mode_s_prime.T @ ff_.T  # 20, t*nl

        proj_output_all.append(proj_output)
        proj_input_all.append(proj_input)
        
    return proj_output_all, proj_input_all


def get_SNRs(r_list):
    '''
    get the signal to noise ratio. r should be a list of array, each of shape (n_modes, trial*2), containing all projected noise-perturbed 
    stationary state for 2 noise level, where linearization is performed around mean state of the first level. 
    Signal strength is quantified as difference between 2 mean states, and noise is quantified as the variance of perturbed states around 
    linearized mean state (perturned states of first noise level).
    '''
    ds_list = [1, 2, 2]
    assert len(r_list) == 3
    signals, varcs, ddcs = [], [], []
    for r, ds in zip(r_list, ds_list):
        r = rearrange(r, 'k (t l) -> t l k', l=2)  # trial, level, k
        r_mean = r.mean(0)  # level, k
        signal = norm((r_mean[0] - r_mean[1]) / ds)
        signals.append(signal)
        
        r_lin = r[:, 0]  # trial, k
        r_lin_mean = r_lin.mean(0, keepdims=True)
        varc = np.trace((r_lin - r_lin_mean).T @ (r_lin - r_lin_mean)) / r_lin.shape[0]
        varc = np.sqrt(varc)
        varcs.append(varc)

        ddc = signal / varc
        ddcs.append(ddc)
        
    return np.stack(signals), np.stack(varcs), np.stack(ddcs)

# ------------------------------------------------------------------------------------------------------------------
# SNR functions
# ------------------------------------------------------------------------------------------------------------------

def analyze_eigenmodes_pre_v3(img, num_modes=None):
    '''
    Better derivative measures 10%-0% at 0%, 30%-10% at 10%, 50%-30% at 30%. 
    '''
    ff = np.load(ff_dir) 
    ff = rearrange(ff, 'n l p k h w -> (n l p) (k h w)')  # n_img*nl*np, Ne

    u = np.load(u_pre_dir) 
    u = seqUnmix(u, t_ind, n_ind, n_imgs=num_imgs, npa=10)  # n_img*nl*np, 16, Ne+Ni
    r = f.r_numpy(u)  # n_img*nl*np, 16, Ne+Ni
    r_eq = r[:, -2:].mean(-2)  # n_img*nl*np, Ne+Ni

    r_eq = rearrange(r_eq, '(n l p) k -> n l p k', n=num_imgs, l=4)
    ff = rearrange(ff, '(n l p) k -> n l p k', n=num_imgs, l=4)
    # take the corresponding noise pattern where linearization performed
    r_eq = r_eq[:, :, 0]  # n, l, Ne+Ni
    ff = ff[:, :, 0]  # n, l, Ne

    proj_output_all, proj_input_all = [], []
    # image specific eigenspectra
    for nl in [0, 1, 2]:
        Jac_path = get_my_epoch_img_nl(Jac_pre_dir, img, postfix[nl])
        eigd = joblib.load(Jac_path)
        # eigspec = eigd['val']
        vl = eigd['vl']#[:f.N_e]
        vl_prime = eigd['vl_prime']#[:f.N_e]

        # get effective dimension
        vl_prime_norm = np.linalg.norm(vl_prime, axis=0)
        eff_dim = np.where(vl_prime_norm == 0)[0][0]
        N = num_modes if num_modes is not None else eff_dim

        # slowest decaying mode
        mode_s = vl[:, :N]  # Ne+Ni, 20
        # mode_s = vl[:, eff_dim: eff_dim+num_modes]
        # mode_s = vl[:, -num_modes:]
        # mode_s = mode_s / norm(mode_s, axis=0, keepdims=True)
        # change of variable modes
        mode_s_prime = vl_prime[:, :N]  # Ne+Ni, 20
        # mode_s_prime = vl_prime[:, eff_dim: eff_dim+num_modes]
        # mode_s_prime = vl_prime[:, -num_modes:]
        # mode_s_prime = mode_s_prime / norm(mode_s_prime, axis=0, keepdims=True)

        # projecting output of adjacent noise level to the mode
        # r_eq_exc = rearrange(r_eq[:, [nl, nl+1], :f.N_e], 'n l k -> (n l) k')  # n*nl, Ne
        r_eq_exc = rearrange(r_eq[:, [nl, nl+1]], 'n l k -> (n l) k')  # n*nl, Ne+Ni
        proj_output = mode_s.T @ r_eq_exc.T  # 20, t*nl

        # projecting input of adjacent noise level to the change of variable mode
        ff_ = rearrange(ff[:, [nl, nl+1], :], 'n l k -> (n l) k')  # n*nl, Ne
        proj_input = mode_s_prime[:f.N_e, :].T @ ff_.T  # 20, t*nl

        proj_output_all.append(proj_output)
        proj_input_all.append(proj_input)
        
    return proj_output_all, proj_input_all


def analyze_eigenmodes_trained_v3(epoch, img, num_modes=None):
    '''
    Better derivative measures 10%-0% at 0%, 30%-10% at 10%, 50%-30% at 30%. 
    '''
    ff = np.load(ff_dir) 
    ff = rearrange(ff, 'n l p k h w -> (n l p) (k h w)')  # n_img*nl*np, Ne

    u_path = get_my_epoch(u_trained_dir, epoch)
    u = np.load(u_path)
    u = seqUnmix(u, t_ind, n_ind, n_imgs=num_imgs, npa=10)  # n_img*nl*np, 16, Ne+Ni
    r = f.r_numpy(u)  # n_img*nl*np, 76, Ne+Ni
    r_eq = r[:, -2:].mean(-2)  # n_img*nl*np, Ne+Ni

    r_eq = rearrange(r_eq, '(n l p) k -> n l p k', n=num_imgs, l=4)
    ff = rearrange(ff, '(n l p) k -> n l p k', n=num_imgs, l=4)
    # take the corresponding noise pattern where linearization performed
    r_eq = r_eq[:, :, 0]  # n, l, Ne+Ni
    ff = ff[:, :, 0]  # n, l, Ne

    proj_output_all, proj_input_all = [], []
    # image specific eigenspectra
    for nl in [0, 1, 2]:
        Jac_path = get_my_epoch_img_nl(Jac_trained_dir, img, postfix[nl], epoch=epoch)
        eigd = joblib.load(Jac_path)
        # eigspec = eigd['val']
        vl = eigd['vl']#[:f.N_e]
        vl_prime = eigd['vl_prime']#[:f.N_e]

        # get effective dimension
        vl_prime_norm = np.linalg.norm(vl_prime, axis=0)
        eff_dim = np.where(vl_prime_norm == 0)[0][0]
        N = num_modes if num_modes is not None else eff_dim

        # slowest decaying mode
        mode_s = vl[:, :N]  # Ne+Ni, 20
        # mode_s = vl[:, eff_dim: eff_dim+num_modes]
        # mode_s = vl[:, -num_modes:]
        # mode_s = mode_s / norm(mode_s, axis=0, keepdims=True)
        # change of variable modes
        mode_s_prime = vl_prime[:, :N]  # Ne+Ni, 20
        # mode_s_prime = vl_prime[:, eff_dim: eff_dim+num_modes]
        # mode_s_prime = vl_prime[:, -num_modes:]
        # mode_s_prime = mode_s_prime / norm(mode_s_prime, axis=0, keepdims=True)

        # projecting output of adjacent noise level to the mode
        # r_eq_exc = rearrange(r_eq[:, [nl, nl+1], :f.N_e], 'n l k -> (n l) k')  # n*nl, Ne
        r_eq_exc = rearrange(r_eq[:, [nl, nl+1]], 'n l k -> (n l) k')  # n*nl, Ne+Ni
        proj_output = mode_s.T @ r_eq_exc.T  # 20, t*nl

        # projecting input of adjacent noise level to the change of variable mode
        ff_ = rearrange(ff[:, [nl, nl+1], :], 'n l k -> (n l) k')  # n*nl, Ne
        proj_input = mode_s_prime[:f.N_e, :].T @ ff_.T  # 20, t*nl

        proj_output_all.append(proj_output)
        proj_input_all.append(proj_input)
        
    return proj_output_all, proj_input_all


def get_normalized_dist(r_list, img):
    """
    Get the distance (estimate of derivatives). r should be a list of array, each of shape (n_modes, num_img*2), containing projected 
    stationary state of all images in 2 adjacent noise level. The linearization is performed around the first level. Derivative is 
    approximated by the distance between 2 projected stationary states/inputs of the specified image, divided by increment of noise level 
    (10% as a unit). The estimated derivative is normalized by the overall distance scale of the projected space, which is the 
    variance of projected stationary states/inputs of all images at the linearized noise level. 
    """
    delta_list = [1, 2, 2]
    assert len(r_list) == 3
    ds, scales, d_norms = [], [], []
    for r, delta in zip(r_list, delta_list):
        r = rearrange(r, 'k (n l) -> n l k', l=2)
        r_img = r[img]  # l, k
        d = norm((r_img[0] - r_img[1]) / delta)
        # dn = d / norm(r_img[0])
        ds.append(d)

        r_scale = r[:, 0]  # n, k
        # r_scale_mean = r_scale.mean(0, keepdims=True)
        # cov_scale = (r_scale - r_scale_mean).T @ (r_scale - r_scale_mean) / num_imgs
        # scale = np.sqrt(np.trace(cov_scale))
        r_scale_ref = r_scale[[img]]
        cov_scale = (r_scale - r_scale_ref).T @ (r_scale - r_scale_ref) / (num_imgs - 1)
        scale = np.sqrt(np.trace(cov_scale))
        # scalen = scale / norm(r_scale_ref)
        scales.append(scale)

        d_norm = d / scale
        d_norms.append(d_norm)
    
    return np.stack(ds), np.stack(scales), np.stack(d_norms)


if __name__ == '__main__':
    # load input TODO: should change ----------------------------------------------------------------------------------
    ff = np.load(ff_dir)
    ff = rearrange(ff, 'n l p k h w -> (n l p) (k h w)')  # n_img*nl*np, Ne
    # ff = np.load(ff_dir) # trial, n_img*nl*np, 16, Ne
    # ff = ff[:, :, -1] # temporally-uncorrelated, trial, n_img*nl*np, Ne
    # ff = rearrange(ff, 't n k -> (t n) k')  # trial*n_img*nl*np, Ne
    # ------------------------------------------------------------------------------------------------------------------
    # pca of inputs
    # nc_ff = modes_ns[0] if modes_ns[0] is not None else 200
    nc_ff = 200
    pca = PCA(n_components=nc_ff)
    ff_reduced = pca.fit_transform(ff)  # t*n_img*nl*np, k
    # ff_reduced = ff

    # calculate all distance
    dist_o = np.zeros((modes_ns[0]//4, 5, 3, num_imgs))
    var_o = np.zeros((modes_ns[0]//4, 5, 3, num_imgs))
    ddc_o = np.zeros((modes_ns[0]//4, 5, 3, num_imgs))
    dist_i = np.zeros((modes_ns[0]//4, 5, 3, num_imgs))
    var_i = np.zeros((modes_ns[0]//4, 5, 3, num_imgs))
    ddc_i = np.zeros((modes_ns[0]//4, 5, 3, num_imgs))
    dist_pre_o = np.zeros((modes_ns[0]//4, 3, num_imgs))
    var_pre_o = np.zeros((modes_ns[0]//4, 3, num_imgs))
    ddc_pre_o = np.zeros((modes_ns[0]//4, 3, num_imgs))
    dist_pre_i = np.zeros((modes_ns[0]//4, 3, num_imgs))
    var_pre_i = np.zeros((modes_ns[0]//4, 3, num_imgs))
    ddc_pre_i = np.zeros((modes_ns[0]//4, 3, num_imgs))
    dist_ff = np.zeros((3, num_imgs))
    var_ff = np.zeros((3, num_imgs))
    ddc_ff = np.zeros((3, num_imgs))

    for n, nm in zip(range(num_imgs), modes_ns):
        # alpha-space TODO: should change ----------------------------------------------------------------------------------
        ff_reduced_ = rearrange(ff_reduced, '(n l p) k -> n l p k', n=num_imgs, l=4, p=10)
        v_ff = [rearrange(ff_reduced_[:, [i, i+1], 0], 'n l k -> k (n l)') for i in [0, 1, 2]]  # pc, n*l
        d_ff_pi, scale_ff_pi, d_norm_ff_pi = get_normalized_dist(v_ff, n)
        # ff_reduced_ = rearrange(ff_reduced, '(t n l p) k -> t n l p k', n=num_imgs, l=4, p=10)
        # v_ff = [rearrange(ff_reduced_[:, n, [i, i+1], 0], 't l k -> k (t l)') for i in [0, 1, 2]]  # pc, t*l
        # d_ff_pi, scale_ff_pi, d_norm_ff_pi = get_SNRs(v_ff)
        # ------------------------------------------------------------------------------------------------------------------
        
        dist_ff[:, n] = d_ff_pi
        var_ff[:, n] = scale_ff_pi
        ddc_ff[:, n] = d_norm_ff_pi

        for nm_ in range(0, nm, 4):
            # default network TODO: should change -------------------------------------------------------------------------------
            proj_output_pre, proj_input_pre = analyze_eigenmodes_pre_v3(img=n, num_modes=nm_+1)
            d_o_pre_pi, scale_o_pre_pi, d_norm_o_pre_pi = get_normalized_dist(proj_output_pre, n)
            d_i_pre_pi, scale_i_pre_pi, d_norm_i_pre_pi = get_normalized_dist(proj_input_pre, n)
            # proj_output_pre, proj_input_pre = analyze_eigenmodes_pre_v3_noise(img=n, num_modes=nm_+1)
            # d_o_pre_pi, scale_o_pre_pi, d_norm_o_pre_pi = get_SNRs(proj_output_pre)
            # d_i_pre_pi, scale_i_pre_pi, d_norm_i_pre_pi = get_SNRs(proj_input_pre)
            # ------------------------------------------------------------------------------------------------------------------
        
            dist_pre_o[nm_//4, :, n] = d_o_pre_pi
            var_pre_o[nm_//4, :, n] = scale_o_pre_pi
            ddc_pre_o[nm_//4, :, n] = d_norm_o_pre_pi
            dist_pre_i[nm_//4, :, n] = d_i_pre_pi
            var_pre_i[nm_//4, :, n] = scale_i_pre_pi
            ddc_pre_i[nm_//4, :, n] = d_norm_i_pre_pi

        # trained network
        dist_o_pi, var_o_pi, ddc_o_pi = [], [], []
        dist_i_pi, var_i_pi, ddc_i_pi = [], [], []
        for e in [0, 1, 2, 3, 4]:
            for nm_ in range(0, nm, 4):
                # TODO: should change --------------------------------------------------------------------------------------
                proj_output, proj_input = analyze_eigenmodes_trained_v3(epoch=e, img=n, num_modes=nm_+1)
                d_o_pipe, scale_o_pipe, d_norm_o_pipe = get_normalized_dist(proj_output, n)
                d_i_pipe, scale_i_pipe, d_norm_i_pipe = get_normalized_dist(proj_input, n)
                # proj_output, proj_input = analyze_eigenmodes_trained_v3_noise(epoch=e, img=n, num_modes=nm_+1)
                # d_o_pipe, scale_o_pipe, d_norm_o_pipe = get_SNRs(proj_output)
                # d_i_pipe, scale_i_pipe, d_norm_i_pipe = get_SNRs(proj_input)
                # ------------------------------------------------------------------------------------------------------------

                dist_o[nm_//4, e, :, n] = d_o_pipe
                var_o[nm_//4, e, :, n] = scale_o_pipe
                ddc_o[nm_//4, e, :, n] = d_norm_o_pipe
                dist_i[nm_//4, e, :, n] = d_i_pipe
                var_i[nm_//4, e, :, n] = scale_i_pipe
                ddc_i[nm_//4, e, :, n] = d_norm_i_pipe

        print(f'[INFO] img={n} done.')

    np.save(os.path.join(out_dir, 'dist_o.npy'), dist_o)  # nm, day, nl, img
    np.save(os.path.join(out_dir, 'var_o.npy'), var_o)  # nm, day, nl, img
    np.save(os.path.join(out_dir, 'ddc_o.npy'), ddc_o)  # nm, day, nl, img
    np.save(os.path.join(out_dir, 'dist_pre_o.npy'), dist_pre_o)  # nm, nl, img
    np.save(os.path.join(out_dir, 'var_pre_o.npy'), var_pre_o) # nm, nl, img
    np.save(os.path.join(out_dir, 'ddc_pre_o.npy'), ddc_pre_o) # nm, nl, img
    np.save(os.path.join(out_dir, 'dist_i.npy'), dist_i)  # nm, day, nl, img
    np.save(os.path.join(out_dir, 'var_i.npy'), var_i)  # nm, day, nl, img
    np.save(os.path.join(out_dir, 'ddc_i.npy'), ddc_i)  # nm, day, nl, img
    np.save(os.path.join(out_dir, 'dist_pre_i.npy'), dist_pre_i)  # nm, nl, img
    np.save(os.path.join(out_dir, 'var_pre_i.npy'), var_pre_i) # nm, nl, img
    np.save(os.path.join(out_dir, 'ddc_pre_i.npy'), ddc_pre_i) # nm, nl, img
    np.save(os.path.join(out_dir, 'dist_ff.npy'), dist_ff)  # nl, img
    np.save(os.path.join(out_dir, 'var_ff.npy'), var_ff)  # nl, img
    np.save(os.path.join(out_dir, 'ddc_ff.npy'), ddc_ff)  # nl, img
