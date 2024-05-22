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
import argparse
import joblib
import os

import torch
from einops import rearrange, repeat


def parse_args():
    parser = argparse.ArgumentParser("Training ProtoRNN")
    # image param
    parser.add_argument("--input_path", type=str, default='SMT-BCM_input_correct/r_in_cifar_all_noise_abs_5_4_10_64_8_8.npy')
    parser.add_argument("--num_imgs", type=int, default=5)
    parser.add_argument("--num_patterns", type=int, default=10)

    parser.add_argument("--weight_path", type=str, default='')
    parser.add_argument("--response_path", type=str, default='')
    parser.add_argument("--out_path", type=str, default='')

    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument("--img_list", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--level_list", type=int, nargs="+", default=[0, 1, 2])

    return parser.parse_args()


def get_my_epoch(input_string, e):
    epoch_pos = input_string.find('epoch')
    end_pos = epoch_pos + 5  # Start from the character after 'epoch'
    
    while end_pos < len(input_string) and input_string[end_pos].isdigit():
        end_pos += 1

    # Replace the number after 'epoch' with e
    new_string = input_string[:epoch_pos + 5] + str(e) + input_string[end_pos:]

    return new_string


args = parse_args()

postfix = {0: 'clear', 1: '10%', 2: '30%', 3: '50%'}
num_imgs = args.num_imgs
num_patterns = args.num_patterns
device = torch.device('cpu')

# get input and mixing index
mix_coef = 2
r_in = np.load(args.input_path)
r_in, t_ind, n_ind = insertTeachingSig(r_in, p=mix_coef)

# get model
num_kernel = 64
num_row = 8
num_col = 8
tau_stim = 150
delta_t = 1 
save_every_stim = 15

f = ProtoRNN(num_kernel, num_row, num_col, tau_stim, delta_t, save_every_stim)
f.to(device)

# static weights
wei = f._wei0.cpu().numpy()
wie = f._wie0.cpu().numpy()

# get tau_inverse
tau_inv_flat = np.concatenate([np.array([1/f.tau_ue]*f.N_e), np.array([1/f.tau_ui]*f.N_i)])  # Ne+Ni
tau_inv = np.diag(tau_inv_flat)

for i in range(0, args.epoch_num):
    # get weight matrix TODO: change for pre ----------------------------------------------------------------------------
    # weight_path_e = get_my_epoch(args.weight_path, i)
    # wee = np.load(weight_path_e)
    wee = f.wrp0.cpu().numpy()
    # -------------------------------------------------------------------------------------------------------------------
    w = np.block([
        [wee, -wei],
        [wie, np.zeros_like(wee)]
    ])

    # got equilibrium point TODO: change for pre ----------------------------------------------------------------------------
    # response_path_e = get_my_epoch(args.response_path, i)
    response_path_e = args.response_path
    u = np.load(response_path_e)
    # -------------------------------------------------------------------------------------------------------------------
    u = seqUnmix(u, t_ind, n_ind, n_imgs=num_imgs, npa=num_patterns)  # n_img*nl*np, 16, Ne+Ni
    u_eq = u[:, -2:].mean(1)  # n_img*nl*np, Ne+Ni
    # r = f.r_numpy(u)
    # r_eq = r[:, -2:].mean(1)  # n_img*nl*np, Ne+Ni

    # get phi_prime
    dphi_dreq = np.maximum(0, 2*u_eq)  # n_img*nl*np, Ne+Ni
    phi_prime_flat = dphi_dreq * np.expand_dims(tau_inv_flat, 0)  # n_img*nl*np, Ne+Ni
    phi_prime_flat = rearrange(phi_prime_flat, '(n l p) a -> n l p a', n=num_imgs, p=num_patterns)  # n_img, nl, np, Ne+Ni
    del dphi_dreq

    # get Jacobian and eigenspectra for sampled attractors TODO: change for pre ---------------------------------------------
    # out_path_e = get_my_epoch(args.out_path, i)
    out_path_e = args.out_path
    os.makedirs(out_path_e, exist_ok=True)
    # -------------------------------------------------------------------------------------------------------------------
    for n in args.img_list:
        for m in args.level_list:
            # get corresponding sensitivity matrix
            phi_prime_mn = np.diag(phi_prime_flat[n, m, 0])  # Ne+Ni, Ne+Ni
            
            # get Jacobian for the single img
            jmn = phi_prime_mn @ w - tau_inv
            # TODO: change for pre -----------------------------------------------------------
            # print(f'[INFO] Get Jacobians for epoch{i}, image{n}, {postfix[m]}')
            print(f'[INFO] Get Jacobians for pre, image{n}, {postfix[m]}')
            # ---------------------------------------------------------------------------------

            # eigendecomposition
            vals, vl, vr = linalg.eig(jmn, left=True)

            vals = np.real(vals)
            vl = np.real(vl)
            vr = np.real(vr)
            vl_prime = phi_prime_mn @ vl
            
            # sort 
            sortind = np.argsort(vals)[::-1]
            vals_sorted = vals[sortind]
            vl_sorted = vl[:, sortind]
            vr_sorted = vr[:, sortind]
            vl_prime_sorted = vl_prime[:, sortind]
            
            data = {'val': vals_sorted, 'vl': vl_sorted, 'vr': vr_sorted, 'vl_prime': vl_prime_sorted}
            joblib.dump(data, os.path.join(out_path_e, f"edc_jac_img{n}_{postfix[m]}_sample.pkl"))
            
            del jmn, vals, vl, vr, vl_prime, vals_sorted, vl_sorted, vr_sorted, vl_prime_sorted
            print(f'\tEigenvalue saved for the Jacobian')
    
    del phi_prime_flat, w
