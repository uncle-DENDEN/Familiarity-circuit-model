from utils import *
from Model_gpu import ProtoRNN
import os.path as osp
import numpy as np
import torch
import argparse
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False


def parse_args():
    parser = argparse.ArgumentParser("Training ProtoRNN")
    # image param
    parser.add_argument("--num_kernel", type=int, default=64, help="number of kernels")
    parser.add_argument("--num_row", type=int, default=5, help="number of rows in 2D rectilinear grid")
    parser.add_argument("--num_col", type=int, default=5, help="number of columns in 2D rectilinear grid")
    parser.add_argument("--mix_coef", type=int, default=2, help="coefficient for mixing teaching signal and noise")
    parser.add_argument("--num_video", type=int, default=20, help='number of video stimuli')
    # simulation param
    parser.add_argument("--tau_stim", type=int, default=150, help='presentation period for one stimulus')
    parser.add_argument("--delta_t", type=int, default=1, help="step size for the newton method")
    parser.add_argument("--gamma", type=int, default=1, help="feedfoward scaling, -> saliency/attentional signal")
    parser.add_argument("--save_every_stim", type=int, default=150, 
                        help="how much time step to save during the stimulus presentation period. If set to 1, then always save the last")
    # testing param
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--set_wrp", type=parse_boolean, default=True)
    parser.add_argument("--wrp_path", type=str, default='Circuits_noise/weight_epoch1_tx25_100ms_64_8_8_mix50%.npy')
    parser.add_argument("--r_in_path", type=str, default='cifarNoiseContinuumTrain_abs_20_10_10_64_8_8.npy')
    parser.add_argument("--out_path", type=str, default='SMT-BCM_response_correct')
    parser.add_argument("--postfix", type=str, default='mix50%', help='name postfix')

    return parser.parse_args()

args = parse_args()

# create dir
Path(args.out_path).mkdir(parents=True, exist_ok=True)

# init model
f = ProtoRNN(args.num_kernel, args.num_row, args.num_col, args.tau_stim, args.delta_t, args.save_every_stim)

# set wrp and theta [Optional]
if args.set_wrp:
    wrp = np.load(args.wrp_path)
    f.wrp = wrp

# move to gpu
f.to(device)

if __name__ == '__main__':
    # get stimuli for running
    r_in = np.load(args.r_in_path)
    # for noise task
    # img_num, nl, npt = r_in.shape[0], r_in.shape[1], r_in.shape[2]
    # r_in, _, _ = insertTeachingSig(r_in, p=args.mix_coef)
    # r_in = torch.tensor(r_in).reshape(-1, 1, f.num_kernel, f.num_row, f.num_col).float().to(device)
    # for all image task
    img_num = r_in.shape[0]
    r_in = torch.tensor(r_in).reshape(-1, 1, f.num_kernel, f.num_row, f.num_col).float().to(device)
    # for video task
    # r_in = torch.tensor(r_in).reshape(args.num_video, -1, f.num_kernel, f.num_row, f.num_col).float().to(device)

    # set simulation param
    f.noise_test = args.noise

    # run training
    ys_test_all = []
    ff_all = []
    # for i in trange(n_trials, desc='trials...', position=0):
    for i in range(args.n_trials):
        ys_test, ff = f.test(r_in, 0)
        ys_test_all.append(ys_test)
        if f.noise_test > 0:
            ff_all.append(ff)

    # save
    ys_test_all = np.stack(ys_test_all).squeeze()
    mt = 'mean' if args.n_trials == 1 else f'{args.n_trials}trial'
    
    # save noise experiments 
    # if args.set_wrp:
    #     train_info = args.wrp_path.split('/')[-1].split('_')[1:3]
    #     ti = f'{train_info[0]}_{train_info[1]}'
    # else:
    #     ti = 'pre'
    # wname = f'r_{ti}_rx{args.gamma}_{args.tau_stim}ms_{mt}_{img_num}_{nl}x{npt}_{args.save_every_stim}_{args.num_kernel}_{args.num_row}_{args.num_col}_{args.postfix}.npy'
    # ffname = f'ff_{ti}_rx{args.gamma}_{args.tau_stim}ms_{mt}_{img_num}_{nl}x{npt}_{args.save_every_stim}_{args.num_kernel}_{args.num_row}_{args.num_col}_{args.postfix}.npy'

    # save familiarity experimehts
    if args.set_wrp:
        train_info = args.wrp_path.split('/')[-1].split('_')[1:3]
        ti = f'{train_info[0]}_{train_info[1]}'
    else:
        ti = 'pre'
    wname = f'r_{ti}_rx{args.gamma}_{args.tau_stim}ms_{mt}_{img_num}_{args.save_every_stim}_{args.num_kernel}_{args.num_row}_{args.num_col}_{args.postfix}.npy'
    
    # save video experimehts
    # vid_num = r_in.shape[0]
    # if args.set_wrp:
    #     train_info = args.wrp_path.split('/')[-1].split('_')[1:3]
    #     ti = f'{train_info[0]}_{train_info[1]}'
    # else:
    #     ti = 'pre'
    # wname = f'r_{ti}_rx{args.gamma}_{args.tau_stim}ms_{mt}_{vid_num}_{args.save_every_stim}_{args.num_kernel}_{args.num_row}_{args.num_col}_{args.postfix}.npy'
    
    ys_test_all = np.stack(ys_test_all).squeeze()
    np.save(osp.join(args.out_path, wname), ys_test_all)

    # if len(ff_all) > 0:
    #     ff_all = np.stack(ff_all).squeeze()
    #     np.save(osp.join(args.out_path, ffname), ff_all)
