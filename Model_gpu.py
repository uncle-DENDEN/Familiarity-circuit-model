from utils import *
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Poisson
import copy

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# _device = torch.device("cpu")

class ProtoRNN(nn.Module):
    def __init__(self, n_kernel, n_row, n_col, tau_stim, delta_t, save_every_stim) -> None:
        super().__init__()
        
        self.num_kernel = n_kernel
        self.num_row = n_row
        self.num_col = n_col

        # simulation parameter
        self.tau_stim = tau_stim
        self.delta_t = delta_t # simulation timestep = 1 ms
        self.save_every_stim = save_every_stim

        # time constant of synaptic input change
        self.tau_ue = 20
        self.tau_ui = 10

        # number of inhibitory neurons
        # N_i = int(N_e / (len(spat_freq) * len(sigma_deno)))
        self.N_e = n_kernel * n_row * n_col
        self.N_i = self.N_e
        # new variable
        self.radius_spatial = 2
        self.radius_inhibition = 1

        self.N_neighbors = n_kernel * (2 * self.radius_spatial + 1) * (2 * self.radius_spatial + 1)
        self.N_inh_neighbors = (2 * self.radius_inhibition + 1) * (2 * self.radius_inhibition + 1) + n_kernel - 1

        # Parameters regarding the weight
        # wrp_max = 50.0 / N_neighbors
        # in version 18:
        self.wrp_max = 25.0 / self.N_neighbors
        self.wei = 1.0 / self.N_i
        #wie = 30.0 / ((2 * radius_inhibition + 1) * (2 * radius_inhibition + 1) * len(spat_freq) * len(sigma_deno))
        self.wie = 30.0 / self.N_inh_neighbors 
        self.wee = 5.0 / self.N_neighbors 

        self.tau_wrp = 2e9  # TODO: scale can be reverted by trimming learning time
        self.tau_theta = 1e7

        # scalars in the dynamical equation
        self.gamma = 1.0
        self.noise_train = 0
        self.noise_test = 1.0

        # threshold for firing
        self.beta = 1.0
        # threshold = np.mean(r_in) * wee * N_neighbors * gamma
        # a very sensitive variable
        self.threshold = 0.0

        # new variable
        self.theta_cap = 40.0

        # in version 14:
        # theta_noise_sd = 10.0
        # in version 15:
        # in version 22:
        # theta_BCM = np.load("./threshold_BCM_10000_multi60_time100.npy")*0.5
        # theta_BCM = theta_BCM.reshape(300)
        # only in this version (13):
        #g1 = np.random.choice(N_e, size = N_e // 3, replace = False)
        #g2 = np.random.choice(np.setdiff1d(np.arange(N_e), g1), size = N_e // 3, replace = False)
        #g3 = np.setdiff1d(np.setdiff1d(np.arange(N_e), g1), g2)
        # self.theta_noise = np.zeros(self.N_e)
        #mu, sigma = 0, 0.1 # mean and standard deviation
        #theta_noise = np.random.normal(mu, sigma, Ne)
        #theta_noise = np.random.normal(mu, sigma, N_e)
        #theta_noise[g1] = theta_BCM[g1] / theta_noise_ratio
        #theta_noise[g2] = 0
        #theta_noise[g3] = -theta_BCM[g3] / theta_noise_ratio

        # theta_noise = np.random.normal(loc=0.0, scale = theta_noise_sd, size=N_e)
        #theta_BCM = theta_BCM + theta_noise

        self.weight_constant = self.wee * self.N_neighbors

        # init weight
        self._init_params()

    @ property
    def wrp0(self):
        return self._wrp0
    
    @ property
    def wrp(self):
        return self._wrp
    
    @ wrp.setter
    def wrp(self, wrp):
        self._wrp = nn.Parameter(torch.tensor(wrp), requires_grad=False)
    
    @ property
    def theta(self):
        theta = getattr(self, '_theta', None)
        if theta is None:
            raise AttributeError('Initial theta is not given')
        else:
            return theta
    
    @ theta.setter
    def theta(self, theta):
        self._theta = nn.Parameter(torch.tensor(theta), requires_grad=False)

    def _init_params(self):
        # Initializing the weights
        wrp = torch.zeros(self.N_e, self.N_e)
        mask = torch.zeros(self.N_e, self.N_e)
        for i in range(self.N_e):
            wrp[i, choosing_neighbors(i, self.num_kernel, self.num_row, self.num_col, self.radius_spatial)] = self.wee
            mask[i, choosing_neighbors(i, self.num_kernel, self.num_row, self.num_col, self.radius_spatial)] = 1
        wrp = (wrp.T / wrp.sum(1) * self.weight_constant).T  # transpose -> output sum fixed 
        
        self._wrp0 = copy.deepcopy(wrp)
        self._wrp = nn.Parameter(wrp, requires_grad=False)
        self._mask = nn.Parameter(mask, requires_grad=False)

        self._wei0 = nn.Parameter(self.wei * torch.ones(self.N_e, self.N_i), requires_grad=False)
        wie0 = self.wie * torch.zeros(self.N_i, self.N_e)
        # num_kernel * num_row * num_col
        for i in range(self.N_i):
            wie0[i, choosing_inhibition_neighbors(i, self.num_kernel, self.num_row, self.num_col, self.radius_spatial)] = self.wie
        self._wie0 = nn.Parameter(wie0, requires_grad=False)

        self._ue0 = nn.Parameter(torch.zeros(self.N_e))
        self._ui0 = nn.Parameter(torch.zeros(self.N_i))

    def _config_simulation(self, n_stimulus, n_item_per_stim):
        save_interval = int(self.tau_stim / self.save_every_stim) # save simulation state every save_interval steps.
        T = self.tau_stim * n_stimulus  # simulation time
        # get simulation time point
        if T % self.delta_t != 0:
            raise ValueError('tau_stim cannot be divided by delta_t')
        n = int(T / self.delta_t)
        ts = np.linspace(0.0, T, num=n, endpoint=False, dtype=np.int64)
        # get num repeat
        if self.tau_stim % (self.delta_t*n_item_per_stim) != 0:
            raise ValueError('tau_stim cannot be divided by n_item_per_stim')
        N_repeats = int(round(self.tau_stim / (self.delta_t*n_item_per_stim)))
        decay = 0.1 ** (1/n_stimulus)
        # save_ts = np.linspace(0, self.tau_stim, num=int(n/save_interval), dtype=np.int64)
        return save_interval, ts, N_repeats, decay
    
    def r(self, ui):
        return torch.pow(f.relu(ui - self.threshold), 2) * self.beta
        # return f.relu(ui - self.threshold) * self.beta
    
    def r_numpy(self, ui):
        return np.power(np.maximum(ui - self.threshold, 0), 2) * self.beta

    @torch.no_grad()
    def _fx(self, t, ue, ui, r_in, decay):
        re = self.r(ue)
        # re = re + self.noise_train * np.random.normal(0, np.sqrt(re))
        ri = self.r(ui)

        # dynamic BCM threshold
        # if (t % int(self.tau_stim) == 0):
        #     self._theta = t/self.tau_stim / (t/self.tau_stim+1) * self._theta + (self.gamma * r_in + self.threshold) ** 2.0 / (t/self.tau_stim+1)
        #     self._theta = np.minimum(self._theta, self.theta_cap * np.ones(self.N_e)) + self.theta_noise
        
        # for every stimulus trained, total pre-synaptic weight = constant
        if t % int(self.tau_stim) == 0:
            self._wrp = nn.Parameter((self._wrp.T / self._wrp.sum(1) * self.weight_constant).T, requires_grad=False)  # constraint impl by normalizing

        Iffue_mean = self.gamma * r_in
        Iffue = Iffue_mean + torch.sqrt(self.noise_train * Iffue_mean / self.delta_t) * torch.randn(self.N_e, device=_device)
        # Iffue = Poisson(self.noise_train * Iffue_mean).sample() if self.noise_train > 0 else Iffue_mean
        # it_noise = f.relu(torch.sqrt(self.noise_train * Iffue_mean / self.delta_t) * torch.rand(self.N_e, device=_device))
        duedt = 1.0 / self.tau_ue * (-ue + self._wrp @ re - self._wei0 @ ri + Iffue)  # + it_noise
        duidt = 1.0 / self.tau_ui * (-ui + self._wie0 @ re)
        # in version 16: add a relu here. TODO: should be removed theoretically
        ue = f.relu(ue + duedt)
        ui = f.relu(ui + duidt)

        # BCM rule
        theta_cov = torch.outer((re * (re - self._theta) * (1 / self._theta)), re)
        dwrpdt = 1.0 / self.tau_wrp * theta_cov * self._mask * (decay ** (t // int(self.tau_stim)))  # what is the last term?
        self._wrp += self.delta_t * dwrpdt

        # clip the weight to valid range between 0 and wrp_max
        self._wrp.clip_(0, self.wrp_max)

        # sliding threshold
        dthetadt = 1.0 / self.tau_theta * (-self._theta + re ** 2)
        self._theta += self.delta_t * dthetadt
        
        # print(theta)
        return ue, ui
    
    def train(self, e, r_in, stim):
        n_stimulus, n_item_per_stim = r_in.shape[0], r_in.shape[1]
        skip, ts, N_repeats, decay = self._config_simulation(n_stimulus, n_item_per_stim)
        ys = []
        
        # set theta bcm
        if getattr(self, '_theta', None) is None:
            raise RuntimeError('theta is not set')
        
        # repeat r_in
        r_in = torch.repeat_interleave(r_in.reshape(n_stimulus*n_item_per_stim, self.N_e), N_repeats, 0)
        
        th = torch.zeros(len(ts), self.N_e)
        above_th = torch.zeros(len(ts), self.N_e)
        for i, t in enumerate(tqdm(ts, desc=f'epoch{e} training...')):
            if t % int(self.tau_stim) == 0:
                # clear the synaptic input for a new stimulus (cut the temporal connections deliberately)
                ue = self._ue0
                ui = self._ui0
                
            ue, ui = self._fx(t, ue, ui, r_in[i], decay)
            re = self.r(ue)
            above_th[i] = (re > self._theta).to(torch.int8)
            th[i] = self._theta

            # record the result
            if (t % self.tau_stim == 0) | ((t + 1) % skip == 0):
                ys.append(torch.cat([ue, ui]).cpu().numpy())
        
        resp = np.stack(ys).reshape(n_stimulus, -1, self.N_e+self.N_i)
        return resp, th, above_th
    
    @torch.no_grad()
    def _ft(self, ue, ui, r_in):
        re = self.r(ue)
        # re = re + self.noise_train * np.random.normal(0, np.sqrt(re))
        ri = self.r(ui)
    
        Iffue_mean = self.gamma * r_in
        # temporally-uncorrelated, stimulus-dependent Gaussian noise
        Iffue = Iffue_mean + torch.sqrt(self.noise_test * Iffue_mean / self.delta_t) * torch.randn(self.N_e, device=_device)
        # Iffue = Poisson(self.noise_test * Iffue_mean).sample() if self.noise_test > 0 else Iffue_mean
        # it_noise = f.relu(torch.sqrt(self.noise_test * Iffue_mean / self.delta_t) * torch.rand(self.N_e, device=_device))
        duedt = 1.0 / self.tau_ue * (-ue + self._wrp @ re - self._wei0 @ ri + Iffue)  # + it_noise
        duidt = 1.0 / self.tau_ui * (-ui + self._wie0 @ re)
        # in version 16: add a relu. TODO: should be removed theoretically
        ue = f.relu(ue + duedt)
        ui = f.relu(ui + duidt)
        
        # print(theta)
        return ue, ui, Iffue
    
    def test(self, r_in, stim):
        n_stimulus, n_item_per_stim = r_in.shape[0], r_in.shape[1]
        skip, ts, N_repeats, _ = self._config_simulation(n_stimulus, n_item_per_stim)
        ys = []
        ff = []

        # repeat r_in
        r_in = torch.repeat_interleave(r_in.reshape(n_stimulus*n_item_per_stim, self.N_e), N_repeats, 0)
        
        for i, t in enumerate(tqdm(ts, desc='testing...', position=0, leave=True)):
            # clear the synaptic input for a new stimulus
            if t % int(self.tau_stim) == 0:
                ue = self._ue0
                ui = self._ui0

            ue, ui, iffue = self._ft(ue, ui, r_in[i])
            
            # record the result
            if (t % self.tau_stim == 0) | ((t + 1) % skip == 0):
                ys.append(torch.cat([ue, ui]).cpu().numpy())
                ff.append(iffue.cpu().numpy())
        
        resp = np.stack(ys).reshape(n_stimulus, -1, self.N_e+self.N_i)
        ff = np.stack(ff).reshape(n_stimulus, -1, self.N_e)
        return resp, ff
