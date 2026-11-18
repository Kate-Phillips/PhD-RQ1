import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata
from scipy.spatial import cKDTree

def compute_tau_mi(x, max_lag=50, n_bins=16):
    x = np.asarray(x).ravel()
    n = len(x)
    max_lag = min(max_lag, n // 4)
    if max_lag < 2:
        return 1

    #discretizes into n_bins quantile bins
    edges = np.quantile(x, np.linspace(0, 1, n_bins+1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    symbols = np.digitize(x, edges) - 1

    def mi(lag):
        a = symbols[:n-lag]
        b = symbols[lag:]
        joint = np.zeros((n_bins, n_bins))
        for a_i, b_i in zip(a, b):
            joint[a_i, b_i] += 1
        joint /= joint.sum()
        p_a = joint.sum(axis=1, keepdims=True)
        p_b = joint.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = joint / (p_a * p_b)
            contrib = joint * np.log2(np.where(ratio > 0, ratio, 1.0))
        return -np.nansum(contrib)

    mis = [mi(lag) for lag in range(1, max_lag+1)]

    #first local minimum
    for i in range(1, len(mis) - 1):
        if mis[i] < mis[i-1] and mis[i] < mis[i+1]:
            return i+1

    return 1

def compute_embedding_dim(x, tau, max_dim=20, rtol=10.0, atol=2.0):
    x = np.asarray(x).ravel()
    n = len(x)

    def fnn_fraction(m):
        n_pts = n-m*tau
        if n_pts < 10:
            return 1.0
        emb = np.stack([x[i*tau: i*tau+n_pts] for i in range(m)], axis=1)

        #for each point, find nearest neighbor in m-dim
        tree = cKDTree(emb)
        dists, idxs = tree.query(emb, k=2)
        nn_dist = dists[:, 1]
        nn_idx = idxs[:, 1]

        #add (m+1)th coordinate
        extra = x[(np.arange(n_pts)+m*tau)][:, None] #shape (n_pts, 1)
        extra_nn = extra[nn_idx]

        #FNN criterion
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(extra-extra_nn).ravel() / np.where(nn_dist > 0, nn_dist, 1e-9)
        fnn = np.mean(ratio > rtol)
        return fnn

    for m in range(1, max_dim):
        if fnn_fraction(m) < 0.1: #<10% false neighbors
            return max(m, 2)

    return max_dim

#tokenizer 1: value-space binning (Chronos-style -- eat 'em kids)
class ValueTokenizer(nn.Module):
    def __init__(self, n_bins=256, d_model=64, patch_size=16, max_len=512):
        super().__init__()
        self.n_bins = n_bins
        self.patch_size = patch_size
        self.embed = nn.Embedding(n_bins, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def tokenize(self, x):
        B, T = x.shape

        #mean-scale per sample
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        z = (x-mu) / sd

        #uniform binning over [-5, 5]
        bins = torch.linspace(-5, 5, self.n_bins-1, device=x.device)
        idx = torch.bucketize(z, bins) #(B, T) values in [0, n_bins-1]

        n_patches = T // self.patch_size #take every patch_sizeth token or average-pool bin indicies
        z_patch = z[:, :n_patches*self.patch_size].reshape(B, n_patches, self.patch_size).mean(-1) #patch-avg the z-scores then bin the patch mean
        idx = torch.bucketize(z_patch, bins)
        return idx

    def forward(self, x):
        idx = self.tokenize(x) #(B, P)
        emb = self.embed(idx) #(B, P, d)
        emb = emb + self.pos[:, :emb.size(1), :]
        return emb

#tokenizer 2: delay-embedded (Takens)
class DelayTokenizer(nn.Module):
    def __init__(self, d_model=64, patch_size=16, max_len=512, m=5, tau=1):
        super().__init__()
        self.m = m
        self.tau = tau
        self.patch_size = patch_size

        #input dim per patch = m*patch_size (flattened delay patches)
        self.proj = nn.Linear(m*patch_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def build_delay_vectors(self, x):
        B, T = x.shape
        T_eff = T-(self.m-1)*self.tau
        if T_eff <= 0:
            raise ValueError(f"Series too short for m={self.m}, tau={self.tau}")
        idxs = [torch.arange(i*self.tau, i*self.tau+T_eff, device=x.device) for i in range(self.m)]
        v = torch.stack([x[:, i] for i in idxs], dim=-1) #(B, T_eff, m)
        return v

    def forward(self, x):
        v = self.build_delay_vectors(x) #(B, T_eff, m)
        B, T_eff, m = v.shape
        n_patches = T_eff // self.patch_size
        if n_patches == 0:
            raise ValueError(f"T_eff={T_eff} < patch_size={self.patch_size}")
        v = v[:, :n_patches*self.patch_size] #(B, n_patches*P, m)
        v = v.reshape(B, n_patches, self.patch_size*m) #(B, n_patches, P*m)
        emb = self.proj(v) #(B, n_patches, d)
        emb = emb + self.pos[:, :n_patches, :]
        return emb

#tokenizer 3: (Ablation study) Shuffle control (falsification)
class ShuffleTokenizer(DelayTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #fix permutation at init
        self.register_buffer("perm", torch.randperm(self.m))

    def forward(self, x):
        v = self.build_delay_vectors(x) #(B, T_eff, m)
        B, T_eff, m = v.shape
        n_patches = T_eff // self.patch_size
        v = v[:, :n_patches*self.patch_size]
        v = v.reshape(B, n_patches, self.patch_size, m)
        v = v[..., self.perm] #fixed perm
        v = v.reshape(B, n_patches, self.patch_size*m)
        emb = self.proj(v) + self.pos[:, :n_patches, :]
        return emb

#control panel
def build_tokenizer(kind, **kwargs):
    if kind == "value":
        accepted = {"n_bins", "d_model", "patch_size", "max_len"}
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return ValueTokenizer(**filtered)
    elif kind == "delay":
        accepted = {"d_model", "patch_size", "max_len", "m", "tau"}
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return DelayTokenizer(**filtered)
    elif kind == "shuffle":
        accepted = {"d_model", "patch_size", "max_len", "m", "tau"}
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return ShuffleTokenizer(**filtered)
    else:
        raise ValueError(f"Unknown Tokenizer: {kind}")