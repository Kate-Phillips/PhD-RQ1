import torch
import torch.nn as nn
from mamba_backbone import MambaBackbone
from tokenizers import build_tokenizer

class ClassificationModel(nn.Module):
    def __init__(self, tokenizer_kind, n_classes, d_model=64, n_layers=4, patch_size=16, m=5, tau=1, n_bins=256):
        super().__init__()
        self.tokenizer = build_tokenizer(tokenizer_kind, d_model=d_model, patch_size=patch_size, m=m, tau=tau, n_bins=n_bins) if tokenizer_kind == "delay" or tokenizer_kind == "shuffle" \
        else build_tokenizer(tokenizer_kind, d_model=d_model, patch_size=patch_size, n_bins=n_bins)

        self.backbone = MambaBackbone(d_model=d_model, n_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        if isinstance(self.tokenizer, nn.Module) and self.tokenizer.__class__.__name__=="ShuffleTokenizer":
            emb = self.tokenizer(x)
        else:
            emb = self.tokenizer(x)
        h = self.backbone(emb) #(B, P, d)
        pooled = h.mean(dim=1) #global avg pool
        return self.head(pooled)

class ForecastingModel(nn.Module):
    def __init__(self, tokenizer_kind, horizon, d_model=64, n_layers=4, patch_size=16, m=5, tau=1, n_bins=256):
        super().__init__()
        self.tokenizer = build_tokenizer(tokenizer_kind, d_model=d_model, patch_size=patch_size, m=m, tau=tau, n_bins=n_bins)
        self.backbone = MambaBackbone(d_model=d_model, n_layers=n_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x, rng=None):
        if isinstance(self.tokenizer, nn.Module) and self.tokenizer.__class__.__name__ == "ShuffleTokenizer":
            emb = self.tokenizer(x, rng=rng)
        else:
            emb = self.tokenizer(x)
        h = self.backbone(emb)
        pooled = h.mean(dim=1)
        return self.head(pooled)