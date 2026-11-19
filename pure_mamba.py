import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from aeon.datasets import load_classification
from mamba_backbone import MambaBackbone

class PureMambaClassifier(nn.Module):
    def __init__(self, n_classes, d_model=64, n_layers=4):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.backbone = MambaBackbone(d_model=d_model, n_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        #x: (B, T)
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        x = (x - mu) / sd
        h = self.in_proj(x.unsqueeze(-1))
        h = self.backbone(h) #(B, T, d_model)
        pooled = h.mean(dim=1) #(B, d_model)
        return self.head(pooled)

class PureMambaPooledClassifier(nn.Module):
    def __init__(self, n_classes, d_model=64, n_layers=4, pool_size=16):
        super().__init__()
        self.pool_size = pool_size
        self.in_proj = nn.Linear(1, d_model)
        self.backbone = MambaBackbone(d_model=d_model, n_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        x = (x - mu) / sd
        h = self.in_proj(x.unsqueeze(-1)) #(B, T, d_model)
        B, T, D = h.shape
        P = T // self.pool_size
        h = h[:, :P*self.pool_size].reshape(B, P, self.pool_size, D).mean(dim=2)
        h = self.backbone(h)
        return self.head(h.mean(dim=1))

class PureMambaForecaster(nn.Module):
    def __init__(self, horizon, d_model=64, n_layers=4):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.backbone = MambaBackbone(d_model=d_model, n_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        x = (x - mu) / sd
        h = self.in_proj(x.unsqueeze(-1))
        h = self.backbone(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)

class PureMambaPooledForecaster(nn.Module):
    def __init__(self, horizon, d_model=64, n_layers=4, pool_size=16):
        super().__init__()
        self.pool_size = pool_size
        self.in_proj = nn.Linear(1, d_model)
        self.backbone = MambaBackbone(d_model=d_model, n_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        x = (x - mu) / sd
        h = self.in_proj(x.unsqueeze(-1)) #(B, T, d)
        B, T, D = h.shape
        P = T // self.pool_size
        h = h[:, :P * self.pool_size].reshape(B, P, self.pool_size, D).mean(dim=2)
        h = self.backbone(h)
        return self.head(h.mean(dim=1))

#data loading and training helpers
def load_ucr(name, split="train"):
    try:
        result = load_classification(name, split=split)
    except TypeError:
        result = load_classification(name, split=split, meta_data=True)

    if isinstance(result, tuple) and len(result) == 3:
        X, y, _ = result
    else:
        X, y = result

    if X.ndim == 3:
        X = X.squeeze(axis=1)

    classes = np.unique(y)
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[v] for v in y], dtype=np.int64)
    return X.astype(np.float32), y

def make_forecasting_pairs(X, horizon_frac=0.25):
    n, T = X.shape
    H = max(1, int(T*horizon_frac))
    return X[:, :-H], X[:, -H:]

def train_classifier(model, X_train, y_train, X_val, y_val, epochs=100, lr=1e-3, batch_size=64, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    amp_device = "cuda" if device == "cuda" else "cpu"
    scaler = GradScaler(amp_device, enabled=(device == "cuda"))
    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_val_t = torch.tensor(X_val, device=device)
    y_val_t = torch.tensor(y_val, device=device)
    best_val, best_state = -1.0, None

    try:
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(X_train_t), device=device)

            for i in range(0, len(X_train_t), batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                with autocast(device_type=amp_device, enabled=(device == "cuda")):
                    logits = model(X_train_t[idx])
                    loss = loss_fn(logits, y_train_t[idx])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            model.eval()

            with torch.no_grad():
                accuracy = (model(X_val_t).argmax(-1) == y_val_t).float().mean().item()

            if accuracy > best_val:
                best_val = accuracy
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
    finally:
        del X_train_t, y_train_t, X_val_t, y_val_t
        del opt, scaler

    return best_val

def train_forecaster(model, X_train, y_train, X_val, y_val, epochs=100, lr=1e-3, batch_size=64, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    amp_device = "cuda" if device == "cuda" else "cpu"
    scaler = GradScaler(amp_device, enabled=(device == "cuda"))
    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_val_t = torch.tensor(X_val, device=device)
    y_val_t = torch.tensor(y_val, device=device)
    best_val, best_state = float("inf"), None

    try:
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(X_train_t), device=device)

            for i in range(0, len(X_train_t), batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad(set_to_none=True)
                with autocast(device_type=amp_device, enabled=(device == "cuda")):
                    pred = model(X_train_t[idx])
                    loss = loss_fn(pred, y_train_t[idx])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            model.eval()

            with torch.no_grad():
                mse = loss_fn(model(X_val_t), y_val_t).item()

            if mse < best_val:
                best_val = mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
    finally:
        del X_train_t, y_train_t, X_val_t, y_val_t
        del opt, scaler
        
    return best_val

#experiments
def pure_classification(name, seed, epochs=100, device="cpu"):
    X, y = load_ucr(name)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)
    n_classes = len(np.unique(y))
    model = PureMambaClassifier(n_classes=n_classes, d_model=64, n_layers=4)
    try:
        train_classifier(model, X_train, y_train, X_val, y_val, epochs=epochs, device=device)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test, device=device))
            accuracy = (logits.argmax(-1).cpu().numpy() == y_test).mean()
        result = float(accuracy)
    finally:
        del model
        torch.cuda.empty_cache()

    return result

def pure_forecasting(name, seed, epochs=100, device="cpu"):
    #seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X, _ = load_ucr(name)
    X_in, y_out = make_forecasting_pairs(X)
    n = len(X)
    train_end = int(0.7*n)
    val_end = int(0.85*n)

    X_train, y_train = X_in[:train_end], y_out[:train_end]
    X_val, y_val = X_in[train_end:val_end], y_out[train_end:val_end]
    X_test, y_test = X_in[val_end:], y_out[val_end:]

    model = PureMambaForecaster(horizon=y_train.shape[1], d_model=64, n_layers=4)
    try:
        train_forecaster(model, X_train, y_train, X_val, y_val, epochs=epochs, device=device)
        model.eval()

        with torch.no_grad():
            pred = model(torch.tensor(X_test, device=device)).cpu().numpy()
        result = float(np.mean((pred-y_test)**2))
    finally:
        del model
        torch.cuda.empty_cache()

    return result

def pooled_classification(name, seed, epochs=100, pool_size=16, device="cpu"):
    X, y = load_ucr(name)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)
    n_classes = len(np.unique(y))

    model = PureMambaPooledClassifier(n_classes=n_classes, d_model=64, n_layers=4, pool_size=pool_size)
    try:
        train_classifier(model, X_train, y_train, X_val, y_val, epochs=epochs, device=device)
        model.eval()

        with torch.no_grad():
            logits = model(torch.tensor(X_test, device=device))
            accuracy = (logits.argmax(-1).cpu().numpy() == y_test).mean()
        result = float(accuracy)
    finally:
        del model
        torch.cuda.empty_cache()

    return result

def pooled_forecasting(name, seed, epochs=100, pool_size=16, device="cpu"):
    #seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, _ = load_ucr(name)
    X_in, y_out = make_forecasting_pairs(X)
    n = len(X)
    train_end = int(0.7*n)
    val_end = int(0.85*n)

    X_train, y_train = X_in[:train_end], y_out[:train_end]
    X_val, y_val = X_in[train_end:val_end], y_out[train_end:val_end]
    X_test, y_test = X_in[val_end:], y_out[val_end:]

    model = PureMambaPooledForecaster(horizon=y_train.shape[1], d_model=64, n_layers=4, pool_size=pool_size)
    try:
        train_forecaster(model, X_train, y_train, X_val, y_val, epochs=epochs, device=device)
        model.eval()

        with torch.no_grad():
            pred = model(torch.tensor(X_test, device=device)).cpu().numpy()
        result = float(np.mean((pred-y_test)**2))
    finally:
        del model
        torch.cuda.empty_cache()

    return result

#main
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classification_datasets = ["ECG200", "FordA", "UWaveGestureLibraryAll", "Yoga", "GunPoint",
            "CBF", "Lightning2", "Wafer", "ElectricDevices", "EthanolLevel",
            "Plane", "Strawberry", "ScreenType", "SonyAIBORobotSurface1"]
    forecasting_datasets = ["ECG200", "FordA", "Wafer", "ElectricDevices"]

    seeds = 5
    epochs = 100

    print("Mamba Baseline (Full Resolution)")

    t0 = time.time()
    print("Classification:")

    for name in classification_datasets:
        accuracies = []
        for s in range(seeds):
            holder = pure_classification(name, s, epochs=epochs, device=device)
            accuracies.append(holder)
        print(f"-{name:<28} Accuracy = {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")

    print("\nForecasting")
    for name in forecasting_datasets:
        mses = []
        for s in range(seeds):
            mses.append(pure_forecasting(name, s, epochs=epochs, device=device))
        print(f"-{name:<28} MSE = {np.mean(mses):.6f} +/- {np.std(mses):.6f}")

    print(f"\nDone in {time.time()-t0:.1f}s")

    print("\nMamba Baseline (pooled)")

    t1 = time.time()
    print("Classification:")

    for name in classification_datasets:
        accuracies = []
        for s in range(seeds):
            accuracies.append(pooled_classification(name, s, epochs=epochs, pool_size=16, device=device))
        print(f"-{name:<28} Accuracy = {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")

    print("\nForecasting")
    for name in forecasting_datasets:
        mses = []
        for s in range(seeds):
            mses.append(pooled_forecasting(name, s, epochs=epochs, device=device))
        print(f"-{name:<28} MSE={np.mean(mses):.6f} +/- {np.std(mses):6f}")

    print(f"\nDone in {time.time()-t1:.1f}s")