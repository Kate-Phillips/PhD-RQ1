import argparse
import csv
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split

from tokenizers import compute_tau_mi, compute_embedding_dim
from models import *

#load data
def load_ucr(name, split="train"):
    try:
        result = load_classification(name, split=split)
    except TypeError:
        result = load_classification(name, split=split, meta_data=True)
    
    if isinstance(result, tuple) and len(result) == 3:\
        X, y, _ = result
    else:
        X, y = result

    if X.ndim == 3:
        X = X.squeeze(axis=1)

    X = X.astype(np.float32)
    classes = np.unique(y)
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[c] for c in y], dtype=np.int64)

    return X, y

def make_forecasting_pairs(X, horizon_frac=0.25):
    n, T = X.shape
    H = max(1, int(T*horizon_frac))
    return X[:, :-H], X[:, -H:]

#train
def train_classifier(model, X_train, y_train, X_val, y_val, epochs=100, lr=1e-3, batch_size=64, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_train = torch.tensor(X_train, device=device)
    y_train = torch.tensor(y_train, device=device)
    X_val = torch.tensor(X_val, device=device)
    y_val = torch.tensor(y_val, device=device)

    best_val = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train), device=device)

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_train[idx])
            loss = loss_fn(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        model.eval()

        with torch.no_grad():
            val_logits = model(X_val)
            val_accuracy = (val_logits.argmax(-1) == y_val).float().mean().item()

        if val_accuracy > best_val:
            best_val = val_accuracy
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return best_val

def train_forecaster(model, X_train, y_train, X_val, y_val, epochs=100, lr=1e-3, batch_size=64, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_train = torch.tensor(X_train, device=device)
    y_train = torch.tensor(y_train, device=device)
    X_val = torch.tensor(X_val, device=device)
    y_val = torch.tensor(y_val, device=device)

    best_val = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train), device=device)

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_train[idx])
            loss = loss_fn(pred, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()

        with torch.no_grad():
            val_pred = model(X_val)
            val_mse = loss_fn(val_pred, y_val).item()

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return best_val

#experiments
def run_classification(name, tokenizer_kind, seed, epochs=100, device="cpu"):
    X, y = load_ucr(name)

    #70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

    #hyperparameters
    m = compute_embedding_dim(X_train[0], tau=compute_tau_mi(X_train[0]))
    tau = compute_tau_mi(X_train[0])
    patch_size = 16

    model = ClassificationModel(tokenizer_kind=tokenizer_kind, n_classes=len(np.unique(y)), d_model=64, n_layers=4, patch_size=patch_size, m=m, tau=tau)
    best_val = train_classifier(model, X_train, y_train, X_val, y_val, epochs=epochs, device=device)
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test, device=device))
        accuracy = (logits.argmax(-1).cpu().numpy() == y_test).mean()

    return {"dataset": name, "tokenizer": tokenizer_kind, "seed": seed, "task": "classification", "val_accuracy": best_val, "test_accuracy": float(accuracy)}

def run_forecasting(name, tokenizer_kind, seed, epochs=100, device="cpu"):
    X, _ = load_ucr(name)
    X_in, y_out = make_forecasting_pairs(X)
    n = len(X)
    train_end = int(0.7*n)
    val_end = int(0.85*n)
    X_train, y_train = X_in[:train_end], y_out[:train_end]
    X_val, y_val = X_in[train_end:val_end], y_out[train_end:val_end]
    X_test, y_test = X_in[val_end:], y_out[val_end:]

    m = compute_embedding_dim(X_train[0], tau = compute_tau_mi(X_train[0]))
    tau = compute_tau_mi(X_train[0])

    model = ForecastingModel(tokenizer_kind=tokenizer_kind, horizon=y_train.shape[1], d_model=64, n_layers=4, patch_size=16, m=m, tau=tau)
    best_val = train_forecaster(model, X_train, y_train, X_val, y_val, epochs=epochs, device=device)
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(X_test, device=device)).cpu().numpy()
    mse = float(np.mean(pred - y_test)**2)

    return {"dataset": name, "tokenizer": tokenizer_kind, "seed": seed, "task": "forecasting", "val_mse": best_val, "test_mse": mse}

#main
if __name__=="__main__":
    print("SCRIPT LOADED")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pilot", action="store_true", help="Run 2 datasets only")
    args = parser.parse_args()
    

    classification_datasets = ["ECG200", "FordA", "UWaveGestureLibraryAll", "Yoga", "GunPoint",
        "CBF", "Lightning2", "Wafer", "ElectricDevices", "EthanolLevel",
        "Plane", "Strawberry", "ScreenType", "SonyAIBORobotSurface1"]
    forecasting_datasets = ["ECG200", "FordA", "Wafer", "ElectricDevices"]

    print(f"Pilot={args.pilot}, datasets={classification_datasets}, seeds={args.seeds}, device={args.device}", flush=True)

    if args.pilot:
        classification_datasets = ["ECG200", "Wafer"]
        forecasting_datasets = ["ECG200"]

    rows = []
    t0 = time.time()

    for name in classification_datasets:
        for tok in ["value", "delay", "shuffle"]:
            for seed in range(args.seeds):
                print(f"[class] {name} | {tok} | seed={seed}")
                rows.append(run_classification(name, tok, seed, epochs=args.epochs, device=args.device))

    for name in forecasting_datasets:
        for tok in ["value", "delay"]:
            for seed in range(args.seeds):
                print(f"[forecast] {name} | {tok} | seed={seed}")
                rows.append(run_forecasting(name, tok, seed, epochs=args.epochs, device=args.device))

    all_fields = ["dataset", "tokenizer", "seed", "task", "val_accuracy", "test_accuracy", "val_mse", "test_mse"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone in {time.time() - t0:.1f}s. Results printed to {args.out}")