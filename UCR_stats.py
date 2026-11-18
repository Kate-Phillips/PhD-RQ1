import numpy as np
from aeon.datasets import load_classification
from scipy import stats

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

def outlier_diagnostics(X, name):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    #global standardization
    mu = float(np.mean(X))
    sd = float(np.std(X))
    if sd == 0:
        return None

    z = (X - mu) / sd

    global_max_z = float(np.abs(z).max())
    frac_5 = float((np.abs(z) > 5).mean())
    frac_10 = float((np.abs(z) > 10).mean())

    #per-sample max |z| and kurtosis
    per_sample_max = []
    per_sample_kurtosis = []

    for i in range(X.shape[0]):
        x_i = X[i]
        mu_i, sd_i = x_i.mean(), x_i.std()
        if sd_i == 0:
            continue
        z_i = (x_i - mu_i) / sd_i
        per_sample_max.append(np.abs(z_i).max())
        
        #excess kurtosis: 0 for guassian, >0 for heavy tails
        per_sample_kurtosis.append(stats.kurtosis(x_i, fisher=True))

    return {
        "dataset": name,
        "n_samples": X.shape[0],
        "n_timesteps": X.shape[1],
        "min_val": float(X.min()),
        "max_val": float(X.max()),
        "global_max_z": global_max_z,
        "frac_beyond_5z": frac_5,
        "frac_beyond_10z": frac_10,
        "per_sample_max_z_mean": float(np.mean(per_sample_max)),
        "per_sample_max_z_max": float(np.max(per_sample_max)),
        "kurtosis_mean": float(np.mean(per_sample_kurtosis))
    }

if __name__ == "__main__":
    datasets = [
        "ECG200", "FordA", "Wafer", "ElectricDevices", "EthanolLevel",
        "Plane", "Strawberry", "GunPoint", "CBF", "Yoga",
        "UWaveGestureLibraryAll", "Lightning2", "ScreenType",
        "SonyAIBORobotSurface1",
    ]

    print(f"{'dataset':<28} {'max_z':>8} {'>5z':>8} {'>10z':>8} "
          f"{'per_samp_max':>14} {'kurtosis':>10} {'range':>20}")

    for name in datasets:
        try:
            X, y = load_ucr(name)
            d = outlier_diagnostics(X, name)
            if d is None:
                print(f"{name:<28} (constant data, skipped)")
                continue
            print(f"{d['dataset']:<28} "
                  f"{d['global_max_z']:>8.3f} "
                  f"{d['frac_beyond_5z']:>8.4f} "
                  f"{d['frac_beyond_10z']:>8.4f} "
                  f"{d['per_sample_max_z_mean']:>14.3f} "
                  f"{d['kurtosis_mean']:>10.3f} "
                  f"\t[{d['min_val']:.2f}, {d['max_val']:.2f}]")
        except Exception as e:
            print(f"{name:<28} ERROR: {e}")
