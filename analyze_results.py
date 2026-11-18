import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n_a, n_b = len(a), len(b)
    pooled = np.sqrt(((n_a-1)*a.var(ddof=1)+(n_b-1)*b.var(ddof=1)) / (n_a+n_b-2))
    return (a.mean() - b.mean()) / (pooled + 1e-9)

def bootstrap_ci(values, n_boot=1000, alpha=0.50, rng=None):
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values)
    boots = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha / 2, 100*(1-alpha/2)])
    return float(np.mean(values)), float(lo), float(hi)

def main(path):
    df = pd.read_csv(path)

    #P1: delay vs value (classification)
    clf = df[df["task"] == "classification"]
    pivot = clf.pivot_table(index=["dataset", "seed"], columns="tokenizer", values="test_accuracy").reset_index()
    delay = pivot["delay"].values
    value = pivot["value"].values
    d = cohens_d(delay, value)
    t, p = stats.ttest_rel(delay, value)
    delta_class = delay - value
    mean_dc, lo_dc, hi_dc = bootstrap_ci(delta_class)
    print(f"\nP1: delay > value (classification)")
    print(f"- Δ accuracy = {mean_dc:.4f} [{lo_dc:.4f}, {hi_dc:.4f}]")
    print(f"- Cohen's d = {d:.3f}, paired t = {t:.3f}, p = {p:.4f}")

    #P2: task x tokenization interaction
    fc = df[df["task"] == "forecasting"]
    fc_pivot = fc.pivot_table(index=["dataset", "seed"], columns="tokenizer", values="test_mse").reset_index()

    #convert mse to a higher == better scale via negation
    fc_delta = -(fc_pivot["delay"].values - fc_pivot["value"].values)

    #normalize by baseline scale to make it comparable to classification accuracy
    fc_baseline = fc_pivot["value"].values
    fc_delta_norm = fc_delta / (np.abs(fc_baseline) + 1e-9)
    mean_df, lo_df, hi_df = bootstrap_ci(fc_delta_norm)
    print(f"\nP2: delay advantage larger for classification")
    print(f"- Classification Δ accuracy = {mean_dc:.4f} [{lo_dc:.4f}, {hi_dc:.4f}]")
    print(f"- Forecasting Δ (normalized mse) = {mean_df:.4f} [{lo_df:.4f}, {hi_df:.4f}]")
    print(f"- Interaction (class - forecast) = {mean_dc - mean_df:.4f}")

    #P3: shuffle ablation (classification only)
    shuffle = pivot["shuffle"].values
    d_shuffle = cohens_d(delay, shuffle)
    t_shuffle, p_shuffle = stats.ttest_rel(delay, shuffle)
    delta_shuffle = delay - shuffle
    mean_ds, lo_ds, hi_ds = bootstrap_ci(delta_shuffle)
    print(f"\nP3: delay > shuffle (classification)")
    print(f"- Δ accuracy = {mean_ds:.4f} [{lo_ds:.4f}, {hi_ds:.4f}]")
    print(f"- Cohen's d = {d_shuffle:.3f}, paired t = {t_shuffle:.3f}, p = {p_shuffle:.4f}")
    if mean_ds < 0.005:
        print("Δ gap is small. Delay advantage may be dimensionality, not structure")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.csv")
    args = parser.parse_args()
    main(args.results)