"""Quantify the label leak found by check_snr_shortcut.py.

The dataset normalises a drone vector over its burst samples only, while a noise
vector is normalised over the whole window, so after mixing at SNR = 10log10(k)
the total power of a drone sample is (k*d + 1)/(k + 1), with d the burst duty
cycle, while a noise sample sits at 1. One scalar therefore encodes the SNR and,
through d, the transmitter. This script measures how much of the task that single
number solves, and recovers d per class to show the mechanism.

    python scripts/analyze_shortcut.py --features results/shortcut/features.csv \
        --out results/shortcut_analysis
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfdrone.data import DEFAULT_CLASSES
from rfdrone.utils import palette, save_fig, save_json, setup_plot_style

FEATURES = ["power", "crest_factor", "duty_proxy", "kurtosis", "spectral_flatness",
            "occupied_bw", "peak_over_median_psd", "spectral_centroid", "spectral_spread"]
BANDS = [(-20, -12), (-10, -2), (0, 10), (12, 30)]


def cv_predict(X, y, seed=0):
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    return cross_val_predict(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))


def theoretical_power(snr_db, duty):
    k = 10 ** (np.asarray(snr_db) / 10)
    return (k * duty + 1) / (k + 1)


def plot_leak(power_tab, duty, acc_band, n_classes, noise_name, out):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.4))
    cols = palette(len(power_tab))
    for (name, row), c in zip(power_tab.iterrows(), cols):
        a0.plot(row.index.astype(float), row.values, marker="o", ms=4, lw=1.6, color=c, label=name)
    if duty:
        snr = np.arange(-20, 31, 2)
        a0.plot(snr, theoretical_power(snr, np.median(list(duty.values()))), ls="--", lw=1.4, color="black",
                label=r"$(k\,d+1)/(k+1)$")
    a0.set_yscale("log")
    a0.set_xlabel("SNR (dB)")
    a0.set_ylabel("Mean total power of the sample")
    a0.legend(fontsize=9, ncol=2)

    x = np.arange(len(acc_band))
    w = 0.38
    a1.bar(x - w / 2, acc_band.detection, w, color=palette(2)[0], label=f"Drone vs {noise_name}")
    a1.bar(x + w / 2, acc_band.multiclass, w, color=palette(2)[1], label=f"{n_classes} classes")
    for xi, (d, m) in enumerate(zip(acc_band.detection, acc_band.multiclass)):
        a1.text(xi - w / 2, d + 0.02, f"{d:.2f}", ha="center", fontsize=10)
        a1.text(xi + w / 2, m + 0.02, f"{m:.2f}", ha="center", fontsize=10)
    a1.axhline(0.5, color="grey", ls=":", lw=1.2)
    a1.axhline(1 / n_classes, color="grey", ls="--", lw=1.2)
    a1.set_xticks(x, [f"{lo} to {hi}" for lo, hi in BANDS])
    a1.set_xlabel("SNR band (dB)")
    a1.set_ylabel("Balanced accuracy from total power alone")
    a1.set_ylim(0, 1.08)
    a1.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    save_fig(fig, out)


def main(features, class_names, out):
    setup_plot_style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features)
    df["class"] = [class_names[t] for t in df.target]
    noise_id = class_names.index("Noise")
    df["is_drone"] = (df.target != noise_id).astype(int)
    n_classes = df.target.nunique()
    print(f"{len(df)} samples, {n_classes} classes present: {sorted(df['class'].unique())}")

    # --- how much does each feature set solve --------------------------------
    rows = {}
    for label, cols in [("power only", ["power"]), ("all features", FEATURES),
                        ("all minus power", [c for c in FEATURES if c != "power"])]:
        p_multi, p_det = cv_predict(df[cols], df.target), cv_predict(df[cols], df.is_drone)
        df[f"pred_multi_{label}"] = p_multi
        rows[label] = {"multiclass": balanced_accuracy_score(df.target, p_multi),
                       "detection": balanced_accuracy_score(df.is_drone, p_det)}
        df[f"pred_det_{label}"] = p_det
    acc = pd.DataFrame(rows).T
    acc.to_csv(out / "accuracy_by_feature_set.csv")
    print("\nbalanced accuracy (chance: "
          f"{1/n_classes:.3f} multiclass, 0.500 detection)\n{acc.round(3).to_string()}")

    # --- where the leak bites -------------------------------------------------
    band_rows = []
    for lo, hi in BANDS:
        g = df[(df.snr >= lo) & (df.snr <= hi)]
        band_rows.append({"band": f"{lo}..{hi}", "n": len(g),
                          "detection": balanced_accuracy_score(g.is_drone, g["pred_det_power only"]),
                          "multiclass": balanced_accuracy_score(g.target, g["pred_multi_power only"])})
    acc_band = pd.DataFrame(band_rows)
    acc_band.to_csv(out / "power_only_by_snr_band.csv", index=False)
    print(f"\npower alone, by SNR band:\n{acc_band.round(3).to_string(index=False)}")

    # --- recover the duty cycle implied by the mixing formula -----------------
    duty = {}
    for name, g in df[df.is_drone == 1].groupby("class"):
        k = 10 ** (g.snr / 10)
        d = ((k + 1) * g.power - 1) / k
        duty[name] = float(np.median(d))
    if duty:
        print("\nimplied burst duty cycle (window is 74.9 ms):")
        for name, d in sorted(duty.items()):
            print(f"  {name:12s} d = {d:.4f}  ->  {d * 74.9:.2f} ms of burst")

    power_tab = df.pivot_table(index="class", columns="snr", values="power", aggfunc="mean")
    power_tab.to_csv(out / "power_by_class_and_snr.csv")
    plot_leak(power_tab, duty, acc_band, n_classes, class_names[noise_id], out / "power_leak")
    save_json({"n_samples": len(df), "classes_present": sorted(df["class"].unique()),
               "accuracy_by_feature_set": rows, "power_only_by_band": band_rows,
               "implied_duty_cycle": duty,
               "note": "drone power follows (k*d+1)/(k+1); noise sits at 1, so total power "
                       "encodes both SNR and transmitter duty cycle"}, out / "summary.json")
    print("\nwrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--class-names", nargs="*", default=DEFAULT_CLASSES)
    ap.add_argument("--out", default="results/shortcut_analysis")
    a = ap.parse_args()
    main(a.features, list(a.class_names), a.out)
