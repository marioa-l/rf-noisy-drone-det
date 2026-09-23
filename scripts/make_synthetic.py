"""Generate a small synthetic dataset in both the v1 (memmap) and v2 (per-file)
layouts to smoke-test the whole pipeline without the real data.

    python scripts/make_synthetic.py data/synthetic [--n 280] [--v2-length 65536]

Drone classes are bursts of a class-specific tone (plus a class-specific
repetition pattern) mixed with Gaussian noise at the sample's SNR, so the
models have something learnable. Noise class is pure Gaussian noise.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfdrone.data import DEFAULT_CLASSES, complex_spectrogram

SNRS = list(range(-20, 31, 2))


def synth_iq(cls, snr_db, length, rng):
    noise = rng.standard_normal((2, length)).astype(np.float32) / np.sqrt(2)
    if DEFAULT_CLASSES[cls] == "Noise":
        return noise
    f = (cls + 1) / 20.0  # class-specific normalised tone frequency
    t = np.arange(length)
    burst = np.zeros(length, dtype=np.float32)
    period, width = length // (cls + 2), length // 12
    for s in range(0, length, period):
        burst[s:s + width] = 1.0
    sig = np.stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]).astype(np.float32) * burst
    sig /= np.sqrt((sig ** 2).sum(0)[burst > 0].mean())
    k = 10 ** (snr_db / 10)
    return ((np.sqrt(k) * sig + noise) / np.sqrt(k + 1)).astype(np.float32)


def main(out, n, v2_length, v1_length=16384, v1_nfft=128):
    rng = np.random.default_rng(0)
    out = Path(out)
    n_cls = len(DEFAULT_CLASSES)
    y = np.array([i % n_cls for i in range(n)])
    snr = np.array([SNRS[i % len(SNRS)] for i in range(n)])
    stats = pd.DataFrame({"class": DEFAULT_CLASSES, "count": np.bincount(y, minlength=n_cls)})
    snr_stats = pd.DataFrame({"SNR": SNRS, "count": [int((snr == s).sum()) for s in SNRS]})

    v1 = out / "v1_memmap"
    v1.mkdir(parents=True, exist_ok=True)
    x_iq = np.lib.format.open_memmap(v1 / "x_iq.npy", "w+", np.float32, (n, 2, v1_length))
    x_spec = np.lib.format.open_memmap(v1 / "x_spec.npy", "w+", np.float32, (n, 2, v1_nfft, v1_length // v1_nfft))
    for i in range(n):
        x_iq[i] = synth_iq(y[i], snr[i], v1_length, rng)
        x_spec[i] = complex_spectrogram(torch.from_numpy(x_iq[i]), n_fft=v1_nfft).numpy()
    x_iq.flush(), x_spec.flush()
    np.save(v1 / "y.npy", y), np.save(v1 / "snr.npy", snr), np.save(v1 / "duty_cycle.npy", np.zeros(n))
    stats.to_csv(v1 / "class_stats.csv"), snr_stats.to_csv(v1 / "SNR_stats.csv")

    v2 = out / "v2_files"
    v2.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        torch.save({"x_iq": torch.from_numpy(synth_iq(y[i], snr[i], v2_length, rng)),
                    "y": torch.tensor(y[i]), "snr": torch.tensor(snr[i])},
                   v2 / f"IQdata_sample{i}_target{y[i]}_snr{snr[i]}.pt")
    stats.to_csv(v2 / "class_stats.csv"), snr_stats.to_csv(v2 / "SNR_stats.csv")
    print(f"synthetic data: {n} samples -> {v1} and {v2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=280)
    ap.add_argument("--v2-length", type=int, default=65536)
    a = ap.parse_args()
    main(a.out, a.n, a.v2_length)
