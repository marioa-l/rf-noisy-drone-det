"""Example signals for the four agreement outcomes of the two models.

Builds one 2^20-sample vector per case with the burst structure of the dataset
(Table II of the JRFID 2024 paper: burst duration, repetition period, channel
spacing), mixes it with lab-like noise (Bluetooth hops, Wi-Fi packets) plus
Gaussian noise at the case SNR, and plots the log power spectrogram next to the
IQ trace.

    python scripts/make_example_signals.py [--out results/scenario] [--seed 3]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfdrone.data import complex_spectrogram, log_power
from rfdrone.utils import save_fig, save_json, setup_plot_style

FS, N, NFFT = 14e6, 2 ** 20, 1024
DUR_MS = N / FS * 1e3
# burst duration(s) [ms], repetition period [ms], channel spacing [MHz], burst bandwidth [MHz]
TX = {"DJI": ([2.18], 630, 1.7, 1.2), "FutabaT14": ([1.4], 330, 3.1, 1.0), "FutabaT7": ([1.7], 288, 2.0, 1.0),
      "Graupner": ([1.9, 3.7], 750, 1.0, 0.8), "Taranis": ([3.1, 4.4], 420, 1.5, 0.9),
      "Turnigy": ([1.3], 61, 2.0, 1.0)}
CASES = [
    dict(outcome="Both correct", color="#009E73", target="Taranis", snr=20, lab_noise=True,
         pred_spec="Taranis", conf_spec=0.99, pred_iq="Taranis", conf_iq=0.95),
    dict(outcome="Only spectrogram correct", color="#0072B2", target="DJI", snr=-12, lab_noise=False,
         pred_spec="DJI", conf_spec=0.86, pred_iq="Noise", conf_iq=0.71),
    dict(outcome="Only IQ correct", color="#E69F00", target="FutabaT7", snr=26, lab_noise=False,
         pred_spec="FutabaT14", conf_spec=0.63, pred_iq="FutabaT7", conf_iq=0.88),
    dict(outcome="Both wrong", color="#D55E00", target="DJI", snr=-20, lab_noise=True,
         pred_spec="Noise", conf_spec=0.79, pred_iq="Noise", conf_iq=0.90)]
BURST_MARK = "#D62728"


def envelope(n, edge=0.05):
    e = np.ones(n)
    k = max(1, int(n * edge))
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))
    e[:k], e[-k:] = ramp, ramp[::-1]
    return e


def bandlimited(n, bw, rng):
    """Unit-power complex noise limited to a bandwidth, as a generic burst waveform."""
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    X = np.fft.fft(x)
    X[np.abs(np.fft.fftfreq(n, 1 / FS)) > bw / 2] = 0
    x = np.fft.ifft(X)
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def add_burst(sig, start, length, bw, freq, rng, gain=1.0):
    length = min(length, len(sig) - start)
    if length < 128:
        return 0
    t = np.arange(length) / FS
    sig[start:start + length] += gain * bandlimited(length, bw, rng) * np.exp(2j * np.pi * freq * t) * envelope(length)
    return length


def drone_signal(cls, rng):
    """Burst train of one transmitter, each burst on its own hop channel."""
    durations, rep_ms, spacing, bw = TX[cls]
    x, mask, bursts = np.zeros(N, complex), np.zeros(N, bool), []
    period = int(rep_ms * 1e-3 * FS)
    n_chan = int(6.0 / spacing)
    start, k = int(rng.integers(0, min(period, N // 3))), 0
    while start < N:
        length = int(durations[k % len(durations)] * 1e-3 * FS)
        freq = rng.integers(-n_chan, n_chan + 1) * spacing * 1e6
        used = add_burst(x, start, length, bw * 1e6, freq, rng)
        if used:
            mask[start:start + used] = True
            bursts.append({"t0_ms": start / FS * 1e3, "t1_ms": (start + used) / FS * 1e3,
                           "f_mhz": freq / 1e6, "bw_mhz": bw})
        start, k = start + period, k + 1
    return x, mask, bursts


def noise_signal(rng, lab=True):
    """Gaussian noise plus, for lab noise, Bluetooth-like hops and Wi-Fi-like packets."""
    n = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    if lab:
        for _ in range(int(rng.integers(25, 40))):  # Bluetooth hops, 1 MHz, 0.4 ms
            add_burst(n, int(rng.integers(0, N)), int(0.4e-3 * FS), 1e6, rng.uniform(-6.5e6, 6.5e6), rng, gain=3.0)
        for _ in range(int(rng.integers(2, 5))):  # Wi-Fi packets across the band, 1 ms
            add_burst(n, int(rng.integers(0, N)), int(1.0e-3 * FS), 13e6, 0.0, rng, gain=1.5)
    return n / np.sqrt(np.mean(np.abs(n) ** 2))


def make_sample(cls, snr_db, lab_noise, rng):
    """Normalise signal and noise as in the paper, then mix at the target SNR."""
    x, mask, bursts = drone_signal(cls, rng)
    x = x / np.sqrt(np.mean(np.abs(x[mask]) ** 2))
    n = noise_signal(rng, lab=lab_noise)
    k = 10 ** (snr_db / 10)
    y = (np.sqrt(k) * x + n) / np.sqrt(k + 1)
    return np.stack([y.real, y.imag]).astype(np.float32), bursts


def add_zoom(ax, power, burst, pad_ms=2.5, pad_mhz=2.0):
    """Inset with the burst region, stretched to its own contrast."""
    t0, t1 = max(0.0, burst["t0_ms"] - pad_ms), min(DUR_MS, burst["t1_ms"] + pad_ms)
    f0, f1 = max(-7.0, burst["f_mhz"] - pad_mhz), min(7.0, burst["f_mhz"] + pad_mhz)
    n_f, n_t = power.shape
    ci = slice(int(t0 / DUR_MS * n_t), max(int(t1 / DUR_MS * n_t), int(t0 / DUR_MS * n_t) + 1))
    ri = slice(int((f0 + 7) / 14 * n_f), max(int((f1 + 7) / 14 * n_f), int((f0 + 7) / 14 * n_f) + 1))
    crop = power[ri, ci]
    axins = ax.inset_axes([0.58, 0.50, 0.40, 0.38])
    axins.imshow(power, origin="lower", aspect="auto", cmap="viridis", rasterized=True,
                 extent=[0, DUR_MS, -FS / 2e6, FS / 2e6],
                 vmin=np.percentile(crop, 20), vmax=np.percentile(crop, 99.8))
    axins.set_xlim(t0, t1)
    axins.set_ylim(f0, f1)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.grid(False)
    for s in axins.spines.values():
        s.set(color=BURST_MARK, lw=1.4)
    ax.indicate_inset_zoom(axins, edgecolor=BURST_MARK, lw=1.2, alpha=1.0)
    axins.set_title("zoom on drone burst", fontsize=9.5, color=BURST_MARK, fontweight="bold", pad=3,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))


def plot_case(fig, gs_spec, gs_iq, iq, bursts, case, show_legend=False):
    target, ok = case["target"], {True: "correct", False: "wrong"}
    power = log_power(complex_spectrogram(torch.from_numpy(iq), n_fft=NFFT))[0].numpy()

    ax = fig.add_subplot(gs_spec)
    im = ax.imshow(power, origin="lower", aspect="auto", cmap="viridis", rasterized=True,
                   extent=[0, DUR_MS, -FS / 2e6, FS / 2e6],
                   vmin=np.percentile(power, 5), vmax=np.percentile(power, 99.9))
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("Log power", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    for b in bursts:
        ax.add_patch(plt.Rectangle((b["t0_ms"] - 0.8, b["f_mhz"] - b["bw_mhz"]), b["t1_ms"] - b["t0_ms"] + 1.6,
                                   2 * b["bw_mhz"], fill=False, ec=BURST_MARK, lw=1.4, ls="--"))
    if bursts:
        add_zoom(ax, power, bursts[0])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (MHz)")
    ax.grid(False)
    ax.set_title(f"{case['outcome']}\nTrue class: {target}, SNR {case['snr']} dB", loc="left", fontsize=13,
                 color=case["color"], fontweight="bold")

    ax2 = fig.add_subplot(gs_iq)
    t = np.arange(0, N, 16) / FS * 1e3
    ax2.plot(t, iq[0, ::16], color="#1B1D21", lw=0.3, rasterized=True, label="I")
    ax2.plot(t, iq[1, ::16], color="#9AA0A6", lw=0.3, rasterized=True, label="Q")
    for b in bursts:
        ax2.axvspan(b["t0_ms"], b["t1_ms"], color=BURST_MARK, alpha=0.18, lw=0)
    lim = np.percentile(np.abs(iq), 99.97) * 1.25
    ax2.set_ylim(-lim, lim)
    ax2.set_xlim(0, DUR_MS)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title(f"Spectrogram model: {case['pred_spec']} ({ok[case['pred_spec'] == target]}, "
                  f"p={case['conf_spec']:.2f})\nIQ model: {case['pred_iq']} ({ok[case['pred_iq'] == target]}, "
                  f"p={case['conf_iq']:.2f})", loc="left", fontsize=11)
    if show_legend:
        ax2.legend(loc="upper right", fontsize=10, ncol=2, markerscale=4, framealpha=0.9)


def main(out, seed):
    setup_plot_style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples = [make_sample(c["target"], c["snr"], c["lab_noise"], rng) for c in CASES]

    fig = plt.figure(figsize=(16, 9))  # one slide with the four cases
    gs = fig.add_gridspec(2, 5, width_ratios=[1.35, 1, 0.25, 1.35, 1], hspace=0.85, wspace=0.85)
    for i, (case, (iq, bursts)) in enumerate(zip(CASES, samples)):
        col = (i % 2) * 3
        plot_case(fig, gs[i // 2, col], gs[i // 2, col + 1], iq, bursts, case, show_legend=i == 0)
    save_fig(fig, out / "example_signals_outcomes")

    for i, (case, (iq, bursts)) in enumerate(zip(CASES, samples)):  # one figure per case
        f = plt.figure(figsize=(12, 4))
        g = f.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.45)
        plot_case(f, g[0, 0], g[0, 1], iq, bursts, case, show_legend=True)
        save_fig(f, out / f"example_signal_{i}_{case['outcome'].lower().replace(' ', '_')}")

    save_json({"seed": seed, "sampling_rate_hz": FS, "vector_length": N, "n_fft": NFFT,
               "transmitters": {k: dict(zip(["burst_ms", "repetition_ms", "spacing_mhz", "bandwidth_mhz"], v))
                                for k, v in TX.items()},
               "cases": [{**c, "bursts": b} for c, (_, b) in zip(CASES, samples)]},
              out / "example_signals_cases.json")
    print("wrote", out / "example_signals_outcomes.pdf", "and 4 single-case figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scenario")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    main(a.out, a.seed)
