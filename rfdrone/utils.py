"""Shared helpers: config loading, seeding, device selection, plotting style."""
import json
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

DRONE_CLASSES_DEFAULT_NOISE = "Noise"


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = str(path)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


# ---------------------------------------------------------------- plotting
def setup_plot_style():
    plt.style.use("seaborn-v0_8-paper")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 13,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def palette(n):
    return plt.cm.viridis(np.linspace(0.05, 0.85, n))


def save_fig(fig, path_without_ext):
    """Save a figure as PDF and PNG."""
    p = Path(path_without_ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"))
    fig.savefig(p.with_suffix(".png"), dpi=300)
    plt.close(fig)
