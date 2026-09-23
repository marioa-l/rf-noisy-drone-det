"""Datasets (v1 memmap and v2 per-file), spectrogram transform and CV splits.

Sample layout returned by every dataset: (x, y, snr, sample_id) where
  x   : float32 tensor, (2, L) for IQ, (2, F, T) complex spectrogram [Re, Im]
        or (1, F, T) log-power spectrogram
  y   : class index (long), snr: int, sample_id: int (stable across runs)
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

FILE_RE = re.compile(r"IQdata_sample(\d+)_target(\d+)_snr(-?\d+)\.pt$")
DEFAULT_CLASSES = ["DJI", "FutabaT14", "FutabaT7", "Graupner", "Noise", "Taranis", "Turnigy"]


# ------------------------------------------------------------ transforms
def complex_spectrogram(iq, n_fft=1024, hop=None):
    """Two-sided complex STFT of an IQ signal, scaled by 1/n_fft.

    iq: (2, L) or (B, 2, L). Returns (2, F, T) or (B, 2, F, T) with channels
    [Re, Im]. Equivalent to the authors' torchaudio Spectrogram(power=None,
    center=False, onesided=False, hann window) followed by /win_length.
    """
    single = iq.dim() == 2
    if single:
        iq = iq[None]
    hop = hop or n_fft
    z = torch.complex(iq[:, 0].float(), iq[:, 1].float())
    win = torch.hann_window(n_fft, device=iq.device)
    spec = torch.stft(z, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win,
                      center=False, onesided=False, return_complex=True)
    spec = torch.view_as_real(spec).permute(0, 3, 1, 2) / n_fft
    return spec[0] if single else spec


def log_power(spec, eps=1e-12):
    """(.., 2, F, T) complex spectrogram -> (.., 1, F, T) log10 magnitude, fft-shifted."""
    mag = torch.sqrt(spec[..., 0, :, :] ** 2 + spec[..., 1, :, :] ** 2 + eps)
    return torch.fft.fftshift(torch.log10(mag), dim=-2).unsqueeze(-3)


def to_representation(iq, representation, spec_mode="complex", n_fft=1024, spec=None):
    """Map a raw IQ sample (2, L) to the model input. `spec` is an optional
    precomputed complex spectrogram (v1 dataset ships one)."""
    if representation == "iq":
        return iq.float()
    spec = spec if spec is not None else complex_spectrogram(iq, n_fft=n_fft)
    return spec.float() if spec_mode == "complex" else log_power(spec.float())


def input_channels(representation, spec_mode):
    return 1 if (representation == "spec" and spec_mode == "logpower") else 2


# -------------------------------------------------------------- datasets
def read_class_names(data_dir):
    p = Path(data_dir) / "class_stats.csv"
    if p.exists():
        return list(pd.read_csv(p, index_col=0)["class"].values)
    return DEFAULT_CLASSES


class BaseDroneDataset(Dataset):
    representation = "iq"
    spec_mode = "complex"
    n_fft = 1024

    def configure(self, representation, spec_mode, n_fft):
        self.representation, self.spec_mode, self.n_fft = representation, spec_mode, n_fft
        return self

    @property
    def in_channels(self):
        return input_channels(self.representation, self.spec_mode)

    def index_of(self, sample_ids):
        """Positions inside this (possibly subsampled) dataset for the given sample ids."""
        lut = {int(s): i for i, s in enumerate(self.sample_ids[self.keep])}
        return [lut[int(s)] for s in sample_ids]

    def full_index(self, sample_id):
        """Position of a sample id in the complete on-disk dataset."""
        return int(np.flatnonzero(self.sample_ids == int(sample_id))[0])


class V1MemmapDataset(BaseDroneDataset):
    """Dataset v1 (NCTA 2023) converted with scripts/convert_v1.py to .npy memmaps:
    x_iq (N,2,16384), x_spec (N,2,128,128), y (N,), snr (N,), duty_cycle (N,)."""

    def __init__(self, data_dir, limit=None, seed=0):
        self.data_dir = Path(data_dir)
        self.targets = np.load(self.data_dir / "y.npy").astype(np.int64)
        self.snrs = np.load(self.data_dir / "snr.npy").astype(np.int64)
        self.sample_ids = np.arange(len(self.targets))
        self.class_names = read_class_names(self.data_dir)
        self.keep = _stratified_subset(self.targets, limit, seed)
        self._arrays = None  # memmaps are opened lazily (and per worker process)

    def _mm(self):
        if self._arrays is None:
            self._arrays = (np.load(self.data_dir / "x_iq.npy", mmap_mode="r"),
                            np.load(self.data_dir / "x_spec.npy", mmap_mode="r"))
        return self._arrays

    def __getstate__(self):
        return {**self.__dict__, "_arrays": None}

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, i):
        j = self.keep[i]
        x_iq, x_spec = self._mm()
        iq = torch.from_numpy(np.array(x_iq[j]))
        spec = torch.from_numpy(np.array(x_spec[j])) if self.representation == "spec" else None
        x = to_representation(iq, self.representation, self.spec_mode, self.n_fft, spec=spec)
        return x, int(self.targets[j]), int(self.snrs[j]), int(self.sample_ids[j])

    def raw(self, sample_id):
        """IQ (2, L) and complex spectrogram (2, F, T) numpy arrays for plotting."""
        x_iq, x_spec = self._mm()
        j = self.full_index(sample_id)
        return np.array(x_iq[j]), np.array(x_spec[j])


class V2FileDataset(BaseDroneDataset):
    """Dataset v2 (JRFID 2024): one IQdata_sample{id}_target{y}_snr{s}.pt per sample
    holding {'x_iq': (2, 1048576) float, 'y': long, 'snr': int}."""

    def __init__(self, data_dir, limit=None, seed=0):
        self.data_dir = Path(data_dir)
        files = sorted(f for f in os.listdir(self.data_dir) if FILE_RE.match(f))
        if not files:
            raise FileNotFoundError(f"no IQdata_sample*.pt files in {self.data_dir}")
        meta = np.array([[int(g) for g in FILE_RE.match(f).groups()] for f in files])
        order = np.argsort(meta[:, 0])
        self.files = [files[i] for i in order]
        self.sample_ids, self.targets, self.snrs = meta[order, 0], meta[order, 1], meta[order, 2]
        self.class_names = read_class_names(self.data_dir)
        self.keep = _stratified_subset(self.targets, limit, seed)

    def __len__(self):
        return len(self.keep)

    def _load_iq(self, j):
        d = torch.load(self.data_dir / self.files[j], map_location="cpu", weights_only=False)
        return d["x_iq"].float()

    def __getitem__(self, i):
        j = self.keep[i]
        x = to_representation(self._load_iq(j), self.representation, self.spec_mode, self.n_fft)
        return x, int(self.targets[j]), int(self.snrs[j]), int(self.sample_ids[j])

    def raw(self, sample_id):
        iq = self._load_iq(self.full_index(sample_id))
        return iq.numpy(), complex_spectrogram(iq, self.n_fft).numpy()


def _stratified_subset(targets, limit, seed):
    idx = np.arange(len(targets))
    if limit is None or limit >= len(targets):
        return idx
    keep, _ = train_test_split(idx, train_size=limit, stratify=targets, random_state=seed)
    return np.sort(keep)


def build_dataset(cfg):
    """cfg['data']: {format: v1|v2, path, representation: iq|spec,
    spec_mode: complex|logpower, n_fft, limit_samples}."""
    d = cfg["data"]
    path = os.path.expandvars(d["path"])
    cls = {"v1": V1MemmapDataset, "v2": V2FileDataset}[d["format"]]
    ds = cls(path, limit=d.get("limit_samples"), seed=cfg.get("seed", 0))
    return ds.configure(d.get("representation", "spec"), d.get("spec_mode", "complex"),
                        d.get("n_fft", 1024))


# ---------------------------------------------------------------- splits
def make_splits(sample_ids, targets, n_folds, seed, val_frac=0.2):
    """Stratified K-fold on class; each sample is in exactly one test fold.
    Within the training part a stratified validation subset is held out."""
    sample_ids, targets = np.asarray(sample_ids), np.asarray(targets)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for k, (tr, te) in enumerate(skf.split(sample_ids, targets)):
        tr, va = train_test_split(tr, test_size=val_frac, stratify=targets[tr], random_state=seed + k)
        splits.append({"train": sample_ids[tr].tolist(), "val": sample_ids[va].tolist(),
                       "test": sample_ids[te].tolist()})
    return splits


def get_splits(cfg, dataset):
    """Load the split file named in cfg['splits']['file'] or create it. Sharing
    the file between the IQ and the spectrogram experiments guarantees that
    both models are tested on identical samples in every fold."""
    s = cfg["splits"]
    path = Path(os.path.expandvars(s["file"]))
    ids = dataset.sample_ids[dataset.keep]
    targets = dataset.targets[dataset.keep]
    if path.exists():
        splits = json.load(open(path))
        got = sorted(sum((f["test"] for f in splits), []))
        if got != sorted(ids.tolist()):
            raise ValueError(f"split file {path} does not match the dataset sample ids")
        return splits
    splits = make_splits(ids, targets, s["n_folds"], cfg.get("seed", 0), s.get("val_frac", 0.2))
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(splits, open(path, "w"))
    return splits
