"""Convert the v1 Kaggle file dataset.pt (~23 GB) into .npy memmaps.

    python scripts/convert_v1.py $DATA_ROOT/v1_raw/dataset.pt $DATA_ROOT/v1_memmap

torch.load(mmap=True) avoids holding the whole file in RAM; if the file was
saved in the legacy (non-zip) format the fallback loads it fully, which needs
about 26 GB of RAM (do this step on the HPC node in that case).
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


def main(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    try:
        d = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    except Exception as e:  # legacy format
        print("mmap load failed, loading fully:", e)
        d = torch.load(src, map_location="cpu", weights_only=False)
    for key, name in [("x_iq", "x_iq"), ("x_spec", "x_spec"), ("y", "y"), ("snr", "snr"),
                      ("duty_cycle", "duty_cycle")]:
        t = d[key]
        out = np.lib.format.open_memmap(dst / f"{name}.npy", mode="w+", dtype=np.float32 if t.is_floating_point() else np.int64,
                                        shape=tuple(t.shape))
        step = 2048
        for i in range(0, t.shape[0], step):
            out[i:i + step] = t[i:i + step].numpy()
        out.flush()
        print(f"{name}: shape {tuple(t.shape)} dtype {out.dtype}")
    for csv in ["class_stats.csv", "SNR_stats.csv"]:
        if (src.parent / csv).exists():
            shutil.copy(src.parent / csv, dst / csv)
    print("done ->", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
