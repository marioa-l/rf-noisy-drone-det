#!/usr/bin/env bash
# Download the datasets (Kaggle API token required: ~/.kaggle/kaggle.json,
# created at kaggle.com > Settings > API > "Create New Token") and the
# authors' pretrained weights (Zenodo, no login).
#
#   export DATA_ROOT=/Volumes/<external-disk>/rf_drone_data
#   bash scripts/download_data.sh v1      # 23 GB zip  -> $DATA_ROOT/v1_raw/dataset.pt
#   bash scripts/download_data.sh v2      # 125 GB zip -> $DATA_ROOT/v2_files/*.pt (needs ~250 GB free while unzipping)
#   bash scripts/download_data.sh zenodo  # 181 MB     -> pretrained/vgg11_bn_CV5_epochs50_lr0.005_batchsize8/
set -euo pipefail
case "${1:-}" in
  v1)
    : "${DATA_ROOT:?set DATA_ROOT to a directory on the external disk}"
    mkdir -p "$DATA_ROOT/v1_raw"
    kaggle datasets download -d sgluege/noisy-drone-rf-signal-classification -p "$DATA_ROOT/v1_raw" --unzip
    ;;
  v2)
    : "${DATA_ROOT:?set DATA_ROOT to a directory on the external disk}"
    mkdir -p "$DATA_ROOT/v2_files"
    kaggle datasets download -d sgluege/noisy-drone-rf-signal-classification-v2 -p "$DATA_ROOT/v2_files" --unzip
    ;;
  zenodo)
    mkdir -p pretrained
    curl -L -o pretrained/zenodo.zip "https://zenodo.org/records/14065652/files/vgg11_bn_CV5_epochs50_lr0.005_batchsize8.zip?download=1"
    unzip -q -o pretrained/zenodo.zip -d pretrained -x "__MACOSX/*"
    rm pretrained/zenodo.zip
    ;;
  *)
    echo "usage: $0 {v1|v2|zenodo}"; exit 1 ;;
esac
