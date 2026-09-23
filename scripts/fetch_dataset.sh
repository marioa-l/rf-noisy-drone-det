#!/usr/bin/env bash
# Download the v2 drone RF dataset on the Gottfried VM.
#
# Default mode streams the Kaggle archive straight into bsdtar, so the .pt files
# appear while the download runs and no 125 GB zip is ever stored. Run it inside
# tmux; it takes hours.
#
#   export DATA_ROOT=$HOME/rf_data          # or a scratch/project filesystem
#   tmux new -s dl
#   bash scripts/fetch_dataset.sh           # stream (default)
#   bash scripts/fetch_dataset.sh zip       # fallback: full zip, then unzip
#   bash scripts/fetch_dataset.sh status    # count files and bytes so far
#
# Needs ~150 GB free in stream mode, ~280 GB in zip mode, and a Kaggle API token
# at ~/.kaggle/kaggle.json (kaggle.com > Settings > API > Create New Token).
set -euo pipefail

DS="sgluege/noisy-drone-rf-signal-classification-v2"
DEST="${DATA_ROOT:-$HOME/rf_data}/v2_files"
MODE="${1:-stream}"
KJ="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"

if [ "$MODE" = "status" ]; then
    echo "files: $(find "$DEST" -name 'IQdata_sample*.pt' 2>/dev/null | wc -l) of 17744"
    du -sh "$DEST" 2>/dev/null || true
    exit 0
fi

mkdir -p "$DEST"
[ -f "$KJ" ] || { echo "missing $KJ (Kaggle token)"; exit 1; }
chmod 600 "$KJ"
KUSER=$(python -c "import json;print(json.load(open('$KJ'))['username'])")
KKEY=$(python -c "import json;print(json.load(open('$KJ'))['key'])")

echo "destination: $DEST"
df -h "$DEST" | tail -1

if [ "$MODE" = "zip" ]; then
    python -m pip install --quiet --upgrade kaggle
    kaggle datasets download -d "$DS" -p "$DEST" --unzip
else
    command -v bsdtar >/dev/null || { echo "bsdtar not found; run setup_env.sh or use 'zip' mode"; exit 1; }
    # --fail keeps an auth error page from being piped into bsdtar as if it were data.
    curl -fsSL --retry 8 --retry-delay 15 --retry-all-errors -u "$KUSER:$KKEY" \
        "https://www.kaggle.com/api/v1/datasets/download/$DS" \
        | bsdtar -x -f - -C "$DEST"
fi

echo "done: $(find "$DEST" -name 'IQdata_sample*.pt' | wc -l) sample files"
ls "$DEST" | grep -i csv || echo "note: class_stats.csv / SNR_stats.csv not seen yet"
