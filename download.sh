#!/usr/bin/env bash
# Everything that needs a large download, in one place (run once, by hand).
#   1. Python environment with CUDA 12.6 PyTorch (~3 GB of wheels)
#   2. Raw single-cell data: sci-Plex3 (Zenodo 7041849, ~2 GB) and Tahoe-100M (Hugging Face, ~300 GB)
# The processed CSR store that data/build_embeddings.py reads is produced by the VirtCell data
# pipeline (vcdata, see its README); point VCDATA at its build/ directory if it lives elsewhere.
set -euo pipefail

ENV=${ENV:-cellot}
if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n "$ENV" python=3.11
fi
conda run -n "$ENV" pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
conda run -n "$ENV" pip install -r "$(dirname "$0")/requirements.txt"

RAW=${RAW:-data/raw}
mkdir -p "$RAW"
# sci-Plex3 (scPerturb h5ad, Srivatsan et al. 2020)
wget -c -O "$RAW/SrivatsanTrapnell2020_sciplex3.h5ad" \
  "https://zenodo.org/record/7041849/files/SrivatsanTrapnell2020_sciplex3.h5ad?download=1"
# Tahoe-100M (Zhang et al. 2025), pinned revision used in the paper
huggingface-cli download tahoebio/Tahoe-100M --repo-type dataset --revision affe86a --local-dir "$RAW/Tahoe-100M"
