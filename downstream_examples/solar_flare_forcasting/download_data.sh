#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
REPO_ID="nasa-ibm-ai4science/surya-bench-flare-forecasting"
REPO_TYPE="dataset"               # change to "model" if needed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="${SCRIPT_DIR}/assets"  # required to exist (per your spec)
TARGET_DIR="${ASSET_DIR}/${REPO_ID#*/}"   # assets/Surya-bench-flare-forecasting

# Optional: use an existing token non-interactively
# Optional: use an existing token non-interactively
HF_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
HF_HUB_ENABLE_HF_TRANSFER=1

have() { command -v "$1" >/dev/null 2>&1; }

# ---- Step 1: Check login ----
# "hf"; otherwise, Python one-liner.
if have hf; then
  HFCLI="hf"
else
  HFCLI=""
fi

# ---- Step 2: Check assets directory exists next to the script ----
echo "==> Checking assets directory at: ${ASSET_DIR}"
if [[ ! -d "${ASSET_DIR}" ]]; then
  echo "ERROR: Required directory '${ASSET_DIR}' does not exist."
  echo "Create it (e.g., 'mkdir -p \"${ASSET_DIR}\"') and re-run."
  exit 1
fi

# ---- Step 3: Download the repo into assets ----
echo "==> Downloading ${REPO_TYPE} '${REPO_ID}' to '${TARGET_DIR}'"

if [[ -n "${HFCLI}" && "${HFCLI}" == "hf" ]]; then
  # Newer CLI alias
  hf download "${REPO_ID}" \
    --repo-type "${REPO_TYPE}" \
    --local-dir "${TARGET_DIR}" \

  hf download nasa-ibm-ai4science/solar_flares_surya \
  --repo-type model --local-dir "${ASSET_DIR}" \
  --include "*.pth"

  hf download nasa-ibm-ai4science/SDO_training \
  --repo-type dataset --local-dir "${ASSET_DIR}" \
  --include "*_index_surya_1_0.csv" "infer_data/*" "scalers.yaml"

else
  # Python fallback using the library API
  python3 - <<PY
import os
from huggingface_hub import snapshot_download
repo_id = "${REPO_ID}"
repo_type = "${REPO_TYPE}"
local_dir = r"${TARGET_DIR}"
token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
snapshot_download(repo_id, repo_type=repo_type, local_dir=local_dir,
                  local_dir_use_symlinks=False, token=token)
snapshot_download(repo_id="nasa-ibm-ai4science/SDO_training", repo_type=repo_type, local_dir=r"${ASSET_DIR}",
                  token=token, allow_patterns=["*_index_surya_1_0.csv", "infer_data/*", "scalers.yaml"])

print("Download complete:", local_dir)
PY
fi

echo "✓ Done. Files are in: ${TARGET_DIR}"
