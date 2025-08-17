#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
REPO_ID="nasa-ibm-ai4science/Surya-bench-solarwind"
REPO_TYPE="dataset"               # change to "model" if needed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="${SCRIPT_DIR}/assets"  # required to exist (per your spec)
TARGET_DIR="${ASSET_DIR}/${REPO_ID#*/}"   # assets/Surya-bench-solarwind

# Optional: use an existing token non-interactively
HF_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"

have() { command -v "$1" >/dev/null 2>&1; }

# ---- Step 1: Check login ----
# Prefer huggingface-cli; fallback to "hf"; otherwise, Python one-liner.
if have huggingface-cli; then
  HFCLI="huggingface-cli"
elif have hf; then
  HFCLI="hf"
else
  HFCLI=""
fi

echo "==> Checking Hugging Face login..."
logged_in=1
if [[ -n "${HFCLI}" ]]; then
  if ${HFCLI} whoami >/dev/null 2>&1; then
    logged_in=0
  fi
fi

if (( logged_in != 0 )); then
  if [[ -n "${HF_TOKEN}" && -n "${HFCLI}" ]]; then
    echo "-> Logging in with token from environment via ${HFCLI}..."
    # Try both syntaxes; ignore failures and re-check
    ${HFCLI} login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null 2>&1 || true
    ${HFCLI} whoami >/dev/null 2>&1 || { echo "ERROR: Token login failed."; exit 1; }
  elif [[ -n "${HFCLI}" ]]; then
    echo "You are not logged in. Launching interactive login..."
    ${HFCLI} login
    ${HFCLI} whoami >/dev/null 2>&1 || { echo "ERROR: Login failed."; exit 1; }
  else
    # No CLI available; ensure we at least have a token for the Python fallback.
    [[ -n "${HF_TOKEN}" ]] || { echo "ERROR: Neither 'huggingface-cli'/'hf' found nor HF token set. Install 'huggingface_hub' or export HUGGINGFACE_HUB_TOKEN."; exit 1; }
  fi
fi
echo "✓ Hugging Face auth OK"

# ---- Step 2: Check assets directory exists next to the script ----
echo "==> Checking assets directory at: ${ASSET_DIR}"
if [[ ! -d "${ASSET_DIR}" ]]; then
  echo "ERROR: Required directory '${ASSET_DIR}' does not exist."
  echo "Create it (e.g., 'mkdir -p \"${ASSET_DIR}\"') and re-run."
  exit 1
fi

# ---- Step 3: Download the repo into assets ----
echo "==> Downloading ${REPO_TYPE} '${REPO_ID}' to '${TARGET_DIR}'"

if [[ -n "${HFCLI}" && "${HFCLI}" == "huggingface-cli" ]]; then
  # Older CLI name
  huggingface-cli download "${REPO_ID}" \
    --repo-type "${REPO_TYPE}" \
    --local-dir "${TARGET_DIR}" \
    --local-dir-use-symlinks False
elif [[ -n "${HFCLI}" && "${HFCLI}" == "hf" ]]; then
  # Newer CLI alias
  hf snapshot download "${REPO_ID}" \
    --repo-type "${REPO_TYPE}" \
    --local-dir "${TARGET_DIR}" \
    --local-dir-use-symlinks False
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
print("Download complete:", local_dir)
PY
fi

echo "✓ Done. Files are in: ${TARGET_DIR}"
