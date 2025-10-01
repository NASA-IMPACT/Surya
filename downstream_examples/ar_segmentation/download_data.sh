#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
REPO_ID="nasa-ibm-ai4science/surya-bench-ar-segmentation"
REPO_TYPE="dataset"               # change to "model" if needed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="${SCRIPT_DIR}/assets"  # required to exist (per your spec)
TARGET_DIR="${ASSET_DIR}/${REPO_ID#*/}"   # assets/Surya-bench-ar-segmentation

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

echo "✓ Download complete"

# ---- Step 4: Find and extract archives inside assets (recursive) ----
echo "==> Scanning for archives to extract under: ${ASSET_DIR}"

# Require tools
have tar   || { echo "ERROR: 'tar' not found."; exit 1; }
# unzip is optional; only required if we see a .zip
if ! have unzip; then
  NEED_UNZIP=$(find "${ASSET_DIR}" -type f -name '*.zip' -print -quit || true)
  if [[ -n "${NEED_UNZIP}" ]]; then
    echo "ERROR: 'unzip' not found but .zip files are present."
    exit 1
  fi
fi

extract_one() {
  local src="$1"
  local base dest
  case "$src" in
    *.tar.gz|*.tgz)   base="${src%.tar.gz}"; base="${base%.tgz}";;
    *.tar.bz2|*.tbz2) base="${src%.tar.bz2}"; base="${base%.tbz2}";;
    *.tar)            base="${src%.tar}";;
    *.zip)            base="${src%.zip}";;
    *) return 0;;
  esac
  dest="${base}_extracted"

  # Avoid re-extracting if destination exists and not empty
  if [[ -d "${dest}" ]] && [[ -n "$(ls -A "${dest}" 2>/dev/null || true)" ]]; then
    echo "-> Skipping (already extracted): ${src}"
    return 0
  fi

  echo "-> Extracting: ${src}"
  mkdir -p "${dest}"
  case "$src" in
    *.tar.gz|*.tgz)   tar -xzf "$src" -C "$dest" ;;
    *.tar.bz2|*.tbz2) tar -xjf "$src" -C "$dest" ;;
    *.tar)            tar -xf  "$src" -C "$dest" ;;
    *.zip)            unzip -q "$src" -d "$dest" ;;
  esac
}

# Use find -print0 to handle spaces/newlines safely
find "${ASSET_DIR}" -type f \
  \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.tar.bz2' -o -name '*.tbz2' -o -name '*.zip' \) \
  -print0 |
while IFS= read -r -d '' f; do
  extract_one "$f"
done


echo "✓ Archive extraction complete"
echo "All done. Data is under: ${TARGET_DIR}"