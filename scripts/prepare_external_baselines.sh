#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-/root/autodl-tmp/cer-rec/external_baselines}"
BASELINE_ARCHIVE="${BASELINE_ARCHIVE:-}"
MANIFEST="${BASELINE_ROOT}/baseline_manifest.jsonl"
mkdir -p "${BASELINE_ROOT}"

import_archive() {
  local archive="$1"
  if [[ -z "${archive}" || ! -f "${archive}" ]]; then
    return 1
  fi
  echo "Importing external baselines from ${archive}"
  tar -xzf "${archive}" -C "${BASELINE_ROOT}"
  if [[ -d "${BASELINE_ROOT}/external_baselines_sources" ]]; then
    shopt -s dotglob nullglob
    for path in "${BASELINE_ROOT}/external_baselines_sources"/*; do
      mv "${path}" "${BASELINE_ROOT}/"
    done
    rmdir "${BASELINE_ROOT}/external_baselines_sources"
  fi
}

clone_or_keep() {
  local name="$1"
  local url="$2"
  if [[ -d "${BASELINE_ROOT}/${name}" ]]; then
    echo "Keeping existing ${name}"
    return 0
  fi
  echo "Cloning ${name} from ${url}"
  git clone --depth 1 "${url}" "${BASELINE_ROOT}/${name}"
}

if ! clone_or_keep LLM-ESR https://github.com/Applied-Machine-Learning-Lab/LLM-ESR.git; then
  echo "Clone failed for LLM-ESR; trying archive fallback"
  import_archive "${BASELINE_ARCHIVE}" || true
fi

clone_or_keep LLM-ESR-liuqidong https://github.com/liuqidong07/LLM-ESR.git || true
clone_or_keep RLMRec https://github.com/HKUDS/RLMRec.git || true

: > "${MANIFEST}"
for name in LLM-ESR LLM-ESR-liuqidong RLMRec; do
  dir="${BASELINE_ROOT}/${name}"
  if [[ -d "${dir}" ]]; then
    commit="unknown"
    if [[ -d "${dir}/.git" ]]; then
      commit=$(git -C "${dir}" rev-parse HEAD 2>/dev/null || echo unknown)
    fi
    python - "$name" "$dir" "$commit" >> "${MANIFEST}" <<'PY'
import json, sys
name, path, commit = sys.argv[1:4]
print(json.dumps({"name": name, "path": path, "commit": commit}, ensure_ascii=False))
PY
  fi
done

echo "Baseline manifest written to ${MANIFEST}"
cat "${MANIFEST}"
