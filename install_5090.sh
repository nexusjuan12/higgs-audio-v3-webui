#!/usr/bin/env bash
set -euo pipefail

export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
export TORCH_VERSION="${TORCH_VERSION:-2.7.0+cu128}"
export TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.0+cu128}"
export HIGGS_DTYPE="${HIGGS_DTYPE:-auto}"
export HIGGS_ASR_DTYPE="${HIGGS_ASR_DTYPE:-auto}"

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install_common.sh" "$@"
