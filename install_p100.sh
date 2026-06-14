#!/usr/bin/env bash
set -euo pipefail

export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
export TORCH_VERSION="${TORCH_VERSION:-2.4.1+cu121}"
export TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.4.1+cu121}"
export HIGGS_DTYPE="${HIGGS_DTYPE:-float16}"
export HIGGS_ASR_DTYPE="${HIGGS_ASR_DTYPE:-float16}"

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install_common.sh" "$@"
