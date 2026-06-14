#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat <<'EOF'
Higgs Audio v3 Web UI has hardware-specific installers:

  ./install_5090.sh   RTX 5090 / Blackwell CUDA build
  ./install_p100.sh   Tesla P100 CUDA 12.1 build

Defaulting to ./install_5090.sh for compatibility with the primary GitHub/Vast target.
Set HIGGS_* environment variables before running if you need custom paths.

EOF

exec "$SCRIPT_DIR/install_5090.sh" "$@"
