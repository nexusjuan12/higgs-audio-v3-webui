#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${HIGGS_APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PID_FILE="${HIGGS_PID_FILE:-$APP_DIR/higgs-audio-v3-webui.pid}"

pids=()
if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    pids+=("$pid")
  fi
fi

while IFS= read -r pid; do
  pids+=("$pid")
done < <(pgrep -f "$APP_DIR/[h]iggs_v3_webui.py" || true)

mapfile -t pids < <(printf "%s\n" "${pids[@]}" | sort -u)

if [[ "${#pids[@]}" -eq 0 ]]; then
  rm -f "$PID_FILE"
  echo "No running Higgs Audio v3 UI process found."
  exit 0
fi

kill "${pids[@]}"
for _ in {1..30}; do
  alive=0
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
      break
    fi
  done
  if [[ "$alive" == "0" ]]; then
    rm -f "$PID_FILE"
    echo "Stopped Higgs Audio v3 UI."
    exit 0
  fi
  sleep 1
done

echo "Process did not stop after 30s; sending SIGKILL."
kill -9 "${pids[@]}" 2>/dev/null || true
rm -f "$PID_FILE"
