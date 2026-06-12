#!/usr/bin/env bash
set -euo pipefail

PID_FILE="${HIGGS_PID_FILE:-/root/higgs-audio-v3-webui.pid}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No pid file: $PID_FILE"
  exit 0
fi

pid="$(cat "$PID_FILE")"
if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "No running Higgs Audio v3 UI process found."
  exit 0
fi

kill "$pid"
for _ in {1..30}; do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Stopped Higgs Audio v3 UI."
    exit 0
  fi
  sleep 1
done

echo "Process did not stop after 30s; sending SIGKILL."
kill -9 "$pid" 2>/dev/null || true
rm -f "$PID_FILE"
