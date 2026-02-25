#!/bin/bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 \"command with spaces\" log_path"
  exit 1
fi

command="$1"
log_path="$2"

log_dir="$(dirname "$log_path")"
mkdir -p "$log_dir"

# source ~/.bashrc

# micromamba activate maggrav

# nohup bash -lc "$command" > "$log_path" 2>&1 &

nohup micromamba run -n maggrav bash -lc "$command" > "$log_path" 2>&1 &


echo "Started: $command"
echo "Logging to: $log_path"