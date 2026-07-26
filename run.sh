#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release

# Activate the Python venv that has mediapipe + opencv
source /home/dranzer/hand-control/.venv/bin/activate

exec ./target/release/air-mouse "$@"
