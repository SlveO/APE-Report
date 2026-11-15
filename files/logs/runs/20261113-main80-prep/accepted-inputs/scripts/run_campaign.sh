#!/usr/bin/env bash
set -euo pipefail
APE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$APE_SCRIPT_DIR/with_env.sh" /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B -u -m main80.campaign "$@"
