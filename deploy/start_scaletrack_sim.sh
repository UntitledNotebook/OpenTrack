#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $(basename "$0") POLICY_NAME MOTION[,MOTION...] [start_sim options]"
    echo "Example: $(basename "$0") scaletrack_4010_landingdr_v1 scaletrack_smoke --iface lo"
    exit 2
fi

POLICY_NAME="$1"
MOTION_NAMES="$2"
shift 2

unset G1_TRACKER_POLICY G1_TRACKER_MOTIONS G1_VAE_POLICY G1_VAE_MOTIONS
export G1_SCALETRACK_POLICY="$POLICY_NAME"
export G1_SCALETRACK_MOTIONS="$MOTION_NAMES"
export G1_SCALETRACK_MODE="${G1_SCALETRACK_MODE:-7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/start_sim_w_torque_projection.sh" "$@"
