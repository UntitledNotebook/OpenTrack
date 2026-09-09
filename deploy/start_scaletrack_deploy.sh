#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-}"
CHECK_ONLY=0

usage() {
    cat <<EOF
Usage: $(basename "$0") POLICY_NAME MOTION[,MOTION...] --iface IFACE [options]

Real-robot launcher for the OpenTrack ScaleTrack controller. Torque projection
is always enabled.

Required:
  POLICY_NAME          Directory under deploy/storage/policy
  MOTION[,MOTION...]   Ordered directories under deploy/storage/data
  --iface IFACE        Physical Unitree DDS interface, for example eth0

Options:
  --param PATH         Param directory relative to deploy/build/bin
  --check-only         Validate interface and assets without starting control
  -h, --help           Show this help

Environment:
  NETWORK_INTERFACE    Alternative to --iface
  G1_SCALETRACK_MODE   ScaleTrack body mode 0..7 (default: 7, WholeBody-14)

Example:
  $(basename "$0") scaletrack_4010_landingdr_v1 \
    scaletrack_single_jump,scaletrack_walk_slow --iface eth0
EOF
}

if [[ $# -eq 1 && ("$1" == "-h" || "$1" == "--help") ]]; then
    usage
    exit 0
fi

if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

POLICY_NAME="$1"
MOTION_NAMES="$2"
shift 2

if [[ ! "$POLICY_NAME" =~ ^[A-Za-z0-9._-]+$ || "$POLICY_NAME" == "." || "$POLICY_NAME" == ".." ]]; then
    echo "[ERROR] Invalid policy name: $POLICY_NAME"
    exit 2
fi

declare -a FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --iface)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --iface requires a value"
                exit 2
            fi
            NETWORK_INTERFACE="$2"
            FORWARD_ARGS+=("--iface" "$2")
            shift 2
            ;;
        --param)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --param requires a value"
                exit 2
            fi
            FORWARD_ARGS+=("--param" "$2")
            shift 2
            ;;
        --check-only)
            CHECK_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$NETWORK_INTERFACE" ]]; then
    echo "[ERROR] Real-robot deployment requires an explicit DDS interface."
    echo "[HINT] Pass --iface eth0 (or the actual robot-facing interface)."
    exit 2
fi

if [[ "$NETWORK_INTERFACE" == "lo" ]]; then
    echo "[ERROR] Refusing real-robot deployment on loopback interface 'lo'."
    echo "[HINT] Use start_scaletrack_sim.sh for MuJoCo simulation."
    exit 2
fi

if command -v ip >/dev/null 2>&1; then
    if ! ip link show "$NETWORK_INTERFACE" >/dev/null 2>&1; then
        echo "[ERROR] Network interface does not exist: $NETWORK_INTERFACE"
        ip -o link | awk -F': ' '{print "  " $2}' || true
        exit 2
    fi
elif [[ ! -d "/sys/class/net/$NETWORK_INTERFACE" ]]; then
    echo "[ERROR] Network interface does not exist: $NETWORK_INTERFACE"
    exit 2
fi

if [[ -n "${CYCLONEDDS_URI:-}" ]]; then
    if [[ "$CYCLONEDDS_URI" == *'name="lo"'* || "$CYCLONEDDS_URI" == *"name='lo'"* ]]; then
        echo "[ERROR] CYCLONEDDS_URI is still restricted to loopback."
        echo "[HINT] Run: unset CYCLONEDDS_URI"
        exit 2
    fi
fi
DEPLOY_EXECUTABLE="$SCRIPT_DIR/build/bin/state_machine_example"
BUILD_CACHE="$SCRIPT_DIR/build/CMakeCache.txt"
if [[ ! -x "$DEPLOY_EXECUTABLE" ]]; then
    echo "[ERROR] Deployment executable is missing: $DEPLOY_EXECUTABLE"
    echo "[HINT] Run: ./build_w_torque_projection.sh"
    exit 2
fi
if [[ ! -f "$BUILD_CACHE" ]] || ! grep -q '^ENABLE_DANCE_TORQUE_PROJECTION:BOOL=ON$' "$BUILD_CACHE"; then
    echo "[ERROR] The current build is not verified as torque-projection enabled."
    echo "[HINT] Run: ./build_w_torque_projection.sh"
    exit 2
fi


SCALETRACK_MODE="${G1_SCALETRACK_MODE:-7}"
if [[ ! "$SCALETRACK_MODE" =~ ^[0-7]$ ]]; then
    echo "[ERROR] G1_SCALETRACK_MODE must be an integer in 0..7, got: $SCALETRACK_MODE"
    exit 2
fi

POLICY_ROOT="$SCRIPT_DIR/storage/policy/$POLICY_NAME"
CHECKPOINT_ROOT="$POLICY_ROOT/checkpoints"
if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
    echo "[ERROR] ScaleTrack checkpoint directory not found: $CHECKPOINT_ROOT"
    exit 2
fi

LATEST_CHECKPOINT=""
LATEST_ITERATION=-1
shopt -s nullglob
for checkpoint_dir in "$CHECKPOINT_ROOT"/*; do
    [[ -d "$checkpoint_dir" ]] || continue
    checkpoint_name="$(basename "$checkpoint_dir")"
    [[ "$checkpoint_name" =~ ^[0-9]+$ ]] || continue
    [[ -f "$checkpoint_dir/policy.onnx" && -f "$checkpoint_dir/metadata.json" ]] || continue
    checkpoint_iteration=$((10#$checkpoint_name))
    if ((checkpoint_iteration > LATEST_ITERATION)); then
        LATEST_ITERATION=$checkpoint_iteration
        LATEST_CHECKPOINT="$checkpoint_dir"
    fi
done
shopt -u nullglob

if [[ -z "$LATEST_CHECKPOINT" ]]; then
    echo "[ERROR] No numeric checkpoint containing policy.onnx and metadata.json found under:"
    echo "        $CHECKPOINT_ROOT"
    exit 2
fi

IFS=',' read -r -a MOTIONS <<< "$MOTION_NAMES"
VALID_MOTION_COUNT=0
for motion in "${MOTIONS[@]}"; do
    motion="${motion#"${motion%%[![:space:]]*}"}"
    motion="${motion%"${motion##*[![:space:]]}"}"
    [[ -n "$motion" ]] || continue
    if [[ ! "$motion" =~ ^[A-Za-z0-9._-]+$ || "$motion" == "." || "$motion" == ".." ]]; then
        echo "[ERROR] Invalid motion name: $motion"
        exit 2
    fi
    reference_path="$SCRIPT_DIR/storage/data/$motion/scaletrack_ref.onnx"
    if [[ ! -f "$reference_path" ]]; then
        echo "[ERROR] ScaleTrack reference not found: $reference_path"
        exit 2
    fi
    ((VALID_MOTION_COUNT += 1))
done

if ((VALID_MOTION_COUNT == 0)); then
    echo "[ERROR] MOTION list is empty"
    exit 2
fi

if ((VALID_MOTION_COUNT > 20)); then
    echo "[ERROR] At most 20 ScaleTrack motions can be registered, got: $VALID_MOTION_COUNT"
    exit 2
fi

echo "========================================"
echo "SCALETRACK REAL-ROBOT DEPLOYMENT"
echo "Policy:             $POLICY_NAME"
echo "Checkpoint:         $(basename "$LATEST_CHECKPOINT")"
echo "Motions:            $MOTION_NAMES"
echo "Motion count:       $VALID_MOTION_COUNT"
echo "ScaleTrack mode:    $SCALETRACK_MODE"
echo "DDS interface:      $NETWORK_INTERFACE"
echo "Torque projection:  ON"
echo "========================================"

if ((CHECK_ONLY)); then
    echo "[SELFCHECK] PASS: assets and real-robot interface are ready."
    exit 0
fi

unset G1_TRACKER_POLICY G1_TRACKER_MOTIONS G1_VAE_POLICY G1_VAE_MOTIONS
export G1_SCALETRACK_POLICY="$POLICY_NAME"
export G1_SCALETRACK_MOTIONS="$MOTION_NAMES"
export G1_SCALETRACK_MODE="$SCALETRACK_MODE"

echo "[SAFETY] Suspend the robot securely and keep the emergency stop accessible."
echo "[SAFETY] The controller will wait at 'Press R2 to start!' before sending active control."

exec "$SCRIPT_DIR/start_deploy_w_torque_projection.sh" "${FORWARD_ARGS[@]}"
