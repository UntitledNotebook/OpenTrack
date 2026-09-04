#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/build/bin"
EXECUTABLE="state_machine_example"
PARAM_PATH="../../state_machine/params/"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-lo}"
LOG_ARCHIVE_ROOT="$SCRIPT_DIR/deploy_logs"
CYCLONEDDS_URI_VALUE="${CYCLONEDDS_URI:-}"
ALLOW_NON_LOOPBACK=0

MODE_NAME="sim_w_torque_projection"
PROJECTION_ENV_VALUE="1"
EXPECTED_PROJECTION_RUNTIME="ON"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--iface IFACE] [--param PARAM_PATH] [--allow-non-loopback]

Options:
  --iface IFACE            Force network interface for DDS (default: lo)
  --param PATH             Param directory path relative to build/bin
                           (default: ../../state_machine/params/)
  --allow-non-loopback     Allow non-loopback DDS interface (unsafe for local-only sim)
  -h, --help               Show help

Environment:
  NETWORK_INTERFACE        Optional default interface if --iface is not provided
  CYCLONEDDS_URI           Optional CycloneDDS XML config. If not set, script enforces loopback-only config.
EOF
}

check_iface() {
    local iface="$1"
    if command -v ip >/dev/null 2>&1; then
        ip link show "$iface" >/dev/null 2>&1
    else
        [[ -d "/sys/class/net/$iface" ]]
    fi
}

default_cyclonedds_uri() {
    cat <<'EOF'
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="lo" priority="default" multicast="true"/>
      </Interfaces>
      <AllowMulticast>false</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
EOF
}

ensure_loopback_config() {
    if [[ -z "$CYCLONEDDS_URI_VALUE" ]]; then
        CYCLONEDDS_URI_VALUE="$(default_cyclonedds_uri)"
    fi

    if [[ "$ALLOW_NON_LOOPBACK" -eq 0 ]]; then
        if [[ "$NETWORK_INTERFACE" != "lo" ]]; then
            echo "[ERROR] Local sim mode requires --iface lo by default."
            echo "[HINT] Use --allow-non-loopback only if you know exactly why you need it."
            exit 1
        fi

        if [[ "$CYCLONEDDS_URI_VALUE" != *"name=\"lo\""* && "$CYCLONEDDS_URI_VALUE" != *"name='lo'"* ]]; then
            echo "[ERROR] CYCLONEDDS_URI does not contain loopback interface name=\"lo\"."
            echo "[HINT] Unset CYCLONEDDS_URI and rerun, or provide a loopback-only config."
            exit 1
        fi
    fi
}

find_latest_log_dir() {
    local log_root="$TARGET_DIR/logs"
    if [[ ! -d "$log_root" ]]; then
        return 0
    fi
    ls -1dt "$log_root"/* 2>/dev/null | head -n1 || true
}

count_pattern() {
    local pattern="$1"
    shift
    local total=0
    local file
    local c

    for file in "$@"; do
        if [[ -f "$file" ]]; then
            c=$(grep -cE "$pattern" "$file" 2>/dev/null || true)
            total=$((total + c))
        fi
    done

    echo "$total"
}

collect_summary() {
    local session_dir="$1"
    local run_exit_code="$2"

    local stdout_file="$session_dir/stdout.log"
    local main_log="$session_dir/main.log"
    local selfcheck_log="$session_dir/selfcheck.log"
    local events_log="$session_dir/events.log"
    local -a files=("$stdout_file" "$main_log" "$events_log")

    local projection_compile projection_runtime iface_value
    if [[ -f "$selfcheck_log" ]]; then
        projection_compile=$(grep -E '^projection_compile=' "$selfcheck_log" | tail -n1 | cut -d= -f2-)
        projection_runtime=$(grep -E '^projection_runtime=' "$selfcheck_log" | tail -n1 | cut -d= -f2-)
        iface_value=$(grep -E '^iface=' "$selfcheck_log" | tail -n1 | cut -d= -f2-)
    fi
    projection_compile="${projection_compile:-UNKNOWN}"
    projection_runtime="${projection_runtime:-UNKNOWN}"
    iface_value="${iface_value:-UNKNOWN}"

    local fsm_total fsm_ok fsm_rejected
    local stand_ok loco_ok dance_ok dance_end tau_clip_hits switcher_hits exit_hits mode_hits

    fsm_total=$(count_pattern "\\[FSM\\]" "${files[@]}")
    fsm_ok=$(count_pattern "\\[FSM\\].*result=OK" "${files[@]}")
    fsm_rejected=$(count_pattern "\\[FSM\\].*result=REJECTED" "${files[@]}")
    stand_ok=$(count_pattern "\\[FSM\\].*to=STAND.*result=OK" "${files[@]}")
    loco_ok=$(count_pattern "\\[FSM\\].*to=LOCO.*result=OK" "${files[@]}")
    dance_ok=$(count_pattern "\\[FSM\\].*to=DANCE.*result=OK" "${files[@]}")
    dance_end=$(count_pattern "\\[DANCE_END\\]" "${files[@]}")
    tau_clip_hits=$(count_pattern "\\[DANCE_TAU_CLIP\\]" "${files[@]}")
    switcher_hits=$(count_pattern "\\[MOTION_SWITCHER\\]" "${files[@]}")
    exit_hits=$(count_pattern "\\[EXIT\\]" "${files[@]}")
    mode_hits=$(count_pattern "\\[MODE\\]" "${files[@]}")

    local validation_result="PASS"
    local validation_reason="NONE"

    if [[ "$projection_runtime" != "$EXPECTED_PROJECTION_RUNTIME" ]]; then
        validation_result="FAIL"
        validation_reason="PROJECTION_RUNTIME_MISMATCH(expected=${EXPECTED_PROJECTION_RUNTIME},actual=${projection_runtime})"
    elif [[ ! -f "$main_log" ]]; then
        validation_result="FAIL"
        validation_reason="MAIN_LOG_MISSING"
    elif [[ ! -f "$selfcheck_log" ]]; then
        validation_result="FAIL"
        validation_reason="SELFCHECK_LOG_MISSING"
    fi

    {
        echo "mode_name=$MODE_NAME"
        echo "expected_projection_runtime=$EXPECTED_PROJECTION_RUNTIME"
        echo "run_exit_code=$run_exit_code"
        echo "projection_compile=$projection_compile"
        echo "projection_runtime=$projection_runtime"
        echo "iface=$iface_value"
        echo "fsm_total=$fsm_total"
        echo "fsm_ok=$fsm_ok"
        echo "fsm_rejected=$fsm_rejected"
        echo "stand_ok_count=$stand_ok"
        echo "loco_ok_count=$loco_ok"
        echo "dance_ok_count=$dance_ok"
        echo "dance_end_count=$dance_end"
        echo "tau_clip_count=$tau_clip_hits"
        echo "motion_switcher_occupied_count=$switcher_hits"
        echo "mode_switch_count=$mode_hits"
        echo "exit_event_count=$exit_hits"
        echo "validation_result=$validation_result"
        echo "validation_reason=$validation_reason"
        echo
        echo "last_60_key_lines:"
        grep -hE "\\[STARTUP\\]|\\[SELFCHECK\\]|\\[WAIT_R2\\]|--------------- Start ---------------|\\[FSM\\]|\\[MODE\\]|\\[EXIT\\]|\\[DANCE_TAU_CLIP\\]|\\[DANCE_END\\]|\\[MOTION_SWITCHER\\]" "${files[@]}" 2>/dev/null | tail -n 60 || true
    } > "$session_dir/summary.txt"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --iface)
                if [[ $# -lt 2 ]]; then
                    echo "[ERROR] --iface requires a value"
                    exit 1
                fi
                NETWORK_INTERFACE="$2"
                shift 2
                ;;
            --param)
                if [[ $# -lt 2 ]]; then
                    echo "[ERROR] --param requires a value"
                    exit 1
                fi
                PARAM_PATH="$2"
                shift 2
                ;;
            --allow-non-loopback)
                ALLOW_NON_LOOPBACK=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "[ERROR] Unknown argument: $1"
                usage
                exit 1
                ;;
        esac
    done
}

main() {
    parse_args "$@"
    ensure_loopback_config

    if [[ ! -d "$TARGET_DIR" ]]; then
        echo "[ERROR] Target directory not found: $TARGET_DIR"
        echo "[HINT] Run ./build_w_torque_projection.sh first"
        exit 1
    fi

    if [[ ! -x "$TARGET_DIR/$EXECUTABLE" ]]; then
        echo "[ERROR] Executable not found or not executable: $TARGET_DIR/$EXECUTABLE"
        echo "[HINT] Run ./build_w_torque_projection.sh first"
        exit 1
    fi

    if [[ ! -d "$TARGET_DIR/$PARAM_PATH" ]]; then
        echo "[ERROR] Param directory not found from build/bin: $PARAM_PATH"
        exit 1
    fi

    if [[ -n "$NETWORK_INTERFACE" ]] && ! check_iface "$NETWORK_INTERFACE"; then
        echo "[ERROR] Interface does not exist: $NETWORK_INTERFACE"
        echo "[INFO] Available interfaces:"
        ip -o link | awk -F': ' '{print $2}' | grep -v lo || true
        echo "lo"
        exit 1
    fi

    mkdir -p "$LOG_ARCHIVE_ROOT"

    local run_tag
    run_tag="$(date +%Y%m%d_%H%M%S)_${MODE_NAME}"
    local session_dir="$LOG_ARCHIVE_ROOT/$run_tag"
    mkdir -p "$session_dir"

    local -a run_cmd=("./$EXECUTABLE" "--param" "$PARAM_PATH" "--iface" "$NETWORK_INTERFACE")

    {
        printf '#!/usr/bin/env bash\n'
        printf 'cd %q\n' "$TARGET_DIR"
        printf 'export G1_ENABLE_DANCE_TORQUE_PROJECTION=%q\n' "$PROJECTION_ENV_VALUE"
        printf 'export CYCLONEDDS_URI=%q\n' "$CYCLONEDDS_URI_VALUE"
        printf 'export G1_LOG_DIR=%q\n' "$session_dir"
        printf '%q ' "${run_cmd[@]}"
        printf '\n'
    } > "$session_dir/cmd.sh"
    chmod +x "$session_dir/cmd.sh"

    {
        echo "mode=$MODE_NAME"
        echo "target_dir=$TARGET_DIR"
        echo "expected_projection_runtime=$EXPECTED_PROJECTION_RUNTIME"
        echo "network_interface=$NETWORK_INTERFACE"
        echo "param_path=$PARAM_PATH"
        echo "cyclonedds_uri=$(echo "$CYCLONEDDS_URI_VALUE" | tr '\n' ' ')"
    } > "$session_dir/meta.txt"

    echo "========================================"
    echo "Mode: $MODE_NAME"
    echo "Working directory: $TARGET_DIR"
    echo "Command: ${run_cmd[*]}"
    echo "Projection env: G1_ENABLE_DANCE_TORQUE_PROJECTION=$PROJECTION_ENV_VALUE"
    echo "DDS interface: $NETWORK_INTERFACE"
    echo "Session log directory: $session_dir"
    echo "========================================"

    local run_exit_code=130
    local _archived=0

    _finalize() {
        if [[ "$_archived" -eq 1 ]]; then
            return
        fi
        _archived=1
        if [[ -f "$session_dir/stdout.log" && ! -e "$session_dir/main.log" ]]; then
            cp -f "$session_dir/stdout.log" "$session_dir/main.log" 2>/dev/null || true
        fi
        collect_summary "$session_dir" "$run_exit_code" || true
        echo "[INFO] Summary: $session_dir/summary.txt"
        echo "[INFO] Stdout:  $session_dir/stdout.log"
        echo "[INFO] Command: $session_dir/cmd.sh"
    }
    trap _finalize EXIT INT TERM

    set +e
    (
        cd "$TARGET_DIR"
        G1_ENABLE_DANCE_TORQUE_PROJECTION="$PROJECTION_ENV_VALUE" \
        CYCLONEDDS_URI="$CYCLONEDDS_URI_VALUE" \
        G1_LOG_DIR="$session_dir" \
        DEPLOY_DISABLE_MOTION_SWITCHER="${DEPLOY_DISABLE_MOTION_SWITCHER:-1}" \
        DEPLOY_DISABLE_AUDIO_CLIENT="${DEPLOY_DISABLE_AUDIO_CLIENT:-1}" \
        "${run_cmd[@]}"
    ) 2>&1 | tee "$session_dir/stdout.log"
    run_exit_code=${PIPESTATUS[0]}
    set -e

    _finalize
    trap - EXIT INT TERM

    exit "$run_exit_code"
}

main "$@"
