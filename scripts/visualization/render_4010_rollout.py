#!/usr/bin/env python3
"""Render an OpenTrack G1-4010 rollout and its reference to a side-by-side MP4."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

# Must be selected before importing MuJoCo on a server without a display.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mujoco
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

DPAD_KEYS = ("Up", "Down", "Left", "Right")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Render actual MuJoCo state and reference side by side."
    )
    parser.add_argument(
        "--sim-log-dir",
        type=Path,
        required=True,
        help="directory containing robot_entry.log and mujoco_qpos_trace.csv",
    )
    parser.add_argument("--dance-csv", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=repo_root / "deploy/sim_interface/mjcf/scene_mjx_flat_terrain.xml",
    )
    parser.add_argument("--blend-frames", type=int, default=50)
    parser.add_argument(
        "--dance-start-ms",
        type=float,
        default=None,
        help="absolute qpos-trace timestamp; normally inferred from SIM_AUTO_PRESS",
    )
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--panel-size", type=int, default=480)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-18.0)
    parser.add_argument("--distance", type=float, default=3.0)
    parser.add_argument(
        "--trim-blend",
        action="store_true",
        help="omit the initial controller transition from the video",
    )
    return parser.parse_args()


def read_dance_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"cycle", "t_ms"}
    required.update(f"jpos_obs_{index}" for index in range(29))
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"{path} is empty or missing columns: {sorted(missing)}")
    return rows


def read_reference(path: Path) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if "qpos" not in {output.name for output in session.get_outputs()}:
        raise ValueError(f"{path} does not contain a qpos output")
    qpos = np.asarray(session.run(["qpos"], {})[0], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"expected reference qpos [N,36], got {qpos.shape}")
    return qpos


def read_trace(path: Path) -> np.ndarray:
    trace = np.genfromtxt(path, delimiter=",", names=True)
    required = {
        "t_ms",
        "sim_enabled",
        "qpos_x",
        "qpos_y",
        "qpos_z",
        "qw",
        "qx",
        "qy",
        "qz",
    }
    if trace.dtype.names is None or not required.issubset(trace.dtype.names):
        raise ValueError(f"{path} does not have the expected qpos trace columns")
    return trace


def infer_dance_start_ms(sim_log_dir: Path, trace: np.ndarray) -> float:
    text = (sim_log_dir / "robot_entry.log").read_text()
    if "physics ENABLED (reason=lowcmd_fresh)" in text:
        raise ValueError(
            "physics was enabled by the first lowcmd, before the scheduled "
            "SimStart; this run cannot be aligned safely. Re-run with "
            "SIM_AUTO_START_ON_LOWCMD unset (or 0), or provide an explicitly "
            "verified --dance-start-ms"
        )
    match = re.search(r"auto-press scheduled: \[(.*?)\]", text)
    if match is None:
        raise ValueError(
            "cannot infer dance start from a manual run; provide --dance-start-ms"
        )
    schedule = {
        button: int(delay)
        for delay, button in re.findall(r"\((\d+), '([^']+)'\)", match.group(1))
    }
    dpad = [button for button in DPAD_KEYS if button in schedule]
    if len(dpad) != 1 or "SimStart" not in schedule:
        raise ValueError(f"unexpected auto-press schedule: {schedule}")

    enabled = np.flatnonzero(trace["sim_enabled"] > 0.5)
    if len(enabled) == 0:
        raise ValueError("physics was never enabled in qpos trace")
    physics_start_ms = float(trace["t_ms"][enabled[0]])
    return physics_start_ms + schedule[dpad[0]] - schedule["SimStart"]


def interpolate_actual_root(
    trace: np.ndarray, timestamps_ms: np.ndarray
) -> np.ndarray:
    if timestamps_ms[0] < trace["t_ms"][0] or timestamps_ms[-1] > trace["t_ms"][-1]:
        raise ValueError(
            "dance timestamps fall outside the qpos trace; check "
            "--dance-start-ms and whether the simulator ran long enough"
        )
    columns = ("qpos_x", "qpos_y", "qpos_z", "qw", "qx", "qy", "qz")
    root = np.column_stack(
        [
            np.interp(timestamps_ms, trace["t_ms"], trace[column])
            for column in columns
        ]
    )
    root[:, 3:7] /= np.linalg.norm(root[:, 3:7], axis=1, keepdims=True).clip(
        min=1.0e-12
    )
    return root


def make_qpos_sequences(
    rows: list[dict[str, str]],
    trace: np.ndarray,
    reference: np.ndarray,
    dance_start_ms: float,
    blend_frames: int,
    trim_blend: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cycles = np.asarray([int(row["cycle"]) for row in rows], dtype=np.int64)
    relative_ms = np.asarray([float(row["t_ms"]) for row in rows])
    actual_root = interpolate_actual_root(trace, dance_start_ms + relative_ms)
    actual_joints = np.asarray(
        [
            [float(row[f"jpos_obs_{joint}"]) for joint in range(29)]
            for row in rows
        ]
    )
    actual = np.concatenate((actual_root, actual_joints), axis=1)

    # Final deployment behavior holds reference frame zero during the blend.
    reference_indices = np.clip(cycles - blend_frames, 0, len(reference) - 1)
    aligned_reference = reference[reference_indices]
    keep = cycles >= blend_frames if trim_blend else np.ones(len(rows), dtype=bool)
    return (
        actual[keep],
        aligned_reference[keep],
        cycles[keep],
        reference_indices[keep],
    )


def render_pose(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    qpos: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (qpos[0], qpos[1], max(0.65, 0.75 * qpos[2]))
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    camera.distance = args.distance
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def annotate(
    actual_image: np.ndarray,
    reference_image: np.ndarray,
    cycle: int,
    reference_index: int,
    actual_z: float,
    reference_z: float,
    fps: float,
) -> np.ndarray:
    image = Image.fromarray(np.concatenate((actual_image, reference_image), axis=1))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=20)
    small_font = ImageFont.load_default(size=16)
    panel_width = image.width // 2

    draw.rectangle((0, 0, image.width, 42), fill=(0, 0, 0, 170))
    draw.text((16, 10), "ACTUAL ROLLOUT", fill=(80, 230, 120, 255), font=font)
    draw.text(
        (panel_width + 16, 10),
        "REFERENCE",
        fill=(100, 180, 255, 255),
        font=font,
    )
    footer = (
        f"t={cycle / fps:5.2f}s   cycle={cycle}   ref frame={reference_index}"
        f"   root z={actual_z:.3f}/{reference_z:.3f} m"
    )
    draw.rectangle(
        (0, image.height - 34, image.width, image.height), fill=(0, 0, 0, 170)
    )
    draw.text((14, image.height - 27), footer, fill="white", font=small_font)
    return np.asarray(image)


def main() -> None:
    args = parse_args()
    if args.blend_frames < 0 or args.fps <= 0 or args.panel_size <= 0:
        raise ValueError(
            "blend frames must be non-negative; fps and size must be positive"
        )

    rows = read_dance_csv(args.dance_csv)
    reference = read_reference(args.reference)
    trace = read_trace(args.sim_log_dir / "mujoco_qpos_trace.csv")
    dance_start_ms = (
        args.dance_start_ms
        if args.dance_start_ms is not None
        else infer_dance_start_ms(args.sim_log_dir, trace)
    )
    actual, aligned_reference, cycles, reference_indices = make_qpos_sequences(
        rows,
        trace,
        reference,
        dance_start_ms,
        args.blend_frames,
        args.trim_blend,
    )

    model = mujoco.MjModel.from_xml_path(str(args.model))
    if model.nq != 36:
        raise ValueError(f"expected a 36-qpos G1 model, got nq={model.nq}")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(
        model, height=args.panel_size, width=args.panel_size
    )
    camera = mujoco.MjvCamera()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.output),
        (args.panel_size * 2, args.panel_size),
        fps=args.fps,
        codec="libx264",
        quality=7,
        macro_block_size=16,
        ffmpeg_log_level="warning",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for actual_qpos, reference_qpos, cycle, reference_index in zip(
            actual, aligned_reference, cycles, reference_indices, strict=True
        ):
            actual_image = render_pose(
                renderer, model, data, camera, actual_qpos, args
            )
            reference_image = render_pose(
                renderer, model, data, camera, reference_qpos, args
            )
            writer.send(
                np.ascontiguousarray(
                    annotate(
                        actual_image,
                        reference_image,
                        int(cycle),
                        int(reference_index),
                        float(actual_qpos[2]),
                        float(reference_qpos[2]),
                        args.fps,
                    )
                )
            )
    finally:
        writer.close()
        renderer.close()

    duration = len(actual) / args.fps
    print(
        f"saved {args.output}: {len(actual)} frames, {duration:.2f} s, "
        f"{args.panel_size * 2}x{args.panel_size}"
    )


if __name__ == "__main__":
    main()
