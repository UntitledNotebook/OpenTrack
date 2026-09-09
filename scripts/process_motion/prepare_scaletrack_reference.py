#!/usr/bin/env python3
"""Build the constant-body reference consumed by OpenTrack ScaleTrack deploy."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.onnx_ml_pb2 import StringStringEntryProto
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


SELECTED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]

# Verified against package_motions output and named MuJoCo FK. IsaacLab stores
# articulation bodies in this breadth-first order, not MJCF depth-first order.
ISAACLAB_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]


def xml_body_names(path: Path) -> list[str]:
    root_body = ET.parse(path).getroot().find("worldbody/body")
    if root_body is None:
        raise ValueError(f"{path}: missing worldbody root body")
    result: list[str] = []

    def visit(body: ET.Element) -> None:
        name = body.get("name")
        if name is None:
            raise ValueError("all robot bodies must be named")
        result.append(name)
        for child in body.findall("body"):
            visit(child)

    visit(root_body)
    return result


def normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=1.0e-12)
    flat = quat.reshape(quat.shape[0], -1, 4)
    for frame in range(1, len(flat)):
        flip = np.sum(flat[frame - 1] * flat[frame], axis=-1) < 0.0
        flat[frame, flip] *= -1.0
    return flat.reshape(quat.shape)


def resample_bodies(
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    source_hz: float,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    if np.isclose(source_hz, target_hz):
        return positions.copy(), normalize_quaternions(quaternions_wxyz)
    if len(positions) < 2:
        raise ValueError("at least two frames are required for resampling")
    duration = (len(positions) - 1) / source_hz
    target_frames = int(np.floor(duration * target_hz + 1.0e-9)) + 1
    source_time = np.arange(len(positions), dtype=np.float64) / source_hz
    target_time = np.arange(target_frames, dtype=np.float64) / target_hz
    kind = "cubic" if len(positions) >= 4 else "linear"
    out_pos = interp1d(source_time, positions, kind=kind, axis=0)(target_time)

    source_quat = normalize_quaternions(quaternions_wxyz)
    out_quat = np.empty((target_frames, source_quat.shape[1], 4), dtype=np.float64)
    for body in range(source_quat.shape[1]):
        rotation = Rotation.from_quat(source_quat[:, body][:, [1, 2, 3, 0]])
        sampled = Slerp(source_time, rotation)(target_time).as_quat()
        out_quat[:, body] = sampled[:, [3, 0, 1, 2]]
    return out_pos, out_quat


def resample_qpos(qpos: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    if np.isclose(source_hz, target_hz):
        return qpos.copy()
    duration = (len(qpos) - 1) / source_hz
    target_frames = int(np.floor(duration * target_hz + 1.0e-9)) + 1
    source_time = np.arange(len(qpos), dtype=np.float64) / source_hz
    target_time = np.arange(target_frames, dtype=np.float64) / target_hz
    result = np.empty((target_frames, qpos.shape[1]), dtype=np.float64)
    other = np.r_[0:3, 7:qpos.shape[1]]
    kind = "cubic" if len(qpos) >= 4 else "linear"
    result[:, other] = interp1d(source_time, qpos[:, other], kind=kind, axis=0)(target_time)
    quat = normalize_quaternions(qpos[:, None, 3:7])[:, 0]
    sampled = Slerp(source_time, Rotation.from_quat(quat[:, [1, 2, 3, 0]]))(target_time).as_quat()
    result[:, 3:7] = sampled[:, [3, 0, 1, 2]]
    return result


def bodies_from_qpos(qpos: np.ndarray, model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos/model mismatch: {qpos.shape}, model nq={model.nq}")
    body_ids = np.asarray([model.body(name).id for name in SELECTED_BODY_NAMES])
    data = mujoco.MjData(model)
    positions = np.empty((len(qpos), len(body_ids), 3), dtype=np.float32)
    rotations = np.empty((len(qpos), len(body_ids), 4), dtype=np.float32)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        positions[frame] = data.xpos[body_ids]
        rotations[frame] = data.xquat[body_ids]
    return positions, rotations


def make_constant_onnx(positions: np.ndarray, rotations: np.ndarray, metadata: dict[str, str]) -> onnx.ModelProto:
    arrays = {
        "body_pos_w": np.ascontiguousarray(positions, dtype=np.float32),
        "body_quat_w": np.ascontiguousarray(rotations, dtype=np.float32),
    }
    nodes = []
    outputs = []
    for name, array in arrays.items():
        tensor = helper.make_tensor(name, TensorProto.FLOAT, array.shape, array.tobytes(), raw=True)
        nodes.append(helper.make_node("Constant", [], [name], name=f"Constant_{name}", value=tensor))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, array.shape))
    graph = helper.make_graph(nodes, "OpenTrackScaleTrackReference", [], outputs)
    model = helper.make_model(
        graph,
        producer_name="prepare_scaletrack_reference",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    for key, value in metadata.items():
        model.metadata_props.append(StringStringEntryProto(key=key, value=str(value)))
    onnx.checker.check_model(model)
    return model


def scalar_frequency(source: np.lib.npyio.NpzFile, override: float | None) -> float:
    if override is not None:
        return override
    for key in ("fps", "frequency"):
        if key in source:
            return float(np.asarray(source[key]).item())
    raise ValueError("NPZ has no fps/frequency; pass --source-frequency")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output scaletrack_ref.onnx")
    parser.add_argument("--source-frequency", type=float)
    parser.add_argument("--target-frequency", type=float, default=50.0)
    parser.add_argument(
        "--model",
        type=Path,
        default=repo / "deploy/sim_interface/mjcf/scene_mjx_flat_terrain.xml",
        help="OpenTrack G1-4010 MuJoCo scene, used when input contains qpos",
    )
    parser.add_argument(
        "--scaletrack-mjcf",
        type=Path,
        default=Path("/mnt/ScaleBFM/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml"),
        help="defines native ScaleTrack full-body array order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_frequency <= 0.0:
        raise ValueError("--target-frequency must be positive")
    with np.load(args.input, allow_pickle=False) as source:
        source_hz = scalar_frequency(source, args.source_frequency)
        if "body_pos_w" in source and "body_quat_w" in source:
            positions = np.asarray(source["body_pos_w"], dtype=np.float64)
            rotations = np.asarray(source["body_quat_w"], dtype=np.float64)
            if positions.ndim != 3 or positions.shape[-1] != 3:
                raise ValueError(f"invalid body_pos_w shape {positions.shape}")
            if rotations.shape != positions.shape[:-1] + (4,):
                raise ValueError(f"body pose shape mismatch: {positions.shape}, {rotations.shape}")
            if positions.shape[1] == len(SELECTED_BODY_NAMES):
                source_format = "scaletrack_selected_bodies"
            else:
                xml_names = xml_body_names(args.scaletrack_mjcf)
                if set(xml_names) != set(ISAACLAB_BODY_NAMES):
                    raise ValueError("ScaleTrack MJCF does not match the verified G1-29 body profile")
                if positions.shape[1] != len(ISAACLAB_BODY_NAMES):
                    raise ValueError(
                        f"native ScaleTrack body count {positions.shape[1]} does not match IsaacLab count {len(ISAACLAB_BODY_NAMES)}"
                    )
                indices = [ISAACLAB_BODY_NAMES.index(name) for name in SELECTED_BODY_NAMES]
                positions = positions[:, indices]
                rotations = rotations[:, indices]
                source_format = "scaletrack_full_bodies"
            positions, rotations = resample_bodies(
                positions, rotations, source_hz, args.target_frequency
            )
        elif "qpos" in source:
            qpos = np.asarray(source["qpos"], dtype=np.float64)
            if qpos.ndim != 2 or qpos.shape[1] != 36:
                raise ValueError(f"expected qpos [N,36], got {qpos.shape}")
            qpos = resample_qpos(qpos, source_hz, args.target_frequency)
            positions, rotations = bodies_from_qpos(qpos, args.model)
            source_format = "mujoco_qpos"
        else:
            raise ValueError("NPZ must contain either body_pos_w/body_quat_w or qpos")

    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError("reference contains NaN or Inf")
    rotations = normalize_quaternions(rotations).astype(np.float32)
    positions = positions.astype(np.float32)
    model = make_constant_onnx(
        positions,
        rotations,
        {
            "format": "opentrack_scaletrack_reference_v1",
            "source_file": str(args.input.resolve()),
            "source_format": source_format,
            "source_frequency_hz": source_hz,
            "target_frequency_hz": args.target_frequency,
            "frames": len(positions),
            "quaternion_order": "wxyz",
            "selected_body_names": json.dumps(SELECTED_BODY_NAMES),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    duration = (len(positions) - 1) / args.target_frequency
    print(
        f"saved {args.output}: {len(positions)} frames at {args.target_frequency:g} Hz "
        f"({duration:.3f} s endpoint duration), source={source_format}"
    )


if __name__ == "__main__":
    main()
