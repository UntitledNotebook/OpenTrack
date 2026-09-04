#!/usr/bin/env python3
"""Convert a G1-4010 NPZ reference to OpenTrack's constant-output ONNX format.

Unlike the original batch converter, this tool reads the NPZ frequency,
resamples scalar-first MuJoCo quaternions correctly, and reconstructs qvel
with the same backward-difference convention used by tracker training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.onnx_ml_pb2 import StringStringEntryProto
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

FEET_ALL_SITES = ("left_foot", "right_foot", "left_foot_top", "right_foot_top")
DEFAULT_JOINT_POS = np.array(
    [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.3, 0.0, 1.28, 0.0, 0.0, 0.0,
        0.2, -0.3, 0.0, 1.28, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)


def _continuous_unit_quaternions_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat /= np.linalg.norm(quat, axis=1, keepdims=True).clip(min=1.0e-12)
    for i in range(1, len(quat)):
        if np.dot(quat[i - 1], quat[i]) < 0.0:
            quat[i] *= -1.0
    return quat


def resample_qpos(qpos: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    if np.isclose(source_hz, target_hz):
        return np.asarray(qpos, dtype=np.float64).copy()
    if len(qpos) < 2:
        raise ValueError("at least two frames are required for resampling")

    # Matches Humanoid_Pipeline interpolate_trajectories: preserve both
    # endpoints and use round(N * target/source) samples.
    output_frames = max(2, round(len(qpos) * target_hz / source_hz))
    source_index = np.arange(len(qpos), dtype=np.float64)
    target_index = np.linspace(0.0, len(qpos) - 1.0, output_frames, endpoint=True)

    result = np.empty((output_frames, qpos.shape[1]), dtype=np.float64)
    other_ids = np.r_[0:3, 7:qpos.shape[1]]
    kind = "cubic" if len(qpos) >= 4 else "linear"
    result[:, other_ids] = interp1d(
        source_index, qpos[:, other_ids], kind=kind, axis=0
    )(target_index)

    quat_wxyz = _continuous_unit_quaternions_wxyz(qpos[:, 3:7])
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    sampled_xyzw = Slerp(source_index, Rotation.from_quat(quat_xyzw))(
        target_index
    ).as_quat()
    result[:, 3:7] = sampled_xyzw[:, [3, 0, 1, 2]]
    return result


def add_default_joint_transitions(
    qpos: np.ndarray, target_hz: float, transition_seconds: float
) -> np.ndarray:
    transition_frames = round(transition_seconds * target_hz)
    if transition_frames <= 0:
        return qpos

    start = np.repeat(qpos[:1], transition_frames, axis=0)
    end = np.repeat(qpos[-1:], transition_frames, axis=0)
    start[:, 7:] = np.linspace(
        DEFAULT_JOINT_POS, qpos[0, 7:], transition_frames, endpoint=True
    )
    end[:, 7:] = np.linspace(
        qpos[-1, 7:], DEFAULT_JOINT_POS, transition_frames, endpoint=True
    )
    return np.concatenate((start, qpos, end), axis=0)


def reconstruct_qvel(qpos: np.ndarray, frequency: float) -> np.ndarray:
    qvel = np.zeros((len(qpos), qpos.shape[1] - 1), dtype=np.float64)
    qvel[1:, :3] = np.diff(qpos[:, :3], axis=0) * frequency
    qvel[1:, 6:] = np.diff(qpos[:, 7:], axis=0) * frequency

    q0 = _continuous_unit_quaternions_wxyz(qpos[:, 3:7])
    w0, x0, y0, z0 = q0[:-1].T
    w1, x1, y1, z1 = q0[1:].T
    # conj(q[t-1]) * q[t]: angular velocity in the previous local frame,
    # matching Humanoid_Pipeline's reference velocity reconstruction.
    rel = np.column_stack(
        (
            w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1,
            w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1,
            w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1,
            w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1,
        )
    )
    flip = rel[:, 0] < 0.0
    rel[flip] *= -1.0
    xyz_norm = np.linalg.norm(rel[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(xyz_norm, np.clip(rel[:, 0], 0.0, None))
    axis = rel[:, 1:] / xyz_norm[:, None].clip(min=1.0e-12)
    qvel[1:, 3:6] = axis * angle[:, None] * frequency
    return qvel


def reference_features(
    qpos: np.ndarray, qvel: np.ndarray, model_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if qpos.shape[1] != model.nq or qvel.shape[1] != model.nv:
        raise ValueError(
            f"trajectory/model mismatch: qpos={qpos.shape}, qvel={qvel.shape}, "
            f"model nq/nv={model.nq}/{model.nv}"
        )
    site_ids = np.array([model.site(name).id for name in FEET_ALL_SITES])
    if np.any(site_ids < 0):
        raise ValueError(f"missing one of the required foot sites: {FEET_ALL_SITES}")

    data = mujoco.MjData(model)
    feet_height = np.empty((len(qpos), 4), dtype=np.float32)
    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        feet_height[frame] = data.site_xpos[site_ids, 2]
    return feet_height, qpos[:, 2:3].astype(np.float32)


def make_constant_onnx(
    arrays: list[np.ndarray], names: list[str], metadata: dict[str, str]
) -> onnx.ModelProto:
    nodes = []
    outputs = []
    for array, name in zip(arrays, names, strict=True):
        array = np.ascontiguousarray(array, dtype=np.float32)
        tensor = helper.make_tensor(
            name=name,
            data_type=TensorProto.FLOAT,
            dims=array.shape,
            vals=array.tobytes(),
            raw=True,
        )
        nodes.append(
            helper.make_node(
                "Constant", inputs=[], outputs=[name], name=f"Constant_{name}", value=tensor
            )
        )
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, array.shape))

    graph = helper.make_graph(nodes, "OpenTrackReference", [], outputs)
    model = helper.make_model(
        graph, producer_name="prepare_4010_reference", opset_imports=[helper.make_opsetid("", 13)]
    )
    for key, value in metadata.items():
        model.metadata_props.append(StringStringEntryProto(key=key, value=str(value)))
    onnx.checker.check_model(model)
    return model


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="source NPZ")
    parser.add_argument("--output", type=Path, required=True, help="output ref_data.onnx")
    parser.add_argument("--target-frequency", type=float, default=50.0)
    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=0.0,
        help="optional default-joint transition on each side; candidates already wrapped should use 0",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=repo_root / "storage/assets/unitree_g1/scene_mjx_flat_terrain.xml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_frequency <= 0.0 or args.transition_seconds < 0.0:
        raise ValueError("frequencies must be positive and transition time non-negative")

    with np.load(args.input, allow_pickle=False) as source:
        if "qpos" not in source or "frequency" not in source:
            raise ValueError("NPZ must contain qpos and scalar frequency")
        qpos_source = np.asarray(source["qpos"], dtype=np.float64)
        source_hz = float(np.asarray(source["frequency"]).item())

    if qpos_source.ndim != 2 or qpos_source.shape[1] != 36:
        raise ValueError(f"expected qpos [N,36], got {qpos_source.shape}")
    if not np.isfinite(qpos_source).all():
        raise ValueError("qpos contains NaN or Inf")

    qpos = resample_qpos(qpos_source, source_hz, args.target_frequency)
    qpos = add_default_joint_transitions(
        qpos, args.target_frequency, args.transition_seconds
    )
    qvel = reconstruct_qvel(qpos, args.target_frequency)
    feet_height, root_height = reference_features(qpos, qvel, args.model)

    model = make_constant_onnx(
        [qpos, qvel, feet_height, root_height],
        ["qpos", "qvel", "feet_height", "root_height"],
        {
            "total_steps": len(qpos),
            "source_frequency_hz": source_hz,
            "target_frequency_hz": args.target_frequency,
            "transition_seconds_each_side": args.transition_seconds,
            "source_file": args.input.name,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    duration = (len(qpos) - 1) / args.target_frequency
    print(
        f"saved {args.output}: {len(qpos)} frames, {args.target_frequency:g} Hz, "
        f"endpoint duration {duration:.3f} s"
    )


if __name__ == "__main__":
    main()
