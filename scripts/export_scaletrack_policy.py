#!/usr/bin/env python3
"""Export a ScaleTrack humanoid Transformer checkpoint for OpenTrack.

The exported ONNX deliberately keeps ScaleTrack's deployment wrapper around the
actor: it builds proprioception, performs G1 forward kinematics, applies the
control-mode mask, and converts the normalized action into an absolute joint
position target.  OpenTrack therefore only has to maintain the short history
buffers and express the future reference bodies in the current robot base.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from torch import nn


JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Isaac/PhysX exposes the same named joints in articulation (breadth-first)
# order. The actor was trained in this order, while Unitree LowState/LowCmd
# and OpenTrack use JOINT_NAMES above.
ISAACLAB_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

SELECTED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]

MODE_NAMES = [
    "Pelvis-1", "UMI-2", "VR-3", "UMI-4", "VR-5",
    "UpperBody-6", "UpperBody-Mobile-7", "WholeBody-14",
]

MODE_BODY_NAMES = [
    ["pelvis"],
    ["left_wrist_yaw_link", "right_wrist_yaw_link"],
    ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"],
    ["left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"],
    ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"],
    [
        "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
    ],
    [
        "pelvis", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
    ],
    SELECTED_BODY_NAMES,
]

CONTEXT_LEN = 3
FUTURE_OFFSETS = [0, 1, 2, 3, 4, 5]
MODE_FEATURE_DIMS = [3, 3, 6, 6]


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    q = quat.reshape(-1, 4)
    v = vec.reshape(-1, 3)
    xyz = q[:, 1:]
    t = torch.cross(xyz, v, dim=-1) * 2.0
    return (v + q[:, :1] * t + torch.cross(xyz, t, dim=-1)).reshape(shape)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    q = quat.reshape(-1, 4)
    v = vec.reshape(-1, 3)
    xyz = q[:, 1:]
    t = torch.cross(xyz, v, dim=-1) * 2.0
    return (v - q[:, :1] * t + torch.cross(xyz, t, dim=-1)).reshape(shape)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_mul_inverse_right(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return quat_mul(q1, torch.cat((q2[..., :1], -q2[..., 1:]), dim=-1))


def parse_mjcf(xml_path: Path) -> tuple[list[str], list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root = ET.parse(xml_path).getroot()
    root_body = root.find("worldbody/body")
    if root_body is None:
        raise ValueError(f"{xml_path}: missing worldbody root body")

    body_names: list[str] = []
    joint_names: list[str] = []
    parent_indices: list[int] = []
    translations: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    axes: list[np.ndarray] = []

    def visit(node: ET.Element, parent: int) -> None:
        body_index = len(body_names)
        name = node.get("name")
        if name is None:
            raise ValueError("all bodies in the G1 kinematic tree must be named")
        body_names.append(name)
        parent_indices.append(parent)
        if body_index != 0:
            joints = node.findall("joint")
            if len(joints) != 1:
                raise ValueError(f"body {name!r} must contain exactly one joint")
            joint = joints[0]
            joint_name = joint.get("name")
            if joint_name is None:
                raise ValueError(f"body {name!r} contains an unnamed joint")
            joint_names.append(joint_name)
            translations.append(np.fromstring(node.get("pos", "0 0 0"), sep=" "))
            rotations.append(np.fromstring(node.get("quat", "1 0 0 0"), sep=" "))
            axes.append(np.fromstring(joint.get("axis", "0 0 1"), sep=" "))
        for child in node.findall("body"):
            visit(child, body_index)

    visit(root_body, -1)
    local_rotation = np.asarray(rotations, dtype=np.float32)
    local_rotation /= np.linalg.norm(local_rotation, axis=1, keepdims=True)
    return (
        body_names,
        joint_names,
        torch.tensor(parent_indices, dtype=torch.long),
        torch.tensor(np.asarray(axes), dtype=torch.float32),
        torch.tensor(np.asarray(translations), dtype=torch.float32),
        torch.tensor(local_rotation, dtype=torch.float32),
    )


def make_mode_table() -> torch.Tensor:
    table = torch.zeros(len(MODE_BODY_NAMES), len(SELECTED_BODY_NAMES), dtype=torch.float32)
    for mode_index, names in enumerate(MODE_BODY_NAMES):
        for name in names:
            table[mode_index, SELECTED_BODY_NAMES.index(name)] = 1.0
    return table


def make_mode_mappings(mode_table: torch.Tensor) -> torch.Tensor:
    pieces = []
    for feature_dim in MODE_FEATURE_DIMS:
        pieces.append(
            mode_table.unsqueeze(-1)
            .expand(-1, -1, feature_dim)
            .reshape(mode_table.shape[0], -1)
        )
    pieces.append(torch.ones(mode_table.shape[0], 1, dtype=torch.float32))
    return torch.cat(pieces, dim=-1)


def g1_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    armature_5020 = 0.003609725
    armature_7520_14 = 0.010177520
    armature_7520_22 = 0.025101925
    armature_4010 = 0.00425
    omega = 10.0 * 2.0 * math.pi
    damping_ratio = 2.0

    def gains(armature: float, multiplier: float = 1.0) -> tuple[float, float]:
        return multiplier * armature * omega**2, multiplier * 2.0 * damping_ratio * armature * omega

    kp_14, kd_14 = gains(armature_7520_14)
    kp_22, kd_22 = gains(armature_7520_22)
    kp_foot, kd_foot = gains(armature_5020, 2.0)
    kp_arm, kd_arm = gains(armature_5020)
    kp_wrist, kd_wrist = gains(armature_4010)

    default = np.zeros(29, dtype=np.float32)
    default[[0, 6]] = -0.312
    default[[3, 9]] = 0.669
    default[[4, 10]] = -0.363
    default[[15, 16, 22]] = 0.2
    default[23] = -0.2
    default[[18, 25]] = 0.6

    kp = np.array(
        [kp_14, kp_22, kp_14, kp_22, kp_foot, kp_foot] * 2
        + [100.0, 300.0, 300.0]
        + [kp_arm] * 5 + [kp_wrist] * 2
        + [kp_arm] * 5 + [kp_wrist] * 2,
        dtype=np.float32,
    )
    kd = np.array(
        [kd_14, kd_22, kd_14, kd_22, kd_foot, kd_foot] * 2
        + [2.0, 5.0, 5.0]
        + [kd_arm] * 5 + [kd_wrist] * 2
        + [kd_arm] * 5 + [kd_wrist] * 2,
        dtype=np.float32,
    )
    torque = np.array(
        [88.0, 139.0, 88.0, 139.0, 50.0, 50.0] * 2
        + [88.0, 50.0, 50.0]
        + [25.0] * 5 + [5.0] * 2
        + [25.0] * 5 + [5.0] * 2,
        dtype=np.float32,
    )
    action_scale = 0.25 * torque / kp
    action_scale[12:15] = 0.25
    return default, action_scale.astype(np.float32), kp, kd, torque


class ScaleTrackOnnxWrapper(nn.Module):
    def __init__(
        self,
        actor: nn.Module,
        task_embedder: nn.Module,
        mode_table: torch.Tensor,
        mode_mappings: torch.Tensor,
        default_dof_pos: torch.Tensor,
        action_scale: torch.Tensor,
        parent_indices: torch.Tensor,
        joint_axes: torch.Tensor,
        local_translation: torch.Tensor,
        local_rotation: torch.Tensor,
        selected_body_indices: torch.Tensor,
        lab_to_xml_joint_indices: torch.Tensor,
        input_to_policy_joint_indices: torch.Tensor,
        policy_to_output_joint_indices: torch.Tensor,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.task_embedder = task_embedder
        self.register_buffer("mode_table", mode_table)
        self.register_buffer("mode_mappings", mode_mappings)
        self.register_buffer("default_dof_pos", default_dof_pos)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("parent_indices", parent_indices)
        self.register_buffer("joint_axes", joint_axes)
        self.register_buffer("local_translation", local_translation)
        self.register_buffer("local_rotation", local_rotation)
        self.register_buffer("selected_body_indices", selected_body_indices)
        self.register_buffer("lab_to_xml_joint_indices", lab_to_xml_joint_indices)
        self.register_buffer("input_to_policy_joint_indices", input_to_policy_joint_indices)
        self.register_buffer("policy_to_output_joint_indices", policy_to_output_joint_indices)
        gravity = torch.zeros(1, CONTEXT_LEN, 3, dtype=torch.float32)
        gravity[..., 2] = -1.0
        self.register_buffer("gravity", gravity)
        tangent = torch.zeros(1, len(FUTURE_OFFSETS), len(SELECTED_BODY_NAMES), 3)
        normal = torch.zeros_like(tangent)
        tangent[..., 0] = 1.0
        normal[..., 2] = 1.0
        self.register_buffer("tangent", tangent)
        self.register_buffer("normal", normal)

    @staticmethod
    def _rope(x: torch.Tensor, inverse_frequency: torch.Tensor) -> torch.Tensor:
        """RoPE without the source module's ONNX-hostile slice assignment."""
        positions = torch.arange(x.shape[1], dtype=x.dtype, device=x.device)
        frequencies = torch.outer(positions, inverse_frequency.to(x.dtype))
        cosine = torch.cos(frequencies).unsqueeze(0)
        sine = torch.sin(frequencies).unsqueeze(0)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cosine - odd * sine
        rotated_odd = even * sine + odd * cosine
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    def _actor_forward(
        self,
        prop_obs: torch.Tensor,
        action_history: torch.Tensor,
        task_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Numerically identical actor path using only ORT-portable operations."""
        prop_tokens = self.actor.prop_projection(prop_obs)
        action_tokens = self.actor.action_projection(action_history)
        tokens = []
        for context_index in range(CONTEXT_LEN):
            tokens.append(prop_tokens[:, context_index : context_index + 1])
            if context_index + 1 < CONTEXT_LEN:
                tokens.append(action_tokens[:, context_index + 1 : context_index + 2])
        tokens.append(self.actor.empty_embedding.expand(prop_tokens.shape[0], -1, -1))
        x = torch.cat(tokens, dim=1)

        mask = torch.zeros(2 * CONTEXT_LEN, 2 * CONTEXT_LEN, dtype=torch.bool, device=x.device)
        mask[:-1, -1] = True
        for block in self.actor.transformer_blocks:
            normalized = block.rmsnorm1(x)
            rope = self._rope(normalized, block.rope.inv_freq)
            attention, _ = block.self_attention(rope, rope, normalized, attn_mask=mask)
            x = x + attention
            normalized = block.rmsnorm2(x)
            condition = block.cond_norm(task_tokens)
            cross_attention, _ = block.cross_attention(
                query=normalized, key=condition, value=condition
            )
            x = x + cross_attention
            x = x + block.feed_forward(block.rmsnorm3(x))
        x = self.actor.final_norm(x)
        return self.actor.projection_head(x[:, -1])

    def forward(
        self,
        root_quat_buffer: torch.Tensor,
        base_ang_vel_buffer: torch.Tensor,
        dof_pos_buffer: torch.Tensor,
        dof_vel_buffer: torch.Tensor,
        last_action_buffer: torch.Tensor,
        target_body_pos_future_to_robot_base: torch.Tensor,
        target_body_rot_future_to_robot_base: torch.Tensor,
        mode_index: torch.Tensor,
        time_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy_dof_pos_buffer = dof_pos_buffer[:, :, self.input_to_policy_joint_indices]
        policy_dof_vel_buffer = dof_vel_buffer[:, :, self.input_to_policy_joint_indices]
        projected_gravity = quat_apply_inverse(root_quat_buffer, self.gravity)
        prop_obs = torch.cat(
            (
                projected_gravity,
                base_ang_vel_buffer,
                policy_dof_pos_buffer - self.default_dof_pos,
                policy_dof_vel_buffer * 0.05,
            ),
            dim=-1,
        )

        dof_pos = policy_dof_pos_buffer[:, -1, :]
        half_angles = dof_pos[:, self.lab_to_xml_joint_indices].unsqueeze(-1) * 0.5
        joint_rot = torch.cat(
            (torch.cos(half_angles), self.joint_axes.unsqueeze(0) * torch.sin(half_angles)),
            dim=-1,
        )

        batch_size = dof_pos.shape[0]
        body_positions = [torch.zeros(batch_size, 3, dtype=dof_pos.dtype, device=dof_pos.device)]
        identity = torch.cat(
            (
                torch.ones(batch_size, 1, dtype=dof_pos.dtype, device=dof_pos.device),
                torch.zeros(batch_size, 3, dtype=dof_pos.dtype, device=dof_pos.device),
            ),
            dim=-1,
        )
        body_rotations = [identity]
        for joint_index, parent_index in enumerate(self.parent_indices[1:].tolist()):
            parent_pos = body_positions[parent_index]
            parent_rot = body_rotations[parent_index]
            position = parent_pos + quat_apply(
                parent_rot, self.local_translation[joint_index].unsqueeze(0).expand(batch_size, -1)
            )
            rotation = quat_mul(
                parent_rot,
                quat_mul(
                    self.local_rotation[joint_index].unsqueeze(0).expand(batch_size, -1),
                    joint_rot[:, joint_index],
                ),
            )
            body_positions.append(position)
            body_rotations.append(rotation)
        body_pos = torch.stack(body_positions, dim=1)[:, self.selected_body_indices]
        body_rot = torch.stack(body_rotations, dim=1)[:, self.selected_body_indices]

        target_pos_rel = target_body_pos_future_to_robot_base - body_pos[:, None]
        target_rot_tan_norm = torch.cat(
            (
                quat_apply(target_body_rot_future_to_robot_base, self.tangent),
                quat_apply(target_body_rot_future_to_robot_base, self.normal),
            ),
            dim=-1,
        )
        current_body_rot = body_rot[:, None].expand_as(target_body_rot_future_to_robot_base)
        target_rot_rel = quat_mul_inverse_right(
            target_body_rot_future_to_robot_base, current_body_rot
        )
        target_rot_rel_tan_norm = torch.cat(
            (quat_apply(target_rot_rel, self.tangent), quat_apply(target_rot_rel, self.normal)),
            dim=-1,
        )
        task_obs = torch.cat(
            (
                target_body_pos_future_to_robot_base.flatten(2, 3),
                target_pos_rel.flatten(2, 3),
                target_rot_tan_norm.flatten(2, 3),
                target_rot_rel_tan_norm.flatten(2, 3),
                time_offsets.to(target_body_pos_future_to_robot_base.dtype),
            ),
            dim=-1,
        )
        mapping = self.mode_mappings[mode_index]
        mode_vector = self.mode_table[mode_index]
        task_input = torch.cat(
            (
                task_obs * mapping.unsqueeze(1),
                mode_vector.unsqueeze(1).expand(-1, task_obs.shape[1], -1),
            ),
            dim=-1,
        )
        action = self._actor_forward(
            prop_obs, last_action_buffer, self.task_embedder(task_input)
        )
        policy_motor_target = action * self.action_scale + self.default_dof_pos
        return policy_motor_target[:, self.policy_to_output_joint_indices], action


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"checkpoint must be named model_<iteration>.pt: {path}") from error


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    candidates = list(path.glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no model_*.pt checkpoints found under {path}")
    return max(candidates, key=checkpoint_iteration).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="model_*.pt or run directory")
    parser.add_argument("--output-root", type=Path, required=True, help="OpenTrack storage/policy/<name>")
    parser.add_argument("--scaletrack-root", type=Path, default=Path("/mnt/ScaleBFM/ScaleTrack"))
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    iteration = checkpoint_iteration(checkpoint)
    scaletrack_root = args.scaletrack_root.resolve()
    network_source = scaletrack_root / "source/my_rsl_rl"
    mjcf_path = scaletrack_root / "source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml"
    if not network_source.is_dir() or not mjcf_path.is_file():
        raise FileNotFoundError(f"invalid ScaleTrack checkout: {scaletrack_root}")
    sys.path.insert(0, str(network_source))
    from my_rsl_rl.networks import HumanoidTransformer, TaskEmbedder

    # Checkpoints are local training artifacts, but still use PyTorch's safe
    # tensor-only loader.  Optimizer tensors are harmless and ignored below.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    actor_state = {key.removeprefix("actor."): value for key, value in state.items() if key.startswith("actor.")}
    task_state = {
        key.removeprefix("actor_task_embedder."): value
        for key, value in state.items()
        if key.startswith("actor_task_embedder.")
    }
    if not actor_state or not task_state:
        raise ValueError("checkpoint does not contain ScaleTrack actor weights")

    prop_dim = int(actor_state["prop_projection.weight"].shape[1])
    action_dim = int(actor_state["action_projection.weight"].shape[1])
    embedding_dim = int(actor_state["prop_projection.weight"].shape[0])
    output_dim = int(actor_state["projection_head.weight"].shape[0])
    task_dim = int(task_state["task_projection.weight"].shape[1])
    ff_dim = int(actor_state["transformer_blocks.0.feed_forward.w.weight"].shape[0])
    layer_ids = {
        int(key.split(".")[1])
        for key in actor_state
        if key.startswith("transformer_blocks.")
    }
    num_layers = max(layer_ids) + 1
    expected = (64, 29, 29, 267)
    actual = (prop_dim, action_dim, output_dim, task_dim)
    if actual != expected:
        raise ValueError(f"unsupported ScaleTrack policy dimensions {actual}; expected {expected}")
    if embedding_dim % args.num_heads != 0:
        raise ValueError("embedding dimension is not divisible by --num-heads")

    actor = HumanoidTransformer(
        prop_obs_dim=prop_dim,
        action_dim=action_dim,
        output_dim=output_dim,
        embed_dim=embedding_dim,
        num_heads=args.num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
    )
    task_embedder = TaskEmbedder(task_dim, embedding_dim, reduced_task_dim=None, hidden_dims=[])
    actor.load_state_dict(actor_state, strict=True)
    task_embedder.load_state_dict(task_state, strict=True)

    body_names, xml_joint_names, parents, axes, translations, rotations = parse_mjcf(mjcf_path)
    if set(xml_joint_names) != set(JOINT_NAMES) or len(xml_joint_names) != len(JOINT_NAMES):
        raise ValueError("ScaleTrack MJCF joint names do not match the G1-29 deployment profile")
    lab_to_xml = torch.tensor([ISAACLAB_JOINT_NAMES.index(name) for name in xml_joint_names], dtype=torch.long)
    input_to_policy = torch.tensor([JOINT_NAMES.index(name) for name in ISAACLAB_JOINT_NAMES], dtype=torch.long)
    policy_to_output = torch.tensor([ISAACLAB_JOINT_NAMES.index(name) for name in JOINT_NAMES], dtype=torch.long)
    selected_indices = torch.tensor([body_names.index(name) for name in SELECTED_BODY_NAMES], dtype=torch.long)
    mode_table = make_mode_table()
    default, action_scale, kp, kd, torque = g1_profile()
    policy_default = default[input_to_policy.numpy()]
    policy_action_scale = action_scale[input_to_policy.numpy()]

    wrapper = ScaleTrackOnnxWrapper(
        actor=actor,
        task_embedder=task_embedder,
        mode_table=mode_table,
        mode_mappings=make_mode_mappings(mode_table),
        default_dof_pos=torch.from_numpy(policy_default),
        action_scale=torch.from_numpy(policy_action_scale),
        parent_indices=parents,
        joint_axes=axes,
        local_translation=translations,
        local_rotation=rotations,
        selected_body_indices=selected_indices,
        lab_to_xml_joint_indices=lab_to_xml,
        input_to_policy_joint_indices=input_to_policy,
        policy_to_output_joint_indices=policy_to_output,
    ).eval()

    torch.manual_seed(42)
    root_quat = torch.zeros(1, CONTEXT_LEN, 4)
    root_quat[..., 0] = 1.0
    base_ang_vel = torch.zeros(1, CONTEXT_LEN, 3)
    dof_pos = torch.from_numpy(default).reshape(1, 1, 29).expand(1, CONTEXT_LEN, 29).clone()
    dof_vel = torch.zeros(1, CONTEXT_LEN, 29)
    action_history = torch.zeros(1, CONTEXT_LEN, 29)
    target_pos = torch.randn(1, len(FUTURE_OFFSETS), len(SELECTED_BODY_NAMES), 3) * 0.1
    target_rot = torch.zeros(1, len(FUTURE_OFFSETS), len(SELECTED_BODY_NAMES), 4)
    target_rot[..., 0] = 1.0
    mode_index = torch.tensor([7], dtype=torch.long)
    time_offsets = torch.tensor(FUTURE_OFFSETS, dtype=torch.long).reshape(1, -1, 1)
    example_inputs = (
        root_quat, base_ang_vel, dof_pos, dof_vel, action_history,
        target_pos, target_rot, mode_index, time_offsets,
    )

    output_dir = args.output_root.resolve() / "checkpoints" / str(iteration)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "policy.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            example_inputs,
            onnx_path,
            input_names=[
                "root_quat_buffer", "base_ang_vel_buffer", "dof_pos_buffer",
                "dof_vel_buffer", "last_action_buffer",
                "target_body_pos_future_to_robot_base",
                "target_body_rot_future_to_robot_base", "mode_index", "time_offsets",
            ],
            output_names=["motor_targets", "action"],
            opset_version=args.opset,
            do_constant_folding=True,
        )

    metadata = {
        "format": "opentrack_scaletrack_policy_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": iteration,
        "policy_architecture": {
            "embedding_dim": embedding_dim,
            "num_heads": args.num_heads,
            "ff_dim": ff_dim,
            "num_layers": num_layers,
        },
        "control_frequency_hz": 50.0,
        "history_buffer_size": CONTEXT_LEN,
        "future_idx": FUTURE_OFFSETS,
        "joint_names": JOINT_NAMES,
        "policy_joint_names": ISAACLAB_JOINT_NAMES,
        "action_names": ISAACLAB_JOINT_NAMES,
        "selected_body_names": SELECTED_BODY_NAMES,
        "mode_names": MODE_NAMES,
        "default_mode_index": 7,
        "mode_feature_dims": MODE_FEATURE_DIMS,
        "mode_mapping_with_time": True,
        "default_dof_pos": default.tolist(),
        "action_scale": action_scale.tolist(),
        "stiffness": kp.tolist(),
        "damping": kd.tolist(),
        "torque_limit": torque.tolist(),
        "reference_frame": "reference-root translation with initial yaw alignment",
        "quaternion_order": "wxyz",
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"checkpoint: {checkpoint}")
    print(f"policy:     {onnx_path}")
    print(f"metadata:   {metadata_path}")
    print(
        f"architecture: prop={prop_dim}, task={task_dim}, embed={embedding_dim}, "
        f"heads={args.num_heads}, ff={ff_dim}, layers={num_layers}"
    )


if __name__ == "__main__":
    main()
