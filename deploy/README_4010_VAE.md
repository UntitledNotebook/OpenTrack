# G1-4010 VAE deployment

This local extension deploys the VAE trained in Humanoid_Pipeline:

- experiment: `09011020_G1TrackingGeneralDR_vae_pulse_racket0_kl0.01_ar0.000_H1_E5_bizhang_4010_jumpreward_teacher1p501B_noisyobs_fromscratch_8gpu_env32768_lr5e-4_350k`
- checkpoint: `vae_action_expert_step_0000342000_best`
- OpenTrack policy name: `4010_vae_pulse_teacher1p501b`

The copied local assets are:

```text
storage/policy/4010_vae_pulse_teacher1p501b/checkpoints/
├── config.json
└── 000000342000/model.onnx
```

They are intentionally kept local and must not be uploaded.

## Controller interface

`FsmVaeController` uses the unified ONNX model. The interface is checked at
startup:

| Tensor | Shape | Meaning |
|---|---:|---|
| `state` | `[1, 93]` | measured actor observations |
| `auxiliary_state` | `[1, 58]` | clean reference-minus-measured joint errors |
| `noise` | `[1, 32]` | VAE reparameterization noise |
| `continuous_actions` | `[1, 29]` | posterior action |
| `continuous_actions_prior` | `[1, 29]` | prior action |

The exact state order is:

```text
gvec_pelvis[3]
gyro_pelvis[3] * 0.05
(joint_pos - default_qpos)[29]
joint_vel[29] * 0.05
last_motor_targets[29]
```

The auxiliary state is:

```text
reference_joint_pos - measured_joint_pos       [29]
(reference_joint_vel - measured_joint_vel)*0.05 [29]
```

For this checkpoint, the output is an **absolute joint-position target**.
It must not be added to the reference pose. The controller reads this from
`student_use_residual_action=false` and cross-checks it against
`output_residual_action=false`.

`last_motor_targets` is initialized from the measured pose. At runtime it is
updated from the command actually sent after torque projection and FSM
transition blending.

## Build

For real-robot preparation, retain torque projection:

```bash
cd /mnt/OpenTrack/deploy
./build_w_torque_projection.sh
```

Both projection-enabled and projection-disabled builds have been compiled
successfully with the VAE controller.

## Local simulation

Do not set the tracker registry at the same time as the VAE registry.

Terminal 1:

```bash
cd /mnt/OpenTrack/deploy
unset G1_TRACKER_POLICY G1_TRACKER_MOTIONS
export G1_VAE_POLICY=4010_vae_pulse_teacher1p501b
export G1_VAE_MOTIONS=rr_stand_still,rr_walk_slow,rr_two_foot_jump,rr_hurdle_jump
export G1_VAE_INFERENCE=posterior
export G1_VAE_NOISE_STD=0
./start_sim_w_torque_projection.sh --iface lo
```

Terminal 2:

```bash
cd /mnt/OpenTrack/deploy/sim_interface
export CYCLONEDDS_HOME=/mnt/cyclonedds-0.10/install
export LD_LIBRARY_PATH=/mnt/cyclonedds-0.10/install/lib
unset SIM_AUTO_START_ON_LOWCMD
/mnt/Humanoid_Pipeline/.venv/bin/python main.py --iface lo
```

Keep `SIM_AUTO_START_ON_LOWCMD` unset (or set it to `0`). If it is set to `1`,
the first lowcmd sent in DAMPING enables physics before the robot has entered
STAND, so the robot can fall before the VAE controller is selected.

Use the same state-machine sequence as the tracker deployment: R2, A, X,
simulation start, then a direction key. Direction slots follow the order in
`G1_VAE_MOTIONS`.

## Inference options

Safe default for reference tracking:

```bash
export G1_VAE_INFERENCE=posterior
export G1_VAE_NOISE_STD=0
```

Optional experimental settings:

```bash
# Sample the posterior:
export G1_VAE_INFERENCE=posterior
export G1_VAE_NOISE_STD=1

# Sample the state-only prior:
export G1_VAE_INFERENCE=prior
export G1_VAE_NOISE_STD=1
export G1_VAE_NOISE_SEED=0
```

The prior does not use the reference error to choose an action and should not
be used as a drop-in reference tracker. It is exposed for latent-policy
experiments. Do not add artificial sensor noise at deployment; the actor was
trained with noisy observations for robustness, while deployment uses the
available measured sensor values.

## Current validation status

The software path is working:

- unified ONNX input/output validation passes;
- posterior zero-noise and prior stochastic startup self-checks pass;
- projection-enabled and projection-disabled builds pass;
- state machine, DDS, reference freeze, action inference, torque projection,
  logging, automatic motion completion, and return to LOCO all execute;
- replaying a logged C++ input frame through Python ONNX Runtime reproduced
  the C++ action within about `0.009 rad` maximum difference.

An earlier headless result that appeared to show immediate falls was invalid:
that run explicitly set `SIM_AUTO_START_ON_LOWCMD=1`, which released the robot
while it was still in DAMPING, about 1.5 seconds before the STAND command. The
offline renderer consequently also aligned DANCE to the wrong time.

With the correct `DAMPING -> STAND -> LOCO -> SimStart -> DANCE` sequence, all
four reference rollouts completed and recovered upright:

| Reference | Post-blend root z min / max / final | Min up-z | Joint MAE |
|---|---:|---:|---:|
| `rr_stand_still` | 0.790 / 0.796 / 0.792 m | 1.000 | 0.065 rad |
| `rr_walk_slow` | 0.774 / 0.816 / 0.795 m | 0.981 | 0.099 rad |
| `rr_two_foot_jump` | 0.713 / 0.981 / 0.790 m | 0.889 | 0.130 rad |
| `rr_hurdle_jump` | 0.640 / 0.970 / 0.802 m | 0.627 | 0.105 rad |

This resolves the apparent discrepancy with Humanoid_Pipeline closed-loop
validation. The hurdle motion has the largest tilt/root-height excursion, so
dynamic motions still require the normal staged hardware bring-up and safety
support; the corrected simulation no longer indicates an immediate VAE fall.
