# Motion Checkpoint List

The current bindings use `G1TrackingGeneralDR4010` for all five motions in group 0.
`B` cycles populated groups; with this configuration it stays in group 0.

| Slot | Button in `LOCO` | Simulation keyboard | Motion |
|---|---|---|---|
| 0 | Up | Up arrow | `rr_stand_still` |
| 1 | Down | Down arrow | `rr_walk_slow` |
| 2 | Left | Left arrow | `rr_two_foot_jump` |
| 3 | Right | Right arrow | `rr_hurdle_jump` |
| 4 | `L1` + Up | Hold `1`, press Up arrow | `vae_train_shun002_first6s` |

All other slots are empty. Select a motion from `LOCO`; after the clip finishes,
the controller returns to `LOCO` and runs `G1-Walk.onnx` again.

## Checkpoint and reference files

The policy is loaded from:

```
storage/policy/G1TrackingGeneralDR4010/checkpoints/<largest-numeric-directory>/policy.onnx
```

The currently installed latest checkpoint is `002001469440`. The observation
order comes from `storage/policy/G1TrackingGeneralDR4010/checkpoints/config.json`.
Each reference is loaded from `storage/data/<motion>/ref_data.onnx`.
The actual selected checkpoint and motion are recorded in `selfcheck.log` and
`events.log` under the launcher's session directory in `deploy_logs/`.

## Simulation sequence

1. Start the simulator and C++ deployment controller over `lo`.
2. Press `4` (`R2`) to start the control loop.
3. Press `a` (`A`) to enter `STAND`.
4. Press `x` (`X`) to enter `LOCO`.
5. Press `F5` to enable MuJoCo physics.
6. Press Up to run `rr_stand_still`, or select another motion from the table.

On the robot, the corresponding sequence is `R2`, `A`, `X`, then the motion
button. `A` returns to standing, `X` returns to locomotion, and `F1` exits the
process (`6` on the simulation keyboard).
