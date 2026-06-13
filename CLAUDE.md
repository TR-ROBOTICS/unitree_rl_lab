# unitree_rl_lab — Project Context

## Environment
- Workstation: x86-64 Ubuntu, RDP from Windows
- Robot: Unitree G1 29DOF, aarch64
- Python: `conda run -n env_isaaclab python3` for all Isaac Lab calls
- Branch: `develop` (main working branch)

## Robot: G1 29DOF
- 29 joints, waist enabled (29dof mode = mode_machine=5)
- Deploy controller: `deploy/robots/g1_29dof/`
- Controller binary must be compiled **on robot** (aarch64)
- DDS topics: `rt/lowcmd`, `rt/lowstate`, `rt/wirelesscontroller`
- IDL: `unitree_hg` (G1), not `unitree_go` (Go2/B2)

### Joint conventions
- `left_shoulder_pitch_joint`: joint index 22 (0-based), negative = arm lifted forward
- `right_shoulder_pitch_joint`: joint index 29 (0-based), negative = arm lifted forward
- `joint_ids_map` in deploy.yaml maps sim→real order

### Controller button mapping (SDK names)
- LB = L1, RB = R1, LT = L2, RT = R2
- L2+Up → FixStand, R1+X → start RL policy

## Training
- Framework: Isaac Lab + RSL-RL
- Main env cfg: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py`
- Custom MDP: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/`
  - `rewards.py` — custom reward functions
  - `commands/arm_command.py` — UniformArmPoseCommand
  - `commands/__init__.py` — exports
- Train: `conda run -n env_isaaclab python3 scripts/rsl_rl/train.py --task=... --num_envs=8192`
- Export: `conda run -n env_isaaclab python3 scripts/rsl_rl/export.py --task=... --run_name=...`

## Deploy pipeline
1. Export `policy.onnx` → copy to `deploy/robots/g1_29dof/config/policy/velocity/v0/`
2. Edit `deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml`
   - `observations:` must match policy input exactly (names, sizes, history_length)
   - `commands:` ranges must match training
3. C++ obs handlers: `deploy/robots/g1_29dof/src/State_RLBase.cpp` (inside `namespace isaaclab`)
   - Built-in obs in `deploy/include/isaaclab/envs/mdp/observations/observations.h`
   - Custom obs use `REGISTER_OBSERVATION(name) { ... return std::vector<float>{...}; }`
4. Compile on robot: `cd build && cmake .. && make -j4`
   - CMakeLists auto-selects onnxruntime by arch (x64 or aarch64)
   - aarch64 onnxruntime needed: `deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0/`

## Key files
- `deploy/robots/g1_29dof/main.cpp` — entry point, mode_machine check
- `deploy/robots/g1_29dof/include/Types.h` — LowCmd/LowState type aliases
- `deploy/include/unitree_joystick_dsl.hpp` — button DSL docs
- `deploy/include/FSM/FSMState.h` — joystick update loop
- `deploy/include/isaaclab/assets/articulation/articulation.h` — `data.joystick` is `UnitreeJoystick*`

## Branches
- `develop` — clean, no arm work
- `feature/arm_movement_backup` — full arm lift training + deploy implementation

## Sim2sim
- MuJoCo bridge: `unitree_mujoco/simulate_python/`
- Virtual gamepad: `unitree_mujoco/simulate_python/virtual_gamepad.py`
  - Commands: `lu/ld` (left arm up/down), `ru/rd` (right arm up/down)

## Notifications
- ntfy.sh used for training complete alerts
