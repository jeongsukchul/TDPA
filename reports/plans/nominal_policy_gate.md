# Nominal robosuite policy gate

## Decision this sprint must enable

Build the code and data contracts needed to answer two questions later:

1. Can a frozen policy trained only on nominal-physics robosuite demonstrations reach at least
   80% success on held-out nominal Push and Lift resets?
2. Once nominal competence is established, do mass/friction shifts cause measurable degradation
   on exactly paired reset states?

This is a **coding and collection gate**. During this sprint, collect only small real-MuJoCo
demonstration fixtures and smoke the model/checkpoint/evaluator paths; do not run a substantive
training job. Do not train or load a physics encoder, physical adapter, RMA/SysID/TAM baseline,
or engineering oracle. The user may execute the full BC training and evaluation commands after
the implementation passes its smoke tests.

## Interfaces to lock first

Add `src/tdpa/policies/nominal_policy.py` with a backend-neutral deployment protocol:

```python
class NominalPolicy(Protocol):
    task: str
    frozen: bool

    def reset(self) -> None: ...
    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray: ...
```

`act()` must:

- accept exactly `{rgbd, proprio}` and reuse `assert_deployment_observation`;
- return one finite normalized action of shape `(4,)` in robosuite order
  `[dx, dy, dz, gripper]`;
- clip to the checkpointed action bounds;
- run under inference mode with every model parameter frozen;
- receive no object state, target coordinates, contact/grasp state, mass, friction, split,
  reset index, policy ID, or expert phase.

Define a model-level chunk contract so BC and a later diffusion policy share the same evaluator:

```python
class ActionChunkModel(nn.Module):
    def forward(
        self,
        rgbd_context,       # [B, context_horizon, 4, H, W]
        proprio_context,    # [B, context_horizon, 10]
        context_mask,       # [B, context_horizon]
    ) -> torch.Tensor:      # [B, prediction_horizon, 4]
        ...
```

The first BC model uses `context_horizon=1`, `prediction_horizon=1`, and
`execution_horizon=1`. A receding-horizon wrapper owns context/action queues and calls `reset()`
at every episode. A future diffusion checkpoint may use longer horizons without changing the
environment or evaluation code. Only define and test this compatibility layer now; do not
implement or train diffusion.

## Implementation plan

### 1. Add a privileged nominal demonstrator

Create `src/tdpa/policies/robosuite_expert.py` with separate finite-state experts for Push and
Lift. The expert is demonstration-only and receives the wrapped environment through an explicit
`PrivilegedDemonstrator` interface; it must never implement `NominalPolicy`.

- **Push:** open gripper, move above and behind the cube relative to the current target direction,
  descend to pushing height, advance through the target, then hold. Use live cube, target, and EEF
  positions only for label generation. Add waypoint tolerances and phase hysteresis so small reset
  jitter does not cause phase chatter.
- **Lift:** open, hover above the cube, descend, close for a fixed dwell, require robosuite's grasp
  check, lift vertically to a clearance height, translate above the target, descend/hold at the
  transport goal. Abort and log grasp-loss or timeout rather than relabeling it as success.
- Convert Cartesian waypoint error to normalized `OSC_POSITION` commands using the configured
  `position_delta_limit`; clip to the live four-dimensional `action_spec`. Do not pass the
  unsupported runtime controller dictionary.

The demonstrator must be reset between episodes. Expert phase, object/target state, grasp/contact
state, and failure reason are diagnostic metadata only.

### 2. Collect real nominal demonstrations with immutable manifests

Add `src/tdpa/data/nominal_demo_collector.py` and a `tdpa-collect-nominal` entry point. Each episode
must use:

```python
make_env(
    task,
    backend="robosuite",
    physics=Physics(nominal_mass, nominal_friction),
    seed=manifest_seed,
    episode_index=manifest_index,
)
```

Use disjoint, predeclared reset-index namespaces, for example train `0..N-1`, validation
`10000..`, and online competence test `20000..`. Never retry with a new index until a success
quota is filled. Record every attempted index, success/failure, horizon, reset state/fingerprint,
expert phase trace, and failure reason so expert attrition cannot be hidden. Fail the full
collector if scripted-expert success is below 95% on either task; failed episodes remain in the
audit manifest but are ineligible for BC training by an explicit flag.

For every transition, preserve the causal convention:

```text
observation[t] -> expert_action[t] -> environment step -> observation[t+1]
```

Write `tdpa-nominal-demo-v1` HDF5 archives, adding `h5py` to the `training` extra. Use chunked,
compressed arrays rather than holding all episodes in RAM:

- RGB `uint8 [T,H,W,3]` and depth `float16 [T,H,W,1]`, reconstructed as the existing float32
  CHW `rgbd` at load time;
- proprio `float32 [T,10]`;
- executed normalized action `float32 [T,4]`;
- valid mask, terminal flag, and episode length;
- privileged diagnostic group and JSON manifest stored separately from model-facing arrays.

The model-facing loader must return only RGB-D, proprioception, valid masks, and expert actions.
It must reject archives with non-nominal physics, duplicate reset indices across train/validation,
unknown deployment fields, inconsistent alignment, non-finite values, or an incompatible format
version. Save the archive SHA-256 and environment/config hashes after closing the file.

### 3. Add the minimal visual BC model

Create `src/tdpa/models/nominal_bc.py`:

- a small four-channel CNN with adaptive pooling for 64x64 RGB-D;
- a proprio MLP for the 10-D state;
- concatenated features followed by a two-layer action head;
- `tanh` output for all four normalized actions;
- no physics latent, task/split embedding, object-state head, reward input, or adapter output.

Use one task-specific model per task. The initial loss is masked motion MSE plus a separately
weighted gripper loss; log the two terms so the near-binary gripper channel cannot dominate or be
ignored. Fit proprio normalization on valid **training** transitions only. Keep RGB/depth in the
wrapper's declared ranges and actions in `[-1,1]`. Validation data may select early stopping but
must never fit normalization. OOD rollouts must not tune architecture or thresholds.

Add `src/tdpa/training/train_nominal_bc.py` and `tdpa-train-nominal-bc`. It may be implemented in
this sprint but must not be run beyond a tiny synthetic/unit fixture. Its real output is not valid
until trained by the user on the collected robosuite archives.

### 4. Make the checkpoint boundary structural

Add `src/tdpa/policies/nominal_checkpoint.py`. Save a tensor-only `.pt` payload plus a readable
`.manifest.json`. Required fields:

```text
format_version: tdpa-nominal-policy-v1
policy_family: bc
task: push | lift
backend: robosuite
frozen: true
observation_spec: rgbd shape/dtype/range, proprio shape/dtype
action_spec: order, shape, normalized bounds, OSC position scale, gripper convention
context_horizon / prediction_horizon / execution_horizon
model configuration and state_dict
training-only normalization statistics
train and validation archive paths + SHA-256
reset-index ranges and successful/attempted episode counts
optimizer steps, epochs, seed, wall time, parameter count
git commit, config hashes, Python/PyTorch/robosuite/MuJoCo versions
```

The loader must use `torch.load(..., weights_only=True)`, validate all fields and dimensions,
instantiate only a registered family, load strictly, call `eval()`, disable gradients, and reject
task/backend/config mismatches. The manifest records the final checkpoint SHA-256 after saving.
Do not serialize an optimizer, demonstrator, environment, Python callable, or privileged arrays
into the deployment artifact.

Retain `FrozenNominalPolicy` and `train_base_policy.py` for synthetic compatibility. New
robosuite evaluators must require a checkpoint path and must never silently fall back to the
synthetic marker servo.

### 5. Add a paired B0 evaluator

Create `src/tdpa/evaluation/nominal_policy_gate.py` and `tdpa-evaluate-nominal`. Do not reuse
`oracle_gate.run_episode`, which constructs the synthetic environment and marker policy.

The new immutable manifest must contain task, seed, episode index, reset fingerprint, requested
mass/friction, split and low/high direction, environment/config hash, policy checkpoint hash,
control frequency, horizon, and renderer. Generate physics with the existing index-addressable
randomizer, but materialize separate cells for:

- `nominal`;
- `id`;
- `ood_mass_low` and `ood_mass_high`;
- `ood_friction_low` and `ood_friction_high`;
- `ood_composition`.

Rows with the same `(task, seed, episode_index)` must resolve identical robot/object/target state
across physics cells. Call `policy.reset()` before each rollout, send `policy.act(observation)`
directly to `env.step(action)`, and close the environment in `finally`. No
`AdapterActionMapper`, oracle context, behavior chirp, physics encoder, or controller dictionary is
allowed.

Record success, final error, completion time, task-specific drop/overshoot fields, peak/RMS
contact force with a clearly versioned force definition, action saturation, observation/action
trace hashes, reset state/fingerprint, steps executed, and per-episode exception. Aggregate by
task, split direction, and seed, with paired bootstrap confidence intervals for differences from
nominal.

Run modes:

- `--mode smoke`: one episode/cell, validates checkpoint loading and finite rollout plumbing;
- `--mode competence`: nominal only, three seeds x 20 episodes, strict pass at success >= 0.80;
- `--mode ood`: all directional cells, three seeds x 20 episodes, descriptive B0 degradation
  only. Refuse OOD mode unless the same checkpoint has a saved passing competence artifact.

The current MuJoCo mass/friction ranges are explicitly uncalibrated. OOD output must retain this
warning and cannot be treated as a benchmark result until task-specific ranges and force units are
audited.

### 6. Configs, scripts, and tests

Add:

- `configs/policy/robosuite_bc_push.yaml` and `robosuite_bc_lift.yaml` for model, loss, optimizer,
  horizons, normalization, and seed;
- `configs/expert/robosuite_push.yaml` and `robosuite_lift.yaml` for waypoint/phase parameters;
- `scripts/collect_nominal_demos.sh`, `scripts/train_nominal_bc.sh`, and
  `scripts/evaluate_nominal_policy.sh`; scripts must use distinct smoke/full artifact paths;
- README commands and a statement that the existing synthetic oracle result is separate.

Tests:

- `tests/test_nominal_demo_dataset.py`: strict observation-action alignment, format/version/hash,
  disjoint reset indices, failed-episode exclusion, model-facing leakage rejection;
- `tests/test_nominal_bc.py`: shapes, finite bounded actions, mask behavior, deterministic inference,
  gradient freezing, context reset, and task/config mismatch rejection;
- `tests/test_nominal_checkpoint.py`: save/load round trip, strict fields, tensor-only payload,
  hash and normalization provenance;
- `tests/test_nominal_policy_eval.py`: identical reset fingerprints across physics cells, no
  privileged policy inputs, direct four-dimensional actions, checkpoint hash, directional
  aggregation, and competence prerequisite for OOD;
- `tests/test_robosuite_expert.py` marked `simulation`: at least three distinct nominal resets per
  task, >=95% expert success, valid HDF5 round trip, no unreported retries, and environment close;
- regression test proving the randomized live target controls reward, success, and final error
  consistently.

## Coding-smoke acceptance (no substantive training)

The implementation sprint passes when:

- all existing tests remain green;
- each expert completes at least three distinct nominal robosuite resets with >=95% success;
- two-episode-per-task smoke archives round-trip with exact causal alignment and no privileged
  model inputs;
- an untrained BC instance passes shape, checkpoint round-trip, freeze, clipping, and evaluator
  smoke tests, while being explicitly ineligible for competence/OOD results;
- the evaluator rejects synthetic policy JSON, wrong-task checkpoints, incomplete metadata,
  OOD-before-competence, non-finite actions, and controller/adaptation inputs;
- no real BC, diffusion, representation, adapter, or baseline optimization is run by Codex.

## Commands after implementation

### Coding smoke only

```bash
conda env update -n TDPA -f environment.yml --prune
conda activate TDPA
export MUJOCO_GL=egl

python -m pytest -q
python -m pytest -q -m simulation tests/test_robosuite_expert.py

python -m tdpa.data.nominal_demo_collector \
  --task push --split smoke --episodes 2 --seed 7 \
  --output artifacts/demos/push_nominal_smoke.hdf5
python -m tdpa.data.nominal_demo_collector \
  --task lift --split smoke --episodes 2 --seed 7 \
  --output artifacts/demos/lift_nominal_smoke.hdf5
```

These commands collect demonstrations but perform no gradient updates.

### Full commands for the user to run later

```bash
# Collect disjoint nominal train/validation datasets.
python -m tdpa.data.nominal_demo_collector \
  --task push --split train --episodes 200 --seed 101 \
  --output artifacts/demos/push_nominal_train.hdf5
python -m tdpa.data.nominal_demo_collector \
  --task push --split validation --episodes 40 --seed 101 \
  --output artifacts/demos/push_nominal_validation.hdf5
python -m tdpa.data.nominal_demo_collector \
  --task lift --split train --episodes 300 --seed 202 \
  --output artifacts/demos/lift_nominal_train.hdf5
python -m tdpa.data.nominal_demo_collector \
  --task lift --split validation --episodes 60 --seed 202 \
  --output artifacts/demos/lift_nominal_validation.hdf5

# Train only the task-specific nominal BC policies.
python -m tdpa.training.train_nominal_bc \
  --task push --train artifacts/demos/push_nominal_train.hdf5 \
  --validation artifacts/demos/push_nominal_validation.hdf5 \
  --config configs/policy/robosuite_bc_push.yaml \
  --output artifacts/policies/push_bc.pt
python -m tdpa.training.train_nominal_bc \
  --task lift --train artifacts/demos/lift_nominal_train.hdf5 \
  --validation artifacts/demos/lift_nominal_validation.hdf5 \
  --config configs/policy/robosuite_bc_lift.yaml \
  --output artifacts/policies/lift_bc.pt

# First require held-out nominal competence.
python -m tdpa.evaluation.nominal_policy_gate \
  --mode competence --task push --checkpoint artifacts/policies/push_bc.pt \
  --seeds 0 1 2 --episodes 20 --strict \
  --output artifacts/evaluation/push_nominal_competence.json
python -m tdpa.evaluation.nominal_policy_gate \
  --mode competence --task lift --checkpoint artifacts/policies/lift_bc.pt \
  --seeds 0 1 2 --episodes 20 --strict \
  --output artifacts/evaluation/lift_nominal_competence.json

# Only after both competence gates pass, measure frozen B0 degradation.
python -m tdpa.evaluation.nominal_policy_gate \
  --mode ood --task push --checkpoint artifacts/policies/push_bc.pt \
  --competence artifacts/evaluation/push_nominal_competence.json \
  --seeds 0 1 2 --episodes 20 \
  --output artifacts/evaluation/push_b0_ood.json
python -m tdpa.evaluation.nominal_policy_gate \
  --mode ood --task lift --checkpoint artifacts/policies/lift_bc.pt \
  --competence artifacts/evaluation/lift_nominal_competence.json \
  --seeds 0 1 2 --episodes 20 \
  --output artifacts/evaluation/lift_b0_ood.json
```

Do not proceed to the MuJoCo oracle gate until both nominal competence artifacts pass and the
directional OOD output shows a meaningful, reproducible degradation worth adapting to.

## Unsupported after this gate

- Diffusion policy implementation or any BC-versus-diffusion comparison;
- policy-shift claims involving an independently trained second policy;
- runtime stiffness, damping, or calibrated physical grip-force control;
- oracle recovery, learned privileged oracle, shared encoder, adapter, SysID, RMA, TAM-like, or
  adaptation-data-efficiency results;
- force-safety claims, calibrated MuJoCo OOD ranges, or paper-level performance claims;
- use of the privileged scripted expert as an evaluation baseline or deployment policy.
