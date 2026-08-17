# Task-Decoupled Physical Adaptation (TDPA)

This repository is a minimal, runnable research pipeline for testing whether a dynamics
representation learned from task-free action-response data reduces downstream physical
adaptation cost. It implements the interfaces, leakage guards, physics splits, representation
objectives, adapters, early baselines, diagnostics, and experiment accounting specified in
`Project_Pipeline.md`.

The default backend is a deterministic, low-cost manipulation surrogate. It makes the complete
pipeline and its scientific sanity checks runnable without MuJoCo, robosuite, demonstrations, or
a GPU. It is an infrastructure and hypothesis-debugging backend, not evidence for a robotics
claim. A separate robosuite / MuJoCo backend now validates real simulator construction, RGB-D,
contacts, indexed resets, live mass/friction mutation, and bounded OSC gain / Panda gripper-force
commands. The learned nominal-policy path is task-specific and separate from the synthetic policy.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .

python -m tdpa.envs.verify_physics --task push
python -m tdpa.data.interaction_collector --task push --episodes 24 --output artifacts/push.npz
python -m tdpa.data.interaction_collector --task lift --episodes 24 --output artifacts/lift.npz
python -m tdpa.training.train_encoder --variant response --datasets artifacts/push.npz artifacts/lift.npz --epochs 2 --output artifacts/response.pt
python -m tdpa.training.train_adapter --task push --encoder artifacts/response_student.pt --episodes 24 --epochs 2 --output artifacts/push_adapter.pt
python -m tdpa.evaluation.evaluate_ood --task push --encoder artifacts/response_student.pt --adapter artifacts/push_adapter.pt
```

Install the development extra and run the test suite with:

```bash
pip install -e ".[dev]"
pytest -q
```

The shell entry points in `scripts/` run the same stages. Outputs under `artifacts/` and `runs/`
are ignored by git.

## MuJoCo smoke gate (no training)

The checked-in Conda environment pins the tested robosuite / MuJoCo API pair. On this machine the
environment is already named `TDPA`; reproduce or update it, then run only the simulator gate:

```bash
conda env update -n TDPA -f environment.yml --prune
conda activate TDPA
export MUJOCO_GL=egl

python -m pytest -q
python -m pytest -q -m simulation tests/test_robosuite_backend.py
python -m tdpa.tools.smoke_robosuite \
  --task all --seed 7 --steps 10 --output artifacts/robosuite_smoke.json
python -m tdpa.tools.verify_physics \
  --backend robosuite --task push --seed 7 --count 2 \
  --output artifacts/robosuite_push_physics.json
python -m tdpa.tools.verify_physics \
  --backend robosuite --task lift --seed 7 --count 2 \
  --output artifacts/robosuite_lift_physics.json
```

Use `MUJOCO_GL=osmesa` only as the CPU-rendering fallback if EGL is unavailable. The smoke contact
probe uses privileged object placement solely to verify table and finger-pad contacts. It is not a
task policy, an OOD recovery result, or evidence about representation learning.

## Frozen nominal-policy gate

Run the implementation smoke before spending time on demonstrations or training. It collects six
small scripted episodes per task, checks causal action chunks, runs a BC forward pass, executes its
frozen wrapper in MuJoCo, and exercises low/high/reset controller readback. It performs no training.
Its checkpoints are marked `untrained_smoke`, rejected by the production loader, and ineligible for
result tables.

```bash
conda activate TDPA
export MUJOCO_GL=egl

python -m tdpa.tools.smoke_nominal_policy \
  --task all --seed 71 --output artifacts/nominal_policy_smoke.json
```

After that passes, these are the substantive commands to run. Collection uses privileged object and
target state only inside the scripted labeler. Model batches contain only causal RGB-D,
proprioception, and future Cartesian/gripper actions. The split is episode-disjoint, normalization
is fitted on training episodes only, and Push / Lift receive separate checkpoints.

```bash
# 1. Collect nominal demonstrations. Failed attempts remain auditable but are not trained on.
python -m tdpa.data.collect_nominal_demos \
  --task push --episodes 200 --seed 100 \
  --output artifacts/nominal/push_demos.hdf5
python -m tdpa.data.collect_nominal_demos \
  --task lift --episodes 200 --seed 200 \
  --output artifacts/nominal/lift_demos.hdf5

# 2. Train task-specific visual action-chunk BC policies.
python -m tdpa.training.train_nominal_policy \
  --task push --dataset artifacts/nominal/push_demos.hdf5 \
  --epochs 100 --device auto --output artifacts/nominal/push_bc.pt
python -m tdpa.training.train_nominal_policy \
  --task lift --dataset artifacts/nominal/lift_demos.hdf5 \
  --epochs 100 --device auto --output artifacts/nominal/lift_bc.pt

# 3. First require >=80% closed-loop nominal success on a locked 3x20 manifest.
python -m tdpa.evaluation.evaluate_nominal_policy \
  --mode competence --task push --checkpoint artifacts/nominal/push_bc.pt \
  --seeds 11 22 33 --episodes 20 --output artifacts/nominal/push_competence.json
python -m tdpa.evaluation.evaluate_nominal_policy \
  --mode competence --task lift --checkpoint artifacts/nominal/lift_bc.pt \
  --seeds 11 22 33 --episodes 20 --output artifacts/nominal/lift_competence.json

# 4. Only after competence passes, measure ID/OOD directionality on paired resets.
python -m tdpa.evaluation.evaluate_nominal_policy \
  --mode ood --task push --checkpoint artifacts/nominal/push_bc.pt \
  --competence-artifact artifacts/nominal/push_competence.json \
  --seeds 11 22 33 --episodes 20 --output artifacts/nominal/push_ood_gate.json
python -m tdpa.evaluation.evaluate_nominal_policy \
  --mode ood --task lift --checkpoint artifacts/nominal/lift_bc.pt \
  --competence-artifact artifacts/nominal/lift_competence.json \
  --seeds 11 22 33 --episodes 20 --output artifacts/nominal/lift_ood_gate.json
```

A low cloning loss is not policy competence. Do not begin adaptation or representation training
unless both task-specific nominal competence gates pass. The OOD output is descriptive: use its
separate low/high cells and paired intervals to decide whether there is a meaningful degradation to
recover. The gripper limit is a per-actuator simulator force cap, not a calibrated total
grasp/contact-force claim; the Push object-table force is likewise only a friction-interface
diagnostic.

## Perfect-context controller upper bound (no training)

After both competence gates pass and directional OOD degradation is established, run the
privileged oracle gate before implementing a learned adapter. It replays the exact hard-cell
seed/reset/physics rows from the OOD artifact twice: frozen B0 and B0 plus a fixed analytical
controller schedule. The schedule receives only the true object mass and friction, never the cell
name, and is clipped through the same action, velocity, stiffness, damping, and gripper-force
bounds intended for a learned adapter.

The default hard cells are Push high friction and Lift high mass, low friction, and composition.
The locked full gate requires the original passing competence artifact and 3-seed x 20-episode OOD
artifact. Run both tasks with:

```bash
conda activate TDPA
export MUJOCO_GL=egl
./scripts/evaluate_robosuite_oracle.sh
```

The Lift defaults use `lift_bc_spatial.pt`, `lift_competence_spatial.json`, and
`lift_ood_spatial.json`. Override any local name without editing the script, for example:

```bash
TDPA_LIFT_CHECKPOINT=artifacts/nominal/lift_bc.pt \
TDPA_LIFT_COMPETENCE=artifacts/nominal/lift_competence.json \
TDPA_LIFT_OOD=artifacts/nominal/lift_ood_gate.json \
./scripts/evaluate_robosuite_oracle.sh
```

A PASS means only that bounded perfect physics context can materially recover each selected
failure cell without excessive saturation or increased simulator force violations. It supports
moving next to learned SysID / RMA baselines. It does not validate deployable adaptation,
task-free representations, data efficiency, or physical safety. If the oracle fails a cell, fix
the control interface or nominal policy before representation training.

## Scientific boundaries

- Deployment models receive only RGB-D, proprioception, and action history.
- Physics values, object state, contact state, and force/torque are stored as privileged arrays
  and are rejected by the deployment-policy wrapper.
- Physics metadata is used only for dataset stratification, diagnostics, oracle adapters, and
  evaluation.
- ID and OOD supports are validated before sampling.
- The frozen-policy and frozen-pretrained-encoder assumptions are asserted in code.
- Results from the surrogate backend are smoke tests. Paper claims require robosuite/MuJoCo and,
  after passing the kill criteria, real-robot validation.

## Layout

Configuration lives in `configs/`; code lives in `src/tdpa/`; tests cover physics splits,
temporal alignment, leakage, model interfaces, bounded outputs, and deterministic evaluation.
Decision-oriented critic artifacts live in `reports/`.

The earliest decisive experiment is oracle-first: if an adapter given true mass and friction
cannot recover OOD task performance, representation learning cannot solve the bottleneck.

## Implementation status

The synthetic backend currently runs the complete smoke path:

1. validated ID/OOD physics manifests and parameter readback;
2. frozen visual-servo Push and Lift policies;
3. an engineering oracle-context gate on identical episode manifests;
4. balanced task-free interaction collection and causal sequence slicing;
5. response-only, privileged-distillation, and hybrid encoder training;
6. frozen-encoder supervised task-adapter training;
7. five-way OOD evaluation, representation probes, and adaptation-curve generation.

B0 and the engineering oracle are executable comparisons. Explicit SysID, domain-randomized,
per-task RMA, multi-task RMA, and TAM-like architectures are scaffolded but do not yet have
equal-budget training runners, so they are intentionally ineligible for a result table. Exact
fidelity gaps are recorded under `reports/baseline_audit/`.

Run a short end-to-end integration check after installation with:

```bash
./scripts/run_smoke_pipeline.sh
```

The critic-gated order and current supported/unsupported claims are preserved in `reports/`.
