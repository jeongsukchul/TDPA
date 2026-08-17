# MVP Plan: Oracle-First Benchmark Viability Gate

```yaml
goal: >-
  Build the smallest runnable Push/Lift simulation slice that can decide whether
  hidden mass and friction shifts create a meaningful adaptation problem and
  whether the proposed bounded physical-correction interface can solve that
  problem when given perfect physics context. Stop before representation
  learning. This is a benchmark and controller viability gate, not evidence for
  the shared-representation hypothesis.

research_question: >-
  On one non-prehensile task (Push-to-Target) and one prehensile task
  (Grasp-Lift-Transport), do frozen nominal controllers succeed at nominal
  physics, fail measurably on held-out mass/friction shifts, and recover through
  bounded local physical corrections when those corrections receive true
  physics context? If not, context estimation cannot be the current bottleneck.

minimum_implementation:
  - >-
    Package skeleton with one documented install command, deterministic seeding,
    validated YAML configuration, and CLI entry points. Use robosuite/MuJoCo and
    PyTorch; do not add Hydra, W&B, robomimic, RGB-D models, or distributed
    training in this slice.
  - >-
    Two robosuite-compatible environments: a custom planar Push-to-Target task
    and a configured Lift/Transport task. The same XML assets, geom appearance,
    body IDs, and observation schema must be reused across all physics values.
  - >-
    A physics randomizer that sets and reads back object mass plus the relevant
    contact-pair friction: object-table for Push and gripper-object for Lift.
    Store mass/friction only in evaluator metadata, never deployment observations.
  - >-
    Frozen, deterministic nominal task controllers that do not receive randomized
    physics, split labels, object IDs, privileged state, or force/torque. They are
    benchmark-scaffolding policies only; later BC/Diffusion policies must satisfy
    the same FrozenTaskPolicy interface.
  - >-
    A bounded adapter action mapper for Cartesian residual and velocity scale on
    both tasks, plus bounded grip-force target on Lift. Stiffness and damping are
    deferred until a failure shows they are needed. Controller-side clipping is
    mandatory as a second boundary.
  - >-
    An oracle-context correction path that is isolated behind an explicit
    privileged/oracle flag and receives true mass/friction only. For the cheapest
    action-space viability test, use a small transparent physics-aware controller
    or fitted bounded adapter; label it an engineering oracle, not the final B6
    learned-oracle comparison.
  - >-
    An evaluation runner that executes identical seeded episode manifests for
    B0 (frozen nominal, no adaptation) and the engineering oracle, reports task
    and physical metrics by split, and exits nonzero when a gate fails.
  - >-
    Automated tests for physics readback, split separation, determinism, bounded
    actions, success/force metrics, privileged-key rejection, and equal evaluation
    manifests. Do not implement interaction datasets or encoders until this gate
    passes.

required_inputs:
  software:
    - Python 3.10 or 3.11
    - PyTorch
    - robosuite with a compatible MuJoCo version
    - NumPy, PyYAML, and pytest
  task_configuration:
    - workspace, target geometry, horizon, control frequency, and success tolerance
    - nominal object asset shared across every physics condition
    - controller and adapter bounds expressed in configuration
  physics_configuration:
    - >-
      A nominal point and disjoint train/ID/OOD-Mass/OOD-Friction/OOD-Composition
      manifests for each task. Start from multipliers relative to a documented
      nominal mass/friction, then save resolved MuJoCo values.
    - >-
      At least two independently varied values per physics axis and an unseen
      heavy-mass plus low-friction composition. OOD values must be outside the
      closed training interval on the shifted axis.
  evaluation_budget:
    - 3 fixed seeds for every reported gate result
    - at least 20 episodes per task, method, split, and seed for the first gate

expected_outputs:
  commands:
    - "python -m tdpa.tools.verify_physics --task push --config configs/physics/ood.yaml"
    - "python -m tdpa.tools.verify_physics --task lift --config configs/physics/ood.yaml"
    - "python -m tdpa.evaluation.oracle_gate --task all --seeds 0 1 2"
    - "pytest -q"
  artifacts:
    - >-
      A machine-readable physics readback table containing requested and actual
      MuJoCo mass/friction for every manifest row.
    - >-
      A JSON/CSV gate report with per-seed and aggregate success, final error,
      completion time, peak/RMS contact force, drop rate, slip rate, saturation
      rate, and episode-manifest hash.
    - >-
      A clear PASS/FAIL decision for nominal competence, OOD degradation, oracle
      recovery, safety bounds, and leakage checks.
  initial_gate_thresholds:
    nominal_competence: >-
      B0 success is at least 80% on nominal physics for each task.
    informative_shift: >-
      For each task, at least one intended OOD axis lowers B0 success by at least
      20 percentage points or produces a predeclared, practically meaningful
      degradation in continuous task error without relying on force alone.
    oracle_recovery: >-
      On the exact same OOD episode manifests, the oracle improves success by at
      least 10 percentage points or recovers at least half of the nominal-to-OOD
      continuous-error gap for each task, without increasing excessive-force or
      adapter-saturation violations.
    interpretation: >-
      These thresholds are internal go/no-go criteria, not paper effect-size
      claims. Preserve raw curves and uncertainty rather than reporting only the
      threshold result.

baselines_affected:
  implemented_now:
    - "B0: frozen nominal policy with no physical adaptation"
    - "Engineering oracle: perfect-context action-interface sanity check"
  interface_scaffolded_now:
    - "B6: learned adapter with privileged/oracle context"
    - "B2: explicit SysID feeding the same adapter input contract"
  intentionally_deferred_until_oracle_gate_passes:
    - "B1: domain-randomized task policy"
    - "B2 training/evaluation: explicit SysID"
    - "B3: per-task RMA"
    - "B4: multi-task RMA"
    - "B5: TAM-like adaptation"
    - "V1/V2/V3 representation variants"

tests_required:
  unit:
    - requested mass equals MuJoCo model readback within numerical tolerance
    - both geoms in the intended friction pair are verified after randomization
    - train, ID, and every shifted OOD manifest obey the declared interval rules
    - the same visual/object identity is not deterministically mapped to physics
    - deployment observation schemas reject mass, friction, force/torque, contact labels, simulator state, split, task/probe ID, and evaluator metadata
    - bounded mappings and controller clipping hold for extreme and non-finite adapter inputs
    - success, drop, slip, force, final-error, and saturation metrics pass synthetic traces with known answers
  integration:
    - reset with the same seed reproduces physics, initial state, and episode manifest
    - changing only mass leaves friction, visuals, target, and initial-state distribution unchanged
    - changing only friction leaves mass, visuals, target, and initial-state distribution unchanged
    - the B0 and oracle runners consume byte-identical episode manifests
    - disabling the explicit oracle flag makes access to privileged context raise an error
    - both tasks complete a short headless CPU smoke rollout
  acceptance:
    - all four gate decisions meet the predeclared thresholds on 3 seeds
    - raw requested/readback physics values and all failed episodes remain inspectable

failure_conditions:
  - requested and actual MuJoCo parameters disagree or the wrong geom/contact pair changes
  - train and OOD supports overlap on an axis declared held out
  - nominal B0 is below the competence gate, so the base task behavior is inadequate
  - B0 does not degrade under any realistic target shift, so adaptation is unnecessary
  - the oracle cannot improve both tasks, indicating an inadequate correction interface, controller, or nominal trajectory
  - apparent oracle gains require unbounded gains/residuals, unsafe force, direct goal access beyond the nominal policy contract, or OOD-specific tuning
  - metrics or initial-state manifests differ between methods
  - any privileged/evaluator field reaches the frozen policy or deployable adapter path
  - physics is encoded by object asset, body/material ID, color, filename, array order, or split metadata
  - results are not repeatable from clean configuration and fixed seeds
  - >-
    If any of the above occurs, stop and repair or pivot the benchmark; do not add
    an encoder, auxiliary loss, RGB-D backbone, or more tasks.

files_expected_to_change:
  - README.md
  - pyproject.toml
  - configs/env/push.yaml
  - configs/env/lift.yaml
  - configs/physics/train.yaml
  - configs/physics/ood.yaml
  - configs/adapter/push.yaml
  - configs/adapter/lift.yaml
  - src/tdpa/__init__.py
  - src/tdpa/config.py
  - src/tdpa/observations.py
  - src/tdpa/envs/make_env.py
  - src/tdpa/envs/push_env.py
  - src/tdpa/envs/lift_env.py
  - src/tdpa/envs/physics_randomization.py
  - src/tdpa/policies/frozen_nominal.py
  - src/tdpa/controllers/adapter_action_mapper.py
  - src/tdpa/controllers/oracle_context.py
  - src/tdpa/evaluation/metrics.py
  - src/tdpa/evaluation/oracle_gate.py
  - src/tdpa/tools/verify_physics.py
  - src/tdpa/utils/seed.py
  - tests/test_physics_randomization.py
  - tests/test_physics_splits.py
  - tests/test_deployment_observations.py
  - tests/test_adapter_bounds.py
  - tests/test_metrics.py
  - tests/test_deterministic_eval.py
  - tests/test_smoke_rollouts.py

research_essential:
  - two distinct interaction regimes, Push and Lift/Transport
  - actual MuJoCo parameter readback rather than trusting configuration values
  - frozen physics-blind nominal-policy contract
  - disjoint mass, friction, composition, and later policy-shift split semantics
  - B0 and perfect-context oracle evaluated on identical episode manifests
  - bounded correction interface and physical safety metrics
  - hard privileged/deployment separation and reproducible per-seed outputs

engineering_convenience:
  - Hydra/OmegaConf composition instead of validated plain YAML
  - Weights & Biases or TensorBoard
  - robomimic BC/Diffusion training in this first gate
  - RGB-D storage and neural visual encoders
  - impedance/stiffness/damping outputs before residual/scale/grip bounds are tested
  - cluster launchers, shell-script matrices, dashboards, and video rendering
  - Isaac Lab migration, real-robot interfaces, Task C, offline RL, and online RL

early_oracle_first_checks:
  - order: "physics readback -> nominal competence -> OOD failure -> oracle recovery -> leakage/safety audit"
  - >-
    Run a tiny deterministic sweep before long rollouts and inspect whether mass
    and friction changes produce response variation in the intended contact regime.
  - >-
    Tune benchmark ranges using B0 and physical plausibility only; never choose
    OOD values based on the proposed representation's performance.
  - >-
    Compare B0 and oracle using the same initial states, resolved physics, policy
    actions, horizons, success definitions, and force limits.
  - >-
    If perfect context cannot recover performance, classify the cause as nominal
    trajectory, action interface, controller, or required replanning before any
    representation work begins.

leakage_risks:
  - true mass/friction or resolved physics config copied into deployment observations
  - privileged object pose/velocity, contact state, force/torque, or simulator arrays exposed through a generic observation dictionary
  - object XML, body/geom/material ID, texture, color, filename, episode order, or seed deterministically identifying physics
  - OOD split labels or evaluation metadata passed through policy/adaptor kwargs
  - task/probe/policy identity becoming a shortcut when interaction pretraining is added
  - future response, future action, or post-contact samples entering a past-history window when sequence data is added
  - normalization statistics fitted using OOD/evaluation trajectories
  - oracle-only context remaining enabled in B0 or future deployable evaluation

claim_enabled_if_passed: >-
  The selected benchmark shifts are informative and a bounded local physical
  correction has a nontrivial perfect-context upper bound on both interaction
  regimes.

claims_not_enabled:
  - task-free pretraining learns reusable physics
  - a latent is task- or policy-invariant
  - pretraining reduces task-specific adaptation data
  - superiority to SysID, per-task RMA, multi-task RMA, or TAM
  - RGB-D is necessary

next_gated_task_if_passed: >-
  Implement learned B6 and explicit SysID with the same adapter/data budget, then
  per-task and multi-task RMA, before collecting the interaction dataset or adding
  response/distillation encoders, as required by the critic-gated development order.
```

The implementation should stop at the first failed gate and preserve the failure
artifact. A failed oracle gate is a useful project result: it localizes the problem
to benchmark design or control capacity before expensive representation work.
