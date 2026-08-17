# Experiment critic re-audit: MVP / synthetic oracle gate

## Decision

**ACCEPT_WITH_LIMITED_CLAIM for synthetic experiment plumbing; REVISE before any research
claim.** The revised oracle runner now supports the narrow statement that, in the deterministic
surrogate and with an intentionally expanded task-specific correction interface, a perfect-context
engineering oracle improves average OOD success on Push and Lift under paired manifests.

It still does not establish MuJoCo/robot benchmark viability, physical safety, representation
reuse, adaptation-data efficiency, or superiority to any learned baseline.

## Re-audit checks

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q -p no:cacheprovider`: **60 passed**.
- In-memory full gate: 3 seeds x 20 episodes per task/method/split, 1,680 rollouts, manifest hash
  `ad8141bdb5b0971f08fc476ec1ff776339208297d4e2a53dd5170cdedc186248`, **PASS**.
- Checked-in artifact now contains 20 episodes per seed/cell and task-separated summaries (60
  episodes per task/method/split), with both task gates passing
  (`artifacts/oracle_gate.json:34445-34462,34495-34518,34664-34672`).
- Average OOD success gain over Mass/Friction/Composition is +27.8 points for Push and +62.2
  points for Lift. Directional oracle success is lower on low friction (Push 75%, Lift 75%), so the
  aggregate gate must not be described as uniform robustness.
- All 60 paired ID/policy-shift rollouts per task and method had different execution-trace hashes.
- Capacity recount after the 7D execution-command repair: deployable student + task adapter 51,183
  parameters; per-task RMA scaffold 51,183; multi-task RMA scaffold 47,551; full pretraining bundle
  110,152. Counts remain diagnostic because no learned-baseline runner enforces them.

## Resolved since the first audit

| Item | Re-audit evidence | Verdict |
|---|---|---|
| Task-pooled reporting | Summary grouping now includes task (`src/tdpa/evaluation/oracle_gate.py:117-127`). | **Resolved.** |
| Label-only policy shift | `apply_behavior_style` changes the actual nominal command (`src/tdpa/policies/behavior.py:8-17`); both oracle and learned evaluators apply it and hash the post-clipping command trace (`src/tdpa/evaluation/oracle_gate.py:92-113`; `src/tdpa/evaluation/evaluate_task.py:47-82`); regression test verifies paired physics and unequal traces (`tests/test_deterministic_eval.py:28-34`). | **Resolved for the synthetic behavior-style shift.** Call it a held-out action style, not transfer to an independently trained policy checkpoint. |
| Evaluation budget/default | Scientific CLI default is 20 (`src/tdpa/evaluation/oracle_gate.py:171-180`), the full runner explicitly requests 20 (`scripts/run_all_evaluations.sh:4`), and the current artifact has the full budget. | **Resolved for the current artifact.** The smoke script still writes its 2-episode result to the same default path (`scripts/run_smoke_pipeline.sh:4`); give smoke and full artifacts distinct filenames to prevent regression. |
| Recovery cherry-picking | Recovery now compares mean success across the three OOD split aggregates (`src/tdpa/evaluation/oracle_gate.py:136-145`). | **Resolved at split-average level.** Low/high directions are still pooled; report both directions and paired uncertainty. |
| History/action contract | Histories now carry the 7D post-clipping execution command (`src/tdpa/controllers/adapter_action_mapper.py:16-28`; `configs/encoder/response.yaml:3-12`), online evaluation uses strict-past history (`src/tdpa/evaluation/evaluate_task.py:47-67`), and online/offline equality is tested (`tests/test_history_alignment.py:10-28`). Unpaired final actions are masked (`src/tdpa/data/sequence_dataset.py:213-230`). | **Resolved for proposed-method plumbing.** Equal history/rate/modalities across learned baselines remain untested because those runners do not exist. |
| Lift actuator/force bound | Lift force limit is 55 while learned grip bound is 60 (`configs/env/lift.yaml:14`; `configs/adapter/lift.yaml:17-22`). | **Partially resolved.** See remaining safety blocker below. |

## Remaining blockers and required repairs

| Severity | Finding and evidence | Required repair |
|---|---|---|
| **Critical** | **The physical oracle gate is still absent.** Task configs use the synthetic backend (`configs/env/push.yaml:2`, `configs/env/lift.yaml:2`), robosuite is rejected because assets are absent (`src/tdpa/envs/make_env.py:22-27`), and “readback” compares requested values with the same synthetic `env.physics` object (`src/tdpa/tools/verify_physics.py:18-33`). Hand equations define contact response (`src/tdpa/envs/push_env.py:22-52`; `src/tdpa/envs/lift_env.py:32-66`). | Implement audited robosuite/MuJoCo assets and resolved body/geom readback, including both geoms in each friction pair. Re-run the immutable 3 x 20 manifests with task-specific units/ranges. Preserve `SYNTHETIC_SMOKE_PASS` as a separate result. |
| **High** | **The expanded interface is intentional but not yet reconciled with the predeclared plan.** The plan says stiffness/damping are deferred (`reports/plans/mvp_plan.md:39-42,206-212`), while Push and Lift enable them (`configs/adapter/push.yaml:6-11`; `configs/adapter/lift.yaml:6-11`) and the engineering oracle uses them (`src/tdpa/controllers/oracle_context.py:20-36`). The prior controlled ablation found that removing deferred gains eliminated Push recovery; this is useful evidence for expanding the interface, not a reason to conceal the revision. | Update the plan, README, oracle audit, and artifact metadata to say the gate tests an **expanded residual/velocity/stiffness/damping/grip interface**, why it was expanded, and that Push recovery depends on stiffness. Preserve the minimal-interface failure and do not retrospectively present the expanded interface as predeclared. |
| **High** | **No fair learned-baseline or data-efficiency experiment exists.** The curve still trains only `pretrained_shared` (`src/tdpa/evaluation/adaptation_curve.py:23-66`); the method list is declarative (`configs/experiment/data_efficiency.yaml:4-10`). Baseline audits continue to mark SysID, per-task RMA, multi-task RMA, domain randomization, and TAM-like as missing budgets/runners/capacity matching (`reports/baseline_audit/explicit_sysid.md:12-18`; `reports/baseline_audit/per_task_rma.md:13-23`; `reports/baseline_audit/multitask_rma.md:13-22`). | Build one immutable-manifest runner with nested training subsets, equal observation/history contracts or explicit ablations, the same bounded action mapper, equal validation/tuning budgets, and logged total/task-specific capacity. Until then no comparative result or representation advantage is supported. |
| **High** | **Pretraining/task/inference cost accounting remains incomplete.** The curve explicitly excludes pretraining from its x-axis and provides no companion ledger (`src/tdpa/evaluation/adaptation_curve.py:62-66`). Encoder metadata omits trajectories/windows, optimizer steps, wall time, and parameter counts (`src/tdpa/training/train_encoder.py:132-161`); adapter metadata omits steps, time, tuning, inference, and privileged oracle-label collection cost (`src/tdpa/training/train_adapter.py:159-183`). | Report separate pretraining, task-adaptation, inference, and real-robot ledgers for every method: trajectories/transitions/windows, gradient steps, wall-clock/GPU-hours, trainable/total/task-specific parameters, tuning trials, latency, and privileged supervision. |
| **High** | **Lift's oracle safety check remains structurally unable to fail at the force boundary.** Although the learned grip bound is 60, the oracle pre-clips grip to exactly the 55 force limit (`src/tdpa/controllers/oracle_context.py:29-36`; `configs/env/lift.yaml:14`). Lift defines contact force as grip force and violation as strict `>` (`src/tdpa/envs/lift_env.py:44-55,71-82`). In the full re-run, five low-friction episodes reached peak force exactly 55 and all reported zero violation. Mapper saturation still ignores stiffness/damping/grip clipping (`src/tdpa/controllers/adapter_action_mapper.py:75-84`). | Do not pre-clip the oracle at the safety threshold; keep actuator clipping independent, record the requested pre-map command, count `>=` or a tolerance-aware threshold, and track saturation for every controller channel. Add known-trace metric tests and report any-step episode violation plus time above limit. |
| **High** | **Nominal competence still covers one fixed initial state and a hand marker servo.** Reset hard-codes initial robot/object state and does not use the RNG (`src/tdpa/envs/base.py:43-77`); nominal rows repeat the same trajectory. Registration checks one episode (`src/tdpa/training/train_base_policy.py:17-20`). The policy is not the planned BC/Diffusion policy (`src/tdpa/policies/frozen_nominal.py:36-89`). | Add seeded, logged object/robot/target/nuisance variation paired across methods and independent of physics. Validate a strong frozen BC/Diffusion policy on that distribution; give domain randomization the same task observations/action space. |
| **Medium** | **The manifest hash still omits resolved initial state, target, horizon/rates, controller/config hashes, and policy hash.** It hashes only task/split/seed/index/physics/policy label (`src/tdpa/evaluation/oracle_gate.py:27-35,74-76`). Action hashes help, but observations and nominal/applied actions are not separately preserved. | Materialize a versioned resolved manifest with state/target/assets/config/checkpoint hashes and log observation, nominal-action, and applied-command trace hashes per episode. Make every baseline consume it. |
| **Medium** | **Directional splits, uncertainty, and tuning separation remain incomplete.** Low/high supports are unions (`configs/physics/ood.yaml:1-6`) and are aggregated into one OOD-Mass/Friction number. The artifact provides raw rows and means but no per-seed intervals. There is no locked validation manifest or logged common hyperparameter-search budget. | Stratify and balance low/high directions, report paired per-seed confidence intervals, create an ID validation split disjoint by physics configuration/episode, lock the final OOD manifest, and log all tuning trials. |
| **Medium** | **Artifact provenance is incomplete.** The full artifact records runtime versions and paths, but `git_commit` is `unversioned` and physics entries are paths rather than content hashes (`artifacts/oracle_gate.json:34463-34493`). | Run from a versioned commit; record source/config hashes, CLI arguments, episode budget, host/backend version, and creation time. Use distinct full and smoke output paths. |

## Claim boundary

Supported now:

- deterministic, task-separated synthetic evaluation on equal physics manifests;
- actual held-out chirp action-style evaluation under paired physics;
- full-budget synthetic OOD degradation and recovery with an expanded engineering oracle;
- corrected strict-past/full-execution-command history plumbing for the proposed method.

Unsupported now:

- MuJoCo or real-robot benchmark viability and physical safety;
- uniform recovery across both low/high directions;
- any learned representation, adaptation-data, policy-invariance, or baseline-superiority claim;
- general compute, data, or deployment efficiency.

The next experiment remains a revised **MuJoCo oracle gate** using the explicitly disclosed expanded
interface. In parallel, the experiment infrastructure needs the fair learned-baseline runner and
four-part cost ledger before any encoder result is scientifically interpretable.
