# Revised MVP Method and Baseline Audit

Date: 2026-08-17  
Scope: re-audit after the representation, history, policy-shift, oracle-gate, and deployment-artifact revisions. No code was changed by this audit.

## Verdict

The synthetic benchmark and proposed-method smoke path are materially stronger. The full predeclared oracle-gate budget now runs, summaries are task-separated, policy shift changes executed trajectories, offline/online histories share one canonical execution-command contract, and deployable evaluation accepts only a student-only encoder artifact.

The main method conclusion is otherwise unchanged: B0 and the engineering oracle are the only end-to-end baselines. Explicit SysID, domain randomization, per-task RMA, multi-task RMA, and TAM-like adaptation remain model scaffolds without equal-budget training or evaluation. The project can now claim that its **synthetic benchmark passes the oracle-first viability gate**, but still cannot make the shared-representation or comparative-baseline claim.

All 60 tests pass under `PYTHONPATH=src pytest -q`. The expanded tests directly cover paired future commands, offline/online history equality, response-target sensitivity, goal-channel masking, student-only deployment loading, task-neutral probe commands, and changed policy-shift action traces.

## What is now fixed

1. **The planned oracle-gate budget is present.** `artifacts/oracle_gate.json` contains 1,680 rollouts: 2 tasks × 2 methods × 7 split/nominal groups × 3 seeds × 20 episodes. Every task/method/split summary therefore contains 60 episodes.

2. **Task summaries no longer merge Push and Lift.** `evaluate_manifest()` groups by `(task, method, split)`.

3. **Oracle recovery is no longer selected by the best OOD split.** The decision compares mean success over OOD-Mass, OOD-Friction, and OOD-Composition. Split-level results remain inspectable.

4. **The checked-in full-budget synthetic gate passes.** B0 nominal success is 1.0 for both tasks. Push B0 falls to 0.617 on OOD-Mass and 0.467 on OOD-Friction; its oracle reaches 1.0 and 0.917. Lift B0 falls to 0.383 on OOD-Mass and 0.0 on OOD-Composition; its oracle reaches 1.0 on both. Reported oracle force-violation and adapter-saturation rates are zero. These are synthetic engineering results only.

5. **Policy shift is behavioral rather than metadata-only.** `apply_behavior_style()` adds a deterministic held-out chirp, action-trace hashes are logged, and a test verifies distinct ID/policy-shift traces under identical physics.

6. **History now records the executed command.** The 7-D history contains the post-clipping 4-D action plus normalized stiffness, damping, and grip-force targets. The first adapted step no longer uses a fabricated action/history entry, and a test equates online and offline strict-past windows.

7. **Deployment artifact separation is enforced.** Encoder training emits a student-only checkpoint. Adapter training and OOD evaluation reject a full teacher/predictor training bundle. The checked student artifact contains only history-encoder weights and declares no privileged inputs.

8. **Indirect task cues were reduced.** The history encoder masks the rendered goal channel, and scripted probe commands are identical across Push and Lift. These fixes strengthen the method boundary but do not prove task invariance.

## Current executability and fidelity

| Method | Status after revision | Remaining decisive gap |
|---|---|---|
| B0 no adaptation | End-to-end synthetic baseline on paired manifests; full-budget artifact exists. | It is a hand-coded visual servo on a surrogate, not a learned robosuite BC/Diffusion policy. The evaluator still does not use `NoAdaptation` as a common baseline object. |
| Engineering oracle | End-to-end synthetic action-interface sanity check; full-budget gate passes. | It remains a manually designed formula, not learned B6 and not a formal upper bound. It does not use the same `PhysicalAdapter` or data budget. |
| B6 learned privileged oracle | Not implemented. | No learned same-adapter row using true context. |
| B1 domain-randomized policy | Forward-only module. | No task-policy training, task/goal input, checkpoint, runner, or data-efficiency curve; omitted from `data_efficiency.yaml`. |
| B2 explicit SysID | Forward-only 7-D-command-history estimator. | No physics-label training, held-out validation, calibration, same-adapter integration, or evaluation. Its 2-D output still does not match the configured 32-D adapter input. |
| B3 per-task RMA | Constructor-only module. | Still no `forward`, loss, optimizer, privileged-policy phase, on-policy history aggregation, checkpoint, or runner. Calling it still invokes `nn.Module`'s missing-forward error. |
| B4 multi-task RMA | Raw tensor-forward module. | No bounded action mapping, task-policy/privileged-context training, optimizer, checkpoint, or evaluation. It remains a shared contextual correction head rather than a trained multi-task RMA policy. |
| B5 TAM-like | Raw tensor-forward module. | Still a high-level action residual with no torque interface, torque history, ideal/perturbed inverse-dynamics target, training, or environment integration. |
| CoRMA | Absent and not required by the current MVP. | No semantic contact latent, deployable F/T history, supervised semantic head, force-regime InfoNCE, or task-family policy pipeline. |
| Shared encoder + task adapter | End-to-end synthetic smoke path with student-only deployment. | Only the response variant has a current-format smoke artifact. No equal-budget baseline curve or evidence that downstream corrections depend usefully on the latent exists. |

## Named-method fidelity

The original [RMA paper](https://arxiv.org/abs/2107.04034) jointly trains a privileged environment encoder and context-conditioned base policy with task reward, then trains an adaptation module to predict that task-relevant latent using iteratively collected on-policy state/action histories. The local RMA classes implement neither stage. A same-adapter encoder trained from scratch would be an important direct control, but must be labeled “RMA-style scratch context + frozen-policy adapter,” not canonical RMA.

The official [TAM paper](https://arxiv.org/abs/2606.06218) adapts after the low-level controller at the torque boundary using proprioceptive/applied-torque histories, an ideal-model physics-residual feature, and ideal-versus-perturbed inverse-dynamics torque targets. `TAMLike` changes a high-level 4-D action from a 3-D tracking residual. It is not a faithful TAM comparator and would give TAM different, weaker sensing, supervision, control authority, and frequency.

[CoRMA](https://arxiv.org/abs/2605.22082) predicts a simulator-only semantic contact context from deployable force/proprio/action history and adds force-regime contrastive regularization within related assembly tasks. It is not implemented here. The revised action-response representation should not be called CoRMA or contrastive RMA.

## Remaining fairness and method blockers

1. **No learned-baseline harness exists.** Only B0 and the oracle consume paired manifests. All applicable learned methods need the same physics rows, initial states, horizons, frozen-policy checkpoints, behavior-policy styles, controller bounds, and metrics.

2. **Supervision is not yet comparable or fully accounted.** TDPA adapters imitate privileged heuristic corrections on oracle-controlled rollouts. Canonical RMA uses task reward and privileged latent distillation; TAM uses inverse-dynamics torque labels; SysID uses explicit parameter labels. Report every source of privileged supervision and every task-specific simulator trajectory. Cheap same-objective proxies must be separated from faithful named baselines.

3. **Frozen-policy and jointly trained-policy comparisons answer different questions.** Implement both the direct frozen-policy scratch-encoder control and a faithfully labeled RMA comparator. Only the former isolates representation pretraining; only the latter answers the “why not RMA?” objection.

4. **Controller authority remains unequal.** TDPA uses a bounded correction dictionary and second clipping layer. Multi-task RMA emits an unbounded raw vector, SysID cannot enter the same adapter, and TAM-like clips a different interface. Same-interface rows need identical masks/bounds; a faithful torque TAM belongs in a separately declared control-authority comparison.

5. **The main adaptation-data experiment is still missing.** `adaptation_curve.py` produces only `pretrained_shared`. There are no scratch, SysID, per-task RMA, multi-task RMA, domain-randomized, or learned-oracle curves over the same nested 1/5/10/20/50/100% manifests.

6. **Adaptation-cost accounting is incomplete.** Baselines need task-specific/shared parameter counts, simulator trajectories, samples, optimizer steps, wall time, pretraining cost, and uncertainty. Current adapter checkpoints log samples and parameters but not steps or wall time.

7. **Adapter supervision still has closed-loop covariate shift.** Examples are gathered while the engineering oracle controls after the first step; evaluation is controlled by the learned adapter. Teacher-forced versus learned closed-loop histories and iterative rollout aggregation remain necessary.

8. **Disabled/inert output supervision remains.** Lift trains a stiffness target although `LiftEnv` does not consume stiffness. Push's disabled damping/grip entries remain in the seven-dimensional regression loss as fixed neutral values. The supervised loss should respect output masks, and every enabled output should affect the audited controller.

9. **Uncertainty is not reported.** The full gate has three seeds and sufficient raw rows, but reports only means. Confidence intervals or per-seed distributions are still required for result tables.

10. **The backend boundary is still a hard claim limit.** `backend="robosuite"` raises an error. The simplified renderer, dynamics, contact, and force equations are infrastructure tests, not MuJoCo manipulation evidence.

## Claims now supportable

The repository now supports the following narrow statements:

- the full-budget synthetic Push/Lift oracle-first gate passes on three seeds;
- B0 and the engineering oracle use identical per-task manifests, and a policy-shift split executes distinct trajectories under paired physics;
- the deployable learned path consumes only a student history encoder, deployable observations, and canonical executed-command history;
- response/distillation/hybrid training interfaces and a response-only end-to-end smoke run are executable.

It still does **not** support adaptation-data-efficiency, superiority/parity against SysID/RMA/multi-task RMA/TAM/CoRMA/domain randomization, learned-oracle performance, MuJoCo/robosuite robustness, or real-robot claims.
