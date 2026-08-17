# MVP Data-Leakage and Temporal-Alignment Audit

Audit date: 2026-08-17  
Scope: current synthetic MVP, including collection, sequence slicing, V1/V2/V3 training, adapter training, learned/oracle evaluation, representation probes, configs, and tests.  
Decision: **FAIL / REVISE**

No direct path was found from true mass/friction, split metadata, privileged object state, contact labels, or F/T arrays into the proposed student encoder or physical adapter. However, four claim-blocking failures remain:

1. `policy_shift` evaluation does not shift the behavior policy.
2. partial future windows feed predictors actions that have no paired response, including the collector's final unexecuted action.
3. task identity is deterministically recoverable from the shared encoder's nominally deployable inputs.
4. stiffness/damping/grip controller commands affect responses but are absent from history and future-action tensors.

The 46 existing tests pass, but they do not cover these failures. OOD policy-shift or task-decoupled representation claims must not be made from this implementation.

## Risk-by-risk verdicts

### DL-1 — True mass/friction reaches the proposed deployment model: PASS

Evidence:

- Environment observations contain exactly `rgbd` and `proprio`; physics is held on the environment object, not returned in the observation (`src/tdpa/envs/base.py:43-46`, `src/tdpa/envs/base.py:86-102`).
- Collected physics values are nested in episode metadata, while privileged arrays occupy a separate namespace (`src/tdpa/data/interaction_collector.py:94-108`).
- `SequenceDataset.__getitem__` emits only RGB-D/proprio/action history plus future action/response training targets and masks; it does not emit metadata or privileged arrays (`src/tdpa/data/sequence_dataset.py:247-277`).
- The adapter receives only nominal action, student latent, and proprioception (`src/tdpa/models/physical_adapter.py:37-46`). In learned evaluation, manifest mass/friction constructs the environment but is never passed to the encoder or adapter (`src/tdpa/evaluation/evaluate_task.py:35-58`).
- Encoder pretraining rejects every archive episode whose metadata split is not exactly `train` (`src/tdpa/training/train_encoder.py:21-26`). Adapter data are sampled only from `PhysicsSplit.TRAIN` (`src/tdpa/training/train_adapter.py:76-85`).

Required fix: none for the current in-repository collector. Before accepting external archives, add schema/provenance validation so a producer cannot relabel mass/friction as columns inside `proprio` or `rgbd`; current validation checks shape and finiteness, not field semantics (`src/tdpa/data/sequence_dataset.py:33-48`, `src/tdpa/data/sequence_dataset.py:67-99`).

### DL-2 — Privileged object/contact/F-T channels reach the proposed deployment model: PASS, with artifact-boundary hardening required

Evidence:

- Object position/velocity, contact force, and contact state appear only in `privileged_observation`; response targets also include object velocity change and contact force (`src/tdpa/envs/base.py:92-112`).
- The training-only wrapper adds `privileged_history` after constructing a safe `SequenceDataset` item and uses the same past indices (`src/tdpa/training/datasets.py:9-30`).
- The teacher consumes `privileged_history`; the student consumes only RGB-D, proprioception, and action history (`src/tdpa/training/train_encoder.py:44-62`). Future responses are target labels under `torch.no_grad`, not student inputs (`src/tdpa/training/train_encoder.py:50-60`).
- Deployment history rejects any observation whose key set differs from exactly `{rgbd, proprio}` (`src/tdpa/data/history_buffer.py:20-25`). The frozen policy separately rejects privileged and unknown keys (`src/tdpa/policies/frozen_nominal.py:11-21`).

Required hardening: export/load a deployment-only student checkpoint. The current `EncoderBundle` contains teacher and response modules (`src/tdpa/models/bundle.py:15-50`), and learned evaluation loads the complete training bundle before selecting `student` (`src/tdpa/models/bundle.py:52-63`, `src/tdpa/evaluation/evaluate_task.py:40-54`). No raw privileged tensor currently reaches those modules, but a student-only artifact would make the boundary structural rather than call-site dependent.

### DL-3 — Split/probe/policy metadata or filenames directly enter the proposed model: PASS

Evidence:

- `task`, `probe_id`, `policy_id`, physics, split, seed, and episode index are stored only in metadata (`src/tdpa/data/interaction_collector.py:100-108`) and are absent from model-facing samples (`src/tdpa/data/sequence_dataset.py:267-277`).
- Dataset paths are used for loading, hashing, and checkpoint provenance, not as model features (`src/tdpa/training/train_encoder.py:21-40`, `src/tdpa/training/train_encoder.py:119-133`).
- Representation diagnostics recover metadata only after latents have been computed (`src/tdpa/evaluation/representation_probe.py:31-48`); probes do not update the encoder.
- `task_id` is an explicit input only in the intentionally task-conditioned multi-task RMA baseline (`src/tdpa/baselines/multitask_rma.py:40-52`), not in the proposed shared encoder or adapter.

Required fix: retain this separation and add a regression test covering every forbidden key in `PRIVILEGED_KEYS`, including task/probe/policy IDs (`src/tdpa/envs/base.py:8-22`); the current parametrized deployment test covers only six of them (`tests/test_deployment_observations.py:22-30`).

### DL-4 — Task identity is available through deployable proxy features: FAIL (high)

Evidence:

- The shared encoder receives the full RGB-D history when `use_rgbd: true` (`configs/encoder/response.yaml:11-13`; `src/tdpa/models/history_encoder.py:54-80`).
- RGB-D channel 2 explicitly renders the task target (`src/tdpa/envs/base.py:123-133`). Push uses target x=0.75 while Lift uses target x=0.45 (`configs/env/push.yaml:11`; `configs/env/lift.yaml:11-12`).
- Lift also changes the initial end-effector state from the base/Push reset, so initial proprioception is task-specific (`src/tdpa/envs/base.py:65-78`; `src/tdpa/envs/lift_env.py:23-27`).
- Probe actions branch on `task`, particularly the gripper command (`src/tdpa/data/interaction_collector.py:28-53`). Thus action history is also a task proxy.
- Read-only audit check: at identical default physics, Push and Lift initial proprioception were unequal; target-channel pixels were `[8,13]` for Push and `[8,11]` for Lift.
- The implemented diagnostics do not include a task-ID probe (`src/tdpa/evaluation/representation_probe.py:27-48`, `src/tdpa/evaluation/representation_probe.py:74-83`).

Required fixes:

1. Remove goal/target rendering from the physics encoder input (the task policy may retain it), or randomize/crop it independently of task.
2. Harmonize or deliberately randomize reset distributions and probe-controller conventions across interaction regimes.
3. Add held-out task-ID and task-from-latent probes, stratified by physics and probe action.
4. Until those controls pass, describe the encoder as shared across two known tasks, not task-decoupled/task-independent.

### DL-5 — Probe/action-style and policy-style shortcuts are adequately controlled: FAIL (medium)

Evidence:

- Physics is reused across the nine primitives, which is a useful anti-shortcut measure (`src/tdpa/data/interaction_collector.py:69-75`). Explicit probe IDs are not model inputs.
- Each episode nevertheless contains one deterministic primitive, and the latent receives its past action history (`src/tdpa/data/interaction_collector.py:28-53`; `src/tdpa/models/history_encoder.py:54-67`).
- Pretraining checks only global episode-count balance and then uses an ordinary shuffled `DataLoader`; it does not use probe-balanced batches or action-matched sampling (`src/tdpa/training/train_encoder.py:27-40`, `src/tdpa/training/train_encoder.py:92-99`).
- The probe/policy diagnostic is an in-sample nearest-centroid score: centroids and accuracy are computed on the same features (`src/tdpa/evaluation/representation_probe.py:60-64`). A single-policy archive also makes policy-ID accuracy non-diagnostic. Cross-policy retrieval and held-out action-response prediction are absent.

Required fixes: add a balanced sampler, action-matched batches/negatives or the declared unbalanced ablation, episode/physics-disjoint train/test probes, policy-ID evaluation containing both policies, cross-policy same-physics retrieval, and held-out-action response prediction. Do not use the present in-sample centroid scores as evidence of invariance.

### DL-6 — Object/body/material identity or episode filenames encode physics: PASS for the synthetic MVP; not audited for MuJoCo

Evidence:

- The synthetic renderer is explicitly independent of mass/friction and uses no object/body/material ID (`src/tdpa/envs/base.py:123-134`).
- Existing tests compare two physics settings at the same seed and confirm identical reset RGB-D and proprioception (`tests/test_deployment_observations.py:11-19`).
- Saved filenames are caller-selected and are not presented to models (`src/tdpa/data/interaction_collector.py:114-132`; `src/tdpa/training/train_encoder.py:21-40`).

Required fix: before enabling robosuite/MuJoCo, add an asset/body/geom/material-ID audit and tests showing that object XML, geom order, textures, filenames, and reset ordering do not determine mass/friction. This PASS cannot be generalized beyond the synthetic backend (`src/tdpa/envs/make_env.py:22-27`).

### DL-7 — Evaluation/OOD data leak through normalization: PASS for current identity normalization

Evidence:

- No training/evaluation path imports or applies `NormalizationStats`; current encoder and oracle checkpoints log `normalization_statistics: identity` (`src/tdpa/training/train_encoder.py:124-133`; `src/tdpa/evaluation/oracle_gate.py:180-189`).
- Therefore no OOD/evaluation trajectory is currently used to fit statistics.

Required fix before enabling normalization: fit on valid, unpadded TRAIN samples only; record dataset hashes and masks; serialize the frozen stats in the checkpoint; reject refitting at evaluation. The generic fitter currently has neither split provenance nor a padding mask (`src/tdpa/data/normalization.py:8-24`).

### TA-1 — Strict-past history excludes future action/response: PASS

Evidence:

- The declared convention is `action[i]` after observation `i`, with its earliest response at `i+1` (`src/tdpa/data/sequence_dataset.py:1-17`).
- For anchor `t`, history indices stop at `t-1`, future actions begin at `t`, and future responses begin at `t+response_offset`; offset must be positive (`src/tdpa/data/sequence_dataset.py:117-142`, `src/tdpa/data/sequence_dataset.py:213-229`).
- The synthetic impulse test verifies that action 3 and response 4 appear in future windows and neither enters history (`tests/test_sequence_dataset.py:63-79`).

Required fix: preserve this convention while repairing TA-2 through TA-4 below.

### TA-2 — Every predictor action in `U+` has its causal response in `R+`: FAIL (high)

Evidence:

- Partial-window mode intentionally permits future-action and future-response masks to differ. The existing test shows the final window has action mask `[True, True, False, False]` but response mask `[True, False, False, False]` (`tests/test_sequence_dataset.py:82-90`).
- `compute_loss` correctly masks the response target with the joint `future_mask`, but both predictors receive the broader `future_action_mask` (`src/tdpa/training/train_encoder.py:44-59`). The predictor can therefore condition on actions with no paired target response.
- The collector records `actions[step]` at every index but executes it only when `step + 1 < episode_length`; the final recorded action is never executed (`src/tdpa/data/interaction_collector.py:82-92`). With length 48 and future horizon 4, the last three anchor windows all expose this unexecuted final action to the predictor.

Required fixes:

1. Pass the joint action-response `future_mask` to both predictors, not `future_action_mask`.
2. Prefer constructing only paired `(action[i], response[i+1])` positions; do not store an unexecuted final action as valid.
3. Add a loss-level regression test that perturbs every unpaired/padded action and proves the loss/prediction is unchanged.

### TA-3 — Deployment warm-up matches training history semantics: FAIL (medium)

Evidence:

- Training's first anchor has one valid past pair, `(observation[0], action[0])`, with the remaining history positions padded (`src/tdpa/data/sequence_dataset.py:187-201`; `tests/test_sequence_dataset.py:52-60`).
- Adapter training and learned evaluation first append `(observation[0], zero_action)`, then after executing action 0 append `(observation[0], applied_action[0])` (`src/tdpa/training/train_adapter.py:89-110`; `src/tdpa/evaluation/evaluate_task.py:41-62`).
- Consequently, deployment at time 1 has two valid entries for the same sensor timestamp while the equivalent training anchor has one. The dummy persists until the fixed-length deque evicts it, so early latents see a mask/timestamp pattern absent from training.
- No test compares online `DeploymentHistory` tensors against an equivalent `SequenceDataset` window.

Required fix: define one bootstrap convention and make online/offline histories byte-equivalent at every warm-up timestep. Add an integration test that feeds a known rollout through both builders and compares values, masks, and source timestamps after every action.

### TA-4 — `u` includes every command/controller target that can change the response: FAIL (high)

Evidence:

- Dataset and model action dimensions are only four (`src/tdpa/data/interaction_collector.py:79`; `configs/encoder/response.yaml:5`).
- Applied execution also includes controller values. Stiffness, damping, and grip force remain in a separate controller dictionary (`src/tdpa/controllers/adapter_action_mapper.py:58-66`). Deployment history appends only `applied.action`, not `applied.controller` (`src/tdpa/evaluation/evaluate_task.py:59-62`; `src/tdpa/training/train_adapter.py:106-110`).
- These omitted commands causally alter the physical response: Push drive is proportional to stiffness (`src/tdpa/envs/push_env.py:22-41`); Lift tracking/grasp/slip depend on damping and grip force (`src/tdpa/envs/lift_env.py:32-56`).
- Interaction pretraining executes actions with default controller targets only (`src/tdpa/data/interaction_collector.py:90-92`), while downstream oracle/adapted rollouts change those targets.

Required fix: define `u` as the full post-clipping execution command, including active controller targets, and store it in both past history and `U+`. Expand the encoder/predictor action schema accordingly, or hold all omitted controller values fixed and remove them from the adapter. Add counterfactual tests where Cartesian action is fixed and one controller target changes; the command tensor and predicted response must change.

### PS-1 — Policy-shift evaluation actually uses a held-out behavior policy: FAIL (critical)

Evidence:

- ID and policy-shift manifests deliberately pair identical physics while assigning `probe_train` versus `chirp` IDs (`src/tdpa/envs/physics_randomization.py:281-289`, `src/tdpa/envs/physics_randomization.py:373-397`).
- Both oracle/no-adaptation evaluation and learned evaluation ignore `row.behavior_policy_id` and always instantiate the same `FrozenNominalPolicy` (`src/tdpa/evaluation/oracle_gate.py:78-96`; `src/tdpa/evaluation/evaluate_task.py:29-61`). The different ID survives only as result metadata in the oracle runner (`src/tdpa/evaluation/oracle_gate.py:101-107`).
- Read-only dynamic check on paired Push rows found byte-equivalent non-label results for `no_adaptation`, `oracle`, and `pretrained_shared`; learned ID and policy-shift metrics were exactly identical.
- The interaction collector itself does implement a real chirp action perturbation (`src/tdpa/data/interaction_collector.py:51-53`). A separate read-only check confirmed paired ID/policy-shift interaction episodes had equal physics but different actions and responses. That working collection path is not used by task evaluation.

Required fixes:

1. Select an actually distinct frozen behavior policy from `behavior_policy_id`, or evaluate the encoder on separately collected ID/chirp histories.
2. Assert paired rows have identical physics/reset state and measurably different action histories.
3. Add a regression test that fails if ID and policy-shift action trajectories are equal.
4. Remove `policy_shift` from reported task/OOD summaries until the shifted behavior is implemented.

## Tests and read-only checks performed

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q -p no:cacheprovider` — **46 passed**.
- Paired ID/policy-shift runtime comparison — same physics; different metadata IDs; all non-label outputs identical for no-adaptation, oracle, and learned adapted Push.
- Paired interaction-collector comparison — same physics; `probe_train` versus `chirp`; actions and responses differ, confirming the collector-side shift is real.
- Cross-task reset inspection — Push/Lift initial proprioception and target RGB channel differ deterministically.

## Merge gate

This audit should be considered passed only after TA-2, TA-3, TA-4, PS-1, and the deterministic task-proxy controls in DL-4 are repaired and covered by tests. Direct physics/privileged-key separation can remain marked PASS, but the full bundle should be replaced by a student-only deployment artifact before any external or real-robot deployment.

---

# Re-audit — 2026-08-17 (post-revision)

This section supersedes the initial decision for the revised code inspected on the same date.

New decision: **ACCEPT_WITH_LIMITED_CLAIM**

The claim-blocking temporal and deployment dataflow defects TA-2, TA-3, TA-4, PS-1, and DL-4 are repaired and covered by tests. The student-only artifact boundary is now structural in the adapter-training and learned-evaluation CLIs. DL-5's collection/sampling controls are materially improved, but its standard diagnostic runner does not yet supply the multi-task, multi-policy, held-out archives needed to produce meaningful task/policy-invariance evidence. The implementation may advance as a synthetic infrastructure pipeline, but it still does not support a claim that the learned latent is task- or policy-invariant.

## Re-audit verdicts

### TA-2 — Paired `U+ -> R+` semantics: PASS

Evidence:

- Future action validity now stops at `length - response_offset`, making every valid action index pair with a valid response index (`src/tdpa/data/sequence_dataset.py:213-231`). The final unexecuted action is no longer a valid future input.
- The collector writes the seven-dimensional execution command only inside the branch that actually executes the action; its last row remains zero (`src/tdpa/data/interaction_collector.py:85-94`).
- Both student and teacher predictors now receive the joint `future_mask`, exactly like the response target (`src/tdpa/training/train_encoder.py:52-67`).
- Tests assert equal future action/response masks, zero values outside the joint mask, and prediction invariance to arbitrary changes at masked positions (`tests/test_history_alignment.py:31-41`; `tests/test_encoder_shapes.py:45-56`; `tests/test_sequence_dataset.py:82-90`).

Residual limitation: the loss-level invariant is established through the predictor and dataset tests rather than by directly calling `compute_loss` with a perturbed batch. This is not a blocker because `compute_loss` passes the tested joint mask to the tested predictor path.

### TA-3 — Offline/online history parity: PASS

Evidence:

- Adapter training and learned evaluation no longer append a dummy reset entry. Step 0 executes without an encoder latent; after execution they append exactly `(previous_observation, applied_execution_command)` (`src/tdpa/training/train_adapter.py:83-113`; `src/tdpa/evaluation/evaluate_task.py:42-70`).
- At step 1 the online history therefore contains the same one strict-past pair as training anchor 1, without a duplicate timestamp.
- The new integration test compares RGB-D, proprioception, execution-command history, and mask byte-for-byte for every warm-up anchor (`tests/test_history_alignment.py:10-28`).

### TA-4 — Complete execution command: PASS for the current synthetic controller

Evidence:

- `AppliedAction.execution_command` contains the post-clipping 4-D action plus normalized stiffness, damping, and grip force (`src/tdpa/controllers/adapter_action_mapper.py:10-28`). Velocity scaling and Cartesian residual are already folded into the post-clipping action (`src/tdpa/controllers/adapter_action_mapper.py:65-79`).
- All three controller values are materialized even when a task holds an output at its nominal value, so the command schema is stable across Push and Lift (`src/tdpa/controllers/adapter_action_mapper.py:74-85`).
- Interaction archives store seven-dimensional commands, and all encoder variants declare `action_dim: 7` (`src/tdpa/data/interaction_collector.py:80-94`; `configs/encoder/response.yaml:3-6`; `configs/encoder/distill.yaml:3-6`; `configs/encoder/hybrid.yaml:3-6`).
- Adapter training and evaluation append `applied.execution_command`, not only the 4-D Cartesian/gripper action (`src/tdpa/training/train_adapter.py:109-112`; `src/tdpa/evaluation/evaluate_task.py:64-67`).
- A counterfactual unit test holds the 4-D action fixed, changes controller targets, and verifies the execution command changes (`tests/test_adapter_bounds.py:49-55`).

Residual limitation: the normalization denominators are hard-coded controller maxima (`300`, `60`, `60`) rather than checkpointed normalization statistics (`src/tdpa/controllers/adapter_action_mapper.py:17-28`). They match the synthetic environment's controller clipping, so this is not leakage, but a future real-controller interface must version these units/scales as part of the command schema.

### PS-1 — Real policy-shift behavior: PASS

Evidence:

- Evaluation now uses `row.behavior_policy_id` to apply a deterministic held-out chirp to the nominal action before mapping/execution (`src/tdpa/policies/behavior.py:8-17`; `src/tdpa/evaluation/oracle_gate.py:92-103`; `src/tdpa/evaluation/evaluate_task.py:47-67`). The ID is used by the external behavior generator and is not passed to the student encoder or adapter.
- Execution-trace hashes are recorded for oracle/no-adaptation and learned adapted rollouts (`src/tdpa/evaluation/oracle_gate.py:90-113`; `src/tdpa/evaluation/evaluate_task.py:72-82`).
- The regression test verifies identical paired physics and different ID/policy-shift trace hashes (`tests/test_deterministic_eval.py:28-34`).
- Targeted read-only runtime checks additionally confirmed distinct trace hashes for `no_adaptation`, `oracle`, and `pretrained_shared` on the same paired Push physics.

### DL-4 — Deterministic task proxies in shared-encoder inputs: PASS for the identified proxies

Evidence:

- Every encoder config enables `mask_goal_channel: true` (`configs/encoder/response.yaml:11-13`; `configs/encoder/distill.yaml:11-13`; `configs/encoder/hybrid.yaml:11-13`). `HistoryEncoder` clones RGB-D and zeros channel 2 before image encoding (`src/tdpa/models/history_encoder.py:70-83`). The task policy still receives the goal channel, as intended.
- Lift now retains the base reset state instead of changing end-effector position, harmonizing initial proprioception with Push (`src/tdpa/envs/base.py:65-78`; `src/tdpa/envs/lift_env.py:23-26`).
- Probe generation no longer branches on task (`src/tdpa/data/interaction_collector.py:30-55`). The collector maps each task through a stable seven-dimensional command schema (`src/tdpa/data/interaction_collector.py:69-94`).
- Tests prove goal-channel changes cannot alter the physics latent and paired Push/Lift probe command tensors are identical (`tests/test_deployment_observations.py:32-44`; `tests/test_interaction_collector.py:33-37`).
- Read-only runtime checks confirmed identical initial proprioception and identical initial RGB-D after goal-channel zeroing across Push and Lift.

Interpretation: physical responses and contact modes can still reveal the interaction regime; that is not equivalent to an explicit task-ID leak. Whether the latent unnecessarily encodes task/regime remains an empirical diagnostic question covered by the limited DL-5 verdict below.

### DL-5A — Probe/action-style shortcut controls in collection/training: PASS

Evidence:

- Multiple probe primitives reuse each physics sample (`src/tdpa/data/interaction_collector.py:72-77`), probe commands are task-agnostic (`src/tdpa/data/interaction_collector.py:30-55`), and future execution commands remain explicit predictor inputs.
- Pretraining validates approximately balanced episode counts, maps every dataset window back to its probe, and uses inverse-frequency `WeightedRandomSampler` weights by default (`src/tdpa/training/train_encoder.py:21-49`, `src/tdpa/training/train_encoder.py:97-106`).
- The unbalanced setting is an explicit anti-shortcut ablation and is logged in checkpoint metadata (`src/tdpa/training/train_encoder.py:137-161`, `src/tdpa/training/train_encoder.py:166-181`).

Residual limitation: inverse-frequency sampling balances the expected sample distribution, not the exact composition of every minibatch. Also, callers can request an episode count not divisible by nine, leaving the final physics sample with only a subset of primitives. Main shell collection uses 90 episodes, which is divisible by nine (`scripts/collect_interactions.sh:4-5`). Neither limitation is a present leakage path, but exact stratified batches and full primitive groups would strengthen the anti-shortcut guarantee.

### DL-5B — Task/policy diagnostics support an invariance claim: FAIL (evidence gate only)

Implemented evidence:

- Diagnostics now include task ID, policy ID, physics keys, same-physics cross-policy retrieval, shuffled-latent response loss, and a counterfactual future-action effect (`src/tdpa/evaluation/representation_probe.py:17-59`; `src/tdpa/evaluation/representation_probe.py:85-159`; `src/tdpa/evaluation/representation_probe.py:169-184`).
- Single-class task/policy probes return `None` instead of a misleading perfect accuracy (`src/tdpa/evaluation/representation_probe.py:76-82`).

Remaining failure:

- Nearest-centroid task/probe/policy scores still fit and evaluate centroids on the same latent samples (`src/tdpa/evaluation/representation_probe.py:76-82`). The program warns that diagnostics are in-sample but does not implement an episode/physics-disjoint fit/evaluation split (`src/tdpa/evaluation/representation_probe.py:175-185`).
- The standard evaluation script invokes the probe separately for each task and supplies only that task's TRAIN interaction archive (`scripts/run_all_evaluations.sh:5-16`). Each such archive has one task class and one `probe_train` policy class, so task-ID and policy-ID scores return `None`; cross-policy retrieval also lacks two policies. Read-only inspection confirmed this for both smoke archives.
- The standard collection script creates only TRAIN archives and does not create paired ID/policy-shift diagnostic archives (`scripts/collect_interactions.sh:4-5`). Consequently, the newly implemented cross-policy metric is not exercised by the advertised pipeline.
- No diagnostic checks the supplied archives against the encoder checkpoint's pretraining dataset hashes, so `response_normalized_mse_on_supplied_data` can still be reported on pretraining data rather than held-out actions (`src/tdpa/training/train_encoder.py:137-146`; `src/tdpa/evaluation/representation_probe.py:118-159`).

Required fixes before claiming task/policy invariance:

1. Collect separate, paired ID and policy-shift interaction archives for both tasks and held-out seeds/physics samples.
2. Invoke one combined diagnostic over both tasks and both policies; fail loudly rather than accepting `None` for required metrics.
3. Fit task/probe/policy heads on one episode/physics partition and score them on a disjoint partition.
4. Compare supplied archive hashes against checkpoint pretraining hashes and label response metrics as train or held-out.

### Student-only deployment artifact boundary: PASS

Evidence:

- Encoder training writes a second checkpoint containing only `bundle.student.state_dict()`, labels it `artifact_type: deployment_student`, declares no privileged inputs, and records the source training-checkpoint hash (`src/tdpa/training/train_encoder.py:149-161`).
- `DeploymentEncoderArtifact` owns only a `HistoryEncoder`; it has no teacher, target encoder, or response predictor (`src/tdpa/models/bundle.py:60-80`).
- The deployment loader rejects any checkpoint lacking the student-only artifact marker and strictly loads its state into that student (`src/tdpa/models/bundle.py:90-100`).
- Adapter training and learned OOD evaluation use only `load_deployment_encoder` (`src/tdpa/training/train_adapter.py:122-137`; `src/tdpa/evaluation/evaluate_ood.py:15-26`). Shell/README commands use the `_student.pt` checkpoint (`scripts/train_adapters.sh:4-10`; `scripts/run_all_evaluations.sh:5-12`; `README.md:22-27`).
- The regression test rejects a full training bundle (`tests/test_encoder_shapes.py:59-70`). A targeted read-only artifact inspection found no teacher/response/predictor keys in the student checkpoint, and the runtime loader rejected the corresponding training checkpoint.

## Re-audit tests and runtime checks

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q -p no:cacheprovider` — **60 passed**.
- Paired ID/policy-shift execution — equal physics and distinct trace hashes for no-adaptation, oracle, and learned adapted Push.
- Student artifact inspection — only the `student` child exists; no teacher/response/predictor state keys; full training checkpoint rejected by deployment loader.
- Cross-task reset/masking — Push/Lift proprioception equal at reset; RGB-D equal after masking goal channel 2.
- Default diagnostic-input inspection — each per-task TRAIN archive contains one task and one policy, confirming DL-5B remains non-diagnostic in the advertised runner.

## Revised merge/claim gate

The leakage and temporal implementation may merge for synthetic infrastructure use. Supported claims are limited to:

- no direct privileged physics channel reaches the proposed deployment student/adapter path;
- causal history/future windows and online history are aligned under the documented synthetic one-step convention;
- policy-shift rows now execute a genuinely different action style;
- identified deterministic goal/reset/probe task proxies are removed from shared-encoder inputs.

Unsupported until DL-5B is repaired and experiments are run:

- the learned latent is task-invariant, probe-invariant, or policy-invariant;
- the representation transfers scientifically across behavior policies;
- any robotics or real-controller conclusion beyond the synthetic backend.
