# Revised MVP Representation Audit

Date: 2026-08-17  
Scope: re-audit of the revised response target, command/history alignment, anti-shortcut sampling, policy-shift path, representation diagnostics, and deployment artifacts. No code was changed by this audit.

## Verdict

The former P0 implementation defects are substantially fixed. The response target is now deterministic, bias-free, parameter-free, and visibly response-sensitive; future commands and responses share exactly the same causal mask; the model sees executed controller targets; goal rendering and task-specific probe commands no longer provide the earlier direct task shortcut; policy shift changes action trajectories; and deployment accepts only a student-only artifact.

The representation pipeline is therefore a credible **experiment scaffold**, not a structurally collapsed one. Claims remain blocked by evidence: the checked response model is a one-epoch smoke run whose predictor is no better when given the correct latent, the required probes are still in-sample by default, cross-policy retrieval has no checked paired-policy result, response features remain unnormalized, V1/V3 current-format results are absent, and the scratch/proprio/downstream-latent ablations are not implemented.

All 60 tests pass. The revised tests directly exercise the key causal, sensitivity, leakage, artifact, and behavior-shift contracts.

## Verified fixes

### 1. Response target collapse risk

`ResponseEncoder` no longer contains a randomly initialized GRU/head. It deterministically summarizes each masked response sequence by per-feature mean, last value, and deviation, then applies a fixed sinusoidal 36×32 projection registered as a buffer. There are no trainable parameters or additive neural biases. A test verifies that a physical-response impulse changes the target.

An independent audit on the 72 complete windows in the current smoke Push/Lift archives confirms that the old near-constant-direction failure is gone:

- zero-norm target fraction: 0;
- mean cosine similarity to the dataset mean direction: 0.231 (previously 0.980);
- minimum cosine similarity: -0.196;
- constant-mean normalized MSE: 0.0480 (previously 0.00123).

This establishes response sensitivity and removes the former trivial target-collapse diagnosis. It does not establish that the chosen fixed projection is the best physical response representation.

### 2. Causal command/response pairing

`SequenceDataset.window_alignment()` now excludes the last command when its causal response does not exist. `future_action_mask`, `future_response_mask`, and the predictor mask are identical for every valid pair. Tests verify that unpaired command values cannot change the model input or prediction.

### 3. Executed controller command

History/future commands are now 7-D: the post-clipping 4-D action plus stiffness/300, damping/60, and grip-force/60. Collection steps the environment with exactly that action/controller pair, and both offline data and online deployment append `AppliedAction.execution_command`. This resolves the earlier omission of controller targets and the offline/online action-contract mismatch.

### 4. Probe and task shortcuts

- a weighted sampler balances probe labels in expectation during training;
- every physics sample is still reused across all nine primitives;
- Push and Lift now receive identical scripted probe command sequences;
- the history encoder masks RGB-D goal channel 2;
- tests verify task-neutral probe commands and goal-channel invariance.

These changes remove direct task-goal/action-label shortcuts. Interaction responses can still legitimately reveal the task/contact regime, which is why the task probe remains important.

### 5. Policy shift

The evaluation runner now transforms nominal actions for the held-out `chirp` behavior and logs an executed-command trace hash. ID and policy shift reuse the same physics but produce different traces, as verified by test and by non-identical checked smoke metrics.

### 6. Diagnostics

`representation_probe.py` now provides:

- mass and friction linear probes;
- probe-, policy-, and task-ID nearest-centroid probes, returning `None` for a one-class label;
- same-physics/cross-policy versus different-physics latent distances;
- normalized response error on caller-supplied archives;
- rolled/shuffled-latent response error;
- a zero-command counterfactual embedding-change diagnostic.

These additions make the intended failure modes measurable. Their current statistical limitations are listed below.

### 7. Deployment isolation

Training writes a separate `_student.pt` checkpoint. `load_deployment_encoder()` checks `artifact_type == "deployment_student"`, instantiates only `HistoryEncoder`, and refuses a training bundle. Adapter training and OOD evaluation both use this loader. The checked student state contains only `head`, `image_encoder`, `sensor_projection`, and `temporal` weights and declares an empty privileged-input list.

## Evidence that is still missing

### Current smoke model does not demonstrate latent use

The checked `smoke_probe.json` reports response MSE 0.06320 and rolled-latent MSE 0.06327—a difference of only 0.00007. Independent controls on the same 72 windows give:

- correct latent: 0.06320;
- rolled latent: 0.06327;
- zero latent: 0.06340;
- zero future command: 0.06317;
- constant target-direction predictor: 0.04804.

This is expected to be weak after one smoke epoch and is not a negative result about the method. It is nevertheless decisive for claims: the checked checkpoint provides no evidence that either history physics or future action is being used usefully. A converged model must beat the constant/action-only controls on held-out data and show a material penalty for physics-mismatched latent shuffling.

### Response and history features remain unnormalized

Checkpoint/evaluation metadata still state `normalization_statistics: identity`. Controller targets are scaled, but response, proprioceptive, and privileged teacher features are not standardized. In the current smoke windows, response-feature standard deviations range from 0 to roughly 0.97, while most kinematic components are below 0.13. Normalizing the final embedding does not correct the relative weighting of input units before projection. Train-only normalization statistics must be checkpointed and reused at deployment.

The target norm also ranges from approximately 0.0024 to 5.88. Normalized MSE gives nearly quiet and high-energy responses equal directional weight. Target-norm/effective-rank reporting and a declared treatment of quiet windows are needed.

### Diagnostics are implemented but not yet publication-valid

| Diagnostic | Revised status | Remaining limitation |
|---|---|---|
| D1 mass/friction linear probe | Implemented | Fit and score still use the same samples. No held-out physics, train-fitted normalization, constant baseline, per-seed uncertainty, or cross-task direction is reported. The checked smoke R2 values of 1.0 are in-sample and not interpretable. |
| D2 task/probe/policy probe | Implemented | Nearest centroids are fitted and scored on the same embeddings; no held-out split, chance/majority baseline, or uncertainty. The default train archive has one policy, correctly returning `None`. |
| D3 cross-policy retrieval | Implemented as a callable metric | The checked smoke result is `None`; default collection/evaluation scripts do not create and jointly pass paired ID and policy-shift archives. Distances use raw Euclidean latent scale and average across task/probe/time rather than explicit matched tuples. |
| D4 held-out action-response prediction | Partial | Response error can be computed on any supplied archive, but no launcher enforces that it is a held-out action distribution. Zero-command sensitivity is not correctness on a real counterfactual response. |
| Latent-use test | Partial | `roll(1)` often pairs adjacent windows from the same episode/physics and is therefore a weak shuffle; batch-size-one leaves it unchanged. Shuffle explicitly across different physics while matching task/probe/action phase. |

`run_all_evaluations.sh` passes one train archive at a time to the probe, so task and policy probes are one-class and cross-policy retrieval cannot run there. The diagnostics require a declared archive matrix and held-out fitting/scoring protocol before producing scientific numbers.

## Remaining objective and shortcut risks

1. **V1 teacher training is still simultaneous with student distillation.** The teacher target is now properly response-sensitive, but the student follows a moving teacher. A staged/frozen-teacher comparison remains useful.

2. **The teacher may encode instantaneous contact state instead of persistent physics.** Its privileged history contains object/EE state, contact force/state, and action, while the short-horizon objective rewards any predictive context. Held-out-action and cross-policy tests must decide whether this is harmful.

3. **Normalized distillation leaves latent norm uncontrolled.** Distillation constrains direction, while the downstream adapter consumes raw `z`. Either normalize the deployed latent or report latent-norm correlations and zero/renormalized-latent ablations.

4. **GRU left padding still advances recurrent biases.** Zeroing padded inputs does not make padding invisible; sequence position/time-since-reset remains available as a phase shortcut. The online/offline windows now agree, but packed sequences or explicit hidden-state reset are required if padding is meant to be ignored.

5. **Balanced sampling is not action-matched sampling.** `WeightedRandomSampler` balances probe labels over an epoch in expectation, not physics/action phase within a batch. The `--unbalanced` flag now enables the requested anti-shortcut ablation, but action-only and action-matched controls remain necessary.

6. **“Task-free” still needs careful wording.** Rewards, goal pixels, and task-specific probe commands are now excluded, which is a strong improvement. Data are nevertheless generated by the Push and Lift task-specific plants/contact models. Until transfer to a held-out task or environment is tested, “reward-free interaction pretraining across the two task regimes” is more defensible than universal task-independent pretraining.

## Ablation status

- `response`, `distill`, and `hybrid` training code is executable.
- `hybrid --unbalanced` now implements the sampling ablation mechanically.
- RGB-D + proprioception is the default.
- no proprio-only configuration/launcher exists;
- no shared random/scratch encoder downstream control exists;
- `representation_ablation.yaml` still declares names and equalization flags that no experiment launcher consumes;
- `adaptation_curve.py` still produces only `pretrained_shared`, with no zero/shuffled-latent or scratch rows.

Only the response smoke artifact was regenerated in the new 7-D-command/student-artifact format. The checked distill and hybrid training artifacts retain the older 4-D action configuration and have no matching student-only artifact, so they are stale and cannot support V1/V3 comparison.

## Priority acceptance criteria

1. Fit and checkpoint train-only normalization for response, proprioception, actions/controller targets, and privileged teacher channels.
2. Train V1/V2/V3 to convergence on current-format data and evaluate response loss on held-out physics plus a genuinely held-out action/policy archive.
3. Add constant, action-only, zero-latent, and explicitly different-physics shuffled-latent baselines; require a predeclared material gap.
4. Split all linear/ID probes into fit and held-out score sets and report chance baselines and three-seed uncertainty.
5. Collect paired ID and policy-shift archives with identical physics and run cross-policy retrieval on task/probe/time-matched groups.
6. Implement proprio-only, RGB-D, balanced/unbalanced, and shared-scratch ablations through one budget-equalized launcher.
7. Evaluate downstream adapters with zero and physics-mismatched latents. If task performance is unchanged, stop and classify the representation as unused.
8. Generate the primary adaptation-data curves against the required baselines before selecting V1/V2/V3 by downstream performance.

## Claims now supportable

The code now supports these implementation claims:

- the response target is deterministic, bias-free, response-sensitive, and not near-constant on the current smoke data;
- action-response windows are causal and fully paired, and include normalized executed controller targets;
- probe sampling is balanced in expectation, direct goal/task-action shortcuts are masked, and policy shift changes behavior;
- task/policy/probe, retrieval, response, latent-shuffle, and action-sensitivity diagnostic functions exist;
- deployment artifacts are student-only and exclude privileged teacher/predictor modules.

It still does **not** support claims that the latent predicts held-out physical response, encodes mass/friction out of sample, is task/probe/policy invariant, improves downstream adaptation, benefits from RGB-D, transfers between tasks, or that hybrid outperforms response/distillation alone.
