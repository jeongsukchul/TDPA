# Post-audit fixes

```yaml
trigger:
  - reports/leakage_audit/mvp_audit.md
  - reports/experiment_reviews/mvp_critic.md
  - reports/representation_audit/mvp_representation_audit.md
fixes:
  temporal_pairing:
    - predictor and target use the same joint future action-response mask
    - unexecuted final commands are invalid and zero in model-facing windows
  online_history:
    - removed duplicate reset observation and dummy action
    - online/offline histories are tested byte-equivalent throughout warm-up
    - GRUs compact valid samples so left padding cannot encode episode phase
  execution_command:
    - history and future actions now contain 4D applied action plus normalized stiffness, damping, and grip targets
    - controller clipping contributes to saturation metrics
  shortcut_controls:
    - goal render channel is masked before the shared physics encoder
    - Push/Lift reset poses and probe commands are harmonized
    - probe-balanced sampling is on by default with an explicit unbalanced ablation flag
  policy_shift:
    - chirp behavior changes real action trajectories at paired physics/reset states
    - action trace hashes are logged and tested distinct
  deployment_boundary:
    - training emits a student-only checkpoint
    - deployment loaders reject full teacher/response bundles
  response_target:
    - replaced biased random GRU target with deterministic mean/last/deviation response projection
    - response channels are normalized from TRAIN interactions only and checkpointed
  diagnostics:
    - added task/policy/probe probes, cross-policy retrieval, constant/zero/shuffled-latent controls, and counterfactual actions
    - optional held-out enforcement rejects encoder-training archive reuse
  safety:
    - oracle grip is capped below the independent force threshold
    - all controller-output clipping is counted as saturation
    - isolated non-finite nominal/correction fallbacks are counted as saturation
  contact_modes:
    - tests cover free-space to contact to free-space
    - tests cover stick to slip to re-stick
verification:
  tests: 64_passed
  lint: ruff_clean
  smoke_pipeline: passed
  oracle_gate:
    manifest_hash: ad8141bdb5b0971f08fc476ec1ff776339208297d4e2a53dd5170cdedc186248
    rollouts: 1680
    decision: synthetic_only_pass
```

These fixes resolve software/dataflow blockers. They do not resolve the missing
MuJoCo backend, equal-budget learned baseline runners, or absence of evidence
that the smoke-trained latent improves downstream adaptation.
