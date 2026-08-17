# Synthetic oracle-first gate result

```yaml
backend: synthetic-v1
claim_scope: infrastructure and bounded-control-interface viability only
manifest_hash: ad8141bdb5b0971f08fc476ec1ff776339208297d4e2a53dd5170cdedc186248
seeds: [0, 1, 2]
episodes_per_task_method_split: 20
decision: PASS_SYNTHETIC_ONLY
checks:
  push:
    nominal_success_b0: 1.0
    ood_mass_success_b0: 0.617
    ood_mass_success_oracle: 1.0
    ood_friction_success_b0: 0.467
    ood_friction_success_oracle: 0.917
    oracle_force_violation_rate: 0.0
    oracle_saturation_rate: 0.0
  lift:
    nominal_success_b0: 1.0
    ood_mass_success_b0: 0.383
    ood_mass_success_oracle: 1.0
    ood_friction_success_b0: 0.667
    ood_friction_success_oracle: 0.883
    ood_composition_success_b0: 0.0
    ood_composition_success_oracle: 1.0
    oracle_force_violation_rate: 0.0
    oracle_saturation_rate: 0.0
unsupported_interpretations:
  - the surrogate result validates robosuite or MuJoCo physics
  - task-free pretraining improves adaptation efficiency
  - the representation is task- or policy-invariant
  - the method beats SysID, RMA, multi-task RMA, CoRMA, or TAM
next_gate: implement and validate the same manifest/readback protocol in robosuite/MuJoCo
```

The full per-episode machine-readable artifact is `artifacts/oracle_gate.json`.
The oracle intentionally uses the expanded configured adapter interface; the
minimal velocity/residual-only interface did not pass the independent critic's
ablation and is not claimed to be sufficient.
