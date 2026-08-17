# MVP implementation review

```yaml
change: >-
  Added a deterministic synthetic Push/Lift research backend, physics split validation,
  interaction archives, causal sequence windows, V1/V2/V3 module interfaces, bounded
  adapters, oracle gate, baseline scaffolds, diagnostics, evaluation CLIs, and tests.
research_reason: >-
  Make the cheapest falsification and integration path runnable before investing in
  MuJoCo assets or long training runs.
assumptions:
  - synthetic sensor timestamp i precedes action i and response i+1
  - synthetic RGB-D markers are a deployable stand-in for a perception pipeline
  - surrogate results cannot support robotics or representation claims
new_hyperparameters:
  - synthetic contact dynamics and force bounds in configs/env
  - bounded adapter ranges in configs/adapter
  - history/future lengths and model sizes in configs/encoder
privileged_inputs_used:
  - teacher_history during V1/V3 training
  - true mass/friction for the explicit oracle and diagnostic metadata
deployment_inputs_used:
  - rgbd
  - proprioception
  - past commanded action
expected_failure_modes:
  - surrogate dynamics do not establish MuJoCo or real-robot validity
  - fixed response-summary target remains a design choice requiring held-out validation
  - learned baseline training runners remain incomplete
  - visual marker policy is benchmark scaffolding, not BC or Diffusion Policy
tests_added:
  - deterministic disjoint physics support
  - causal impulse alignment and future-leakage exclusion
  - privileged-key rejection and visually identical physics
  - adapter/model bounds and shape contracts
  - archive roundtrip and deterministic equal-manifest evaluation
  - oracle-first gate
```
