# Baseline audit: TAM-like (B5)

```yaml
baseline_name: tam_like
paper_or_repo_reference: https://github.com/Dongwon-Son/TAM
reference_commit_checked: 8b1e1bb8e6723bcb507319d1d6d097c258c52e79
implementation_source: src/tdpa/baselines/tam_like.py
observation_space: nominal action and instantaneous tracking residual
action_space: bounded residual on normalized task action
history_length: none in current approximation
task_conditioning: none
physics_randomization: not yet trained
training_data_budget: not yet allocated
trainable_parameters: small MLP; not capacity-matched
optimization_steps: not yet run
hyperparameter_search_budget: not yet run
evaluation_split: required five-way protocol
known_deviation_from_original_method: >-
  Major. The checked TAM repository uses history-conditioned torque correction, simulated
  rollout data, JAX/MuJoCo or MJX training, multi-robot workflows, and deployment-side torque
  history contracts. The local class is only a closest-interface residual control and must
  always be labeled TAM-like.
status: scaffold_only_do_not_label_results_as_tam
```

