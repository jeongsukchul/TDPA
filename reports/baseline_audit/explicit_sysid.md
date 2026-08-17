# Baseline audit: Explicit system identification (B2)

```yaml
baseline_name: explicit_sysid
official_source_or_reference: Project_Pipeline.md B2
implementation_source: src/tdpa/baselines/explicit_sysid.py
observation_space: proprioception and action history
action_space: estimates mass and friction; must feed the same PhysicalAdapter
history_length: experiment-controlled and must match the learned-latent method
task_conditioning: none in estimator
physics_randomization: train support only
training_data_budget: not yet allocated
trainable_parameters: implemented estimator architecture; adapter must be capacity-matched
optimization_steps: not yet run
hyperparameter_search_budget: not yet run
evaluation_split: required five-way protocol
known_deviation_from_original_method: no canonical paper is claimed; estimator training/evaluation is pending
status: architecture_only_not_eligible_for_main_table
```

