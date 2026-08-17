# Baseline audit: Domain-randomized policy (B1)

```yaml
baseline_name: domain_randomized
official_source_or_reference: generic domain-randomization control
implementation_source: src/tdpa/baselines/domain_randomized.py
observation_space: proprioception in the current scaffold
action_space: 4D normalized task action
history_length: none
task_conditioning: separate model per task
physics_randomization: must use train support with composition holdout excluded
training_data_budget: not yet allocated
trainable_parameters: architecture is implemented; count depends on configured hidden size
optimization_steps: not yet run
hyperparameter_search_budget: not yet run
evaluation_split: required five-way protocol
known_deviation_from_original_method: no BC/diffusion training or visual backbone is implemented
status: scaffold_only_not_eligible_for_main_table
```

