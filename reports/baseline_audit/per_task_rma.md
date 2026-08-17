# Baseline audit: Per-task RMA (B3)

```yaml
baseline_name: per_task_rma
paper_or_repo_reference: https://github.com/yichao-liang/rma4rma
reference_commit_checked: 2f938f6518709ac8cbda05c294b7765c6d16630d
implementation_source: src/tdpa/baselines/rma.py
observation_space: configurable RGB-D, proprioception, and action history
action_space: same bounded PhysicalAdapter contract as TDPA
history_length: must match TDPA in a comparison
task_conditioning: separate model per task
physics_randomization: train support only
training_data_budget: not yet allocated
trainable_parameters: implemented but not yet capacity-matched or reported
optimization_steps: not yet run
hyperparameter_search_budget: not yet run
evaluation_split: required five-way protocol
known_deviation_from_original_method: >-
  This scaffold is not the official RMA-squared reproduction. The reference uses a
  two-stage privileged PPO policy/environment encoder followed by a CNN plus temporal-conv
  adapter trained by latent regression in customized ManiSkill2. This repository currently
  provides only a task-local history encoder plus bounded adapter.
status: scaffold_only_do_not_label_results_as_official_rma
```

