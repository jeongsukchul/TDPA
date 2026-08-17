# Baseline audit: Multi-task RMA (B4)

```yaml
baseline_name: multitask_rma
paper_or_repo_reference: project-required extension of https://github.com/yichao-liang/rma4rma
reference_commit_checked: 2f938f6518709ac8cbda05c294b7765c6d16630d
implementation_source: src/tdpa/baselines/multitask_rma.py
observation_space: configurable RGB-D, proprioception, and action history
action_space: raw shared correction head; bounded task-aware mapping is still required
history_length: must match TDPA
task_conditioning: learned task embedding
physics_randomization: joint Push/Lift train support
training_data_budget: not yet allocated
trainable_parameters: implemented but not capacity-matched or reported
optimization_steps: not yet run
hyperparameter_search_budget: not yet run
evaluation_split: required five-way protocol
known_deviation_from_original_method: >-
  There is no official multi-task RMA implementation claimed here. The current class is an
  architectural scaffold and lacks the privileged-policy first stage, bounded output mapping,
  training runner, and equal-budget tuning needed for a valid result.
status: scaffold_only_not_eligible_for_main_table
```

