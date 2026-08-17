# Baseline audit: No adaptation (B0)

```yaml
baseline_name: no_adaptation
official_source_or_reference: Project_Pipeline.md B0
implementation_source: src/tdpa/baselines/no_adaptation.py and evaluation/oracle_gate.py
observation_space: deployable RGB-D and proprioception only
action_space: frozen nominal 4D Cartesian/gripper action
history_length: none
task_conditioning: separate frozen controller per task
physics_randomization: evaluation manifest only
training_data_budget: zero
trainable_parameters: zero
optimization_steps: zero
hyperparameter_search_budget: zero
evaluation_split: ID, two-sided OOD mass, two-sided OOD friction, composition, policy shift
known_deviation_from_original_method: none; this is the project control
status: executable
```

