# Baseline audit: Privileged oracle (B6)

```yaml
baseline_name: privileged_oracle
official_source_or_reference: Project_Pipeline.md B6 and oracle-first rule
implementation_source: src/tdpa/controllers/oracle_context.py
observation_space: true mass and friction behind explicit enabled flag
action_space: same bounded residual/scale/gain/grip-force mapper used downstream
history_length: none; perfect current context
task_conditioning: transparent task-specific equations
physics_randomization: same byte-identical manifest as B0
training_data_budget: zero; engineering upper bound, not learned B6
trainable_parameters: zero
optimization_steps: zero
hyperparameter_search_budget: hand-selected on benchmark dynamics; disclose as engineering oracle
evaluation_split: all configured splits
known_deviation_from_original_method: this is a transparent controller oracle, not a learned privileged adapter
status: executable_upper_bound_only
```

