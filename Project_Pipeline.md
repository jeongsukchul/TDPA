# Project Brief for Codex
## Task-Decoupled Physical Adaptation for Force-Aware Manipulation

**Status:** Research implementation specification  
**Primary objective:** Build a minimal but publication-grade experimental pipeline to test whether a dynamics representation learned independently of downstream task rewards can reduce task-specific physical adaptation cost across distinct manipulation interaction regimes.

---

# 1. Project Motivation

Modern robot imitation-learning, diffusion-policy, and VLA-style systems are strong at learning **semantic and geometric behavior** from RGB/RGB-D demonstrations. However, the same nominal behavior can fail when physical interaction conditions change substantially.

Examples:

- the same object geometry but different mass,
- the same grasp geometry but different gripper-object friction,
- the same push trajectory but different object-table friction,
- the same motion reference but different contact/load response.

A vision-based task policy may know **what motion should be attempted**, but this does not imply that the same trajectory, velocity, impedance, grip force, or contact force is appropriate under different physical conditions.

Existing adaptation paradigms cover parts of this problem:

- **RMA-style adaptation:** infers a hidden dynamics/context latent from interaction history, but the latent is usually learned together with a task-specific policy/reward.
- **TAM-style torque adaptation:** learns a reusable low-level dynamics correction layer, but primarily tries to recover execution of a prescribed reference rather than letting object/contact physics modify the behavior envelope itself.
- **Force-aware VLA methods:** incorporate force feedback into generalist policy pipelines, but the force adaptation is usually coupled to a particular policy/action decoder and does not directly answer whether physical interaction knowledge can be separately pretrained and reused across policies/tasks.
- **Explicit system identification:** estimates mass/friction/etc., but requires deciding in advance which physical parameters are sufficient and identifiable.

The project therefore asks a narrower and testable question:

> **Can an action-response representation pretrained independently of downstream task rewards reduce the amount of task-specific physical adaptation needed by frozen manipulation policies?**

The target is **not** to claim a universal physics foundation model at the beginning.

---

# 2. Core Research Hypothesis

Let a nominal task policy be

\[
\bar a_t = \pi_k(o_t, g_k)
\]

where \(k\) identifies a downstream manipulation task.

Instead of re-training the full task policy for every mass/friction distribution, learn a reusable dynamics/context encoder

\[
z_t = E_\phi(H_t)
\]

from task-independent interaction data, where the interaction history contains

\[
H_t =
\{RGBD_{t-h:t}, q_{t-h:t}, \dot q_{t-h:t}, u_{t-h:t}\}.
\]

A small downstream physical adapter then uses

\[
A_k(\bar a_t, z_t, s_t)
\]

to modify only the physical execution of the frozen task policy.

The main empirical claim to test is:

> A pretrained shared dynamics encoder should allow a new task to reach strong OOD physical robustness with substantially less task-specific adaptation data than training an RMA-style adaptation mechanism from scratch for that task.

This claim must be tested against **both per-task RMA and multi-task RMA**.

---

# 3. Important Scope Constraints

## 3.1 What this project is NOT initially claiming

Do **not** initially implement or claim:

- universal adaptation to arbitrary manipulation tasks,
- global re-planning from physical context,
- autonomous regrasp strategy selection,
- arbitrary switching between prehensile and non-prehensile strategies,
- a full contact world model,
- long-horizon learned dynamics prediction,
- a foundation model replacement,
- a completely shared adapter for every task.

These are possible follow-up directions only if the minimal hypothesis succeeds.

## 3.2 First-paper scope

The first implementation should focus on:

1. **shared dynamics/context representation**,  
2. **small downstream physical adapters**,  
3. **large mass/friction shifts**,  
4. **distinct interaction regimes**,  
5. **frozen base policies**,  
6. **adaptation-data efficiency**,  
7. **OOD robustness**,  
8. **real-robot validation after simulation success**.

---

# 4. Recommended Initial Software Stack

## Primary stack

- **robosuite**
- **robomimic**
- MuJoCo backend
- PyTorch
- Hydra or OmegaConf for configuration
- Weights & Biases or TensorBoard for logging

Reasons:

- manipulation tasks are easy to modify,
- dynamics parameters can be randomized,
- RGB/depth and proprioception are available,
- operational-space control is accessible,
- robomimic provides strong frozen BC / Diffusion-policy baselines,
- much lower engineering overhead than starting directly in a large GPU simulator.

## Second-stage stack

After the hypothesis is validated:

- Isaac Lab for higher-throughput experiments,
- variable impedance / wrench-control validation,
- larger-scale randomization.

Do not migrate to Isaac Lab before the core hypothesis is tested.

---

# 5. Minimal Task Set

Start with exactly two tasks.

## Task A: Non-prehensile Push-to-Target

Physics variables:

- object mass \(m\),
- object-table friction \(\mu_{table}\).

Base policy:

- BC / Diffusion Policy from nominal demonstrations.

Adapter outputs initially:

- Cartesian action residual \(\Delta x_{ee}\),
- velocity scale \(\alpha_v\),
- optional stiffness \(K\).

Metrics:

- success rate,
- final target error,
- completion time,
- peak contact force,
- overshoot.

## Task B: Grasp-Lift-Transport

Physics variables:

- object mass \(m\),
- gripper-object friction \(\mu_{grip}\).

Base policy:

- BC / Diffusion Policy from nominal demonstrations.

Adapter outputs initially:

- velocity scale \(\alpha_v\),
- local Cartesian residual \(\Delta x_{ee}\),
- Cartesian stiffness \(K\),
- damping \(D\),
- grip-force target \(F_{grip}\).

Metrics:

- success rate,
- drop rate,
- slip rate,
- peak/RMS interaction force,
- completion time.

## Optional Task C after A/B succeed

Wipe / surface following:

- tool-surface friction,
- surface/contact compliance.

Do not add Task C until shared-encoder transfer on A/B is meaningful.

---

# 6. Physics Randomization Protocol

Use visually identical objects whenever possible.

For each task define:

### Training range

Example only; actual values must be task-specific:

```yaml
mass:
  train: [m_min, m_max]

friction:
  train: [mu_min, mu_max]
```

### Evaluation splits

Implement five splits:

1. **ID**
   - mass and friction inside train range.

2. **OOD-Mass**
   - mass outside train range,
   - friction inside.

3. **OOD-Friction**
   - friction outside train range,
   - mass inside.

4. **OOD-Composition**
   - individually seen ranges or boundary ranges,
   - unseen heavy-mass + low-friction combinations.

5. **Policy-shift evaluation**
   - evaluate the physics encoder using interaction histories generated by a behavior policy not used during encoder pretraining.

This is important because the latent may otherwise encode policy style instead of dynamics.

---

# 7. Interaction Pretraining Dataset

The dynamics encoder must not be trained only on downstream task rollouts.

Create a separate interaction dataset:

\[
D_{int}
\]

using scripted or random physical probes.

Recommended interaction primitives:

- free-space Cartesian motion,
- short push,
- short pull,
- hold,
- lift pulse,
- vertical loading,
- lateral loading,
- normal surface press,
- short sliding motion.

For every randomized physical configuration:

1. sample a physical configuration \(\theta\),
2. execute multiple probe primitives,
3. store the complete action-response history,
4. optionally store privileged simulator state and F/T data,
5. do not use downstream task reward.

Record:

```text
RGB / depth
joint positions
joint velocities
commanded action
controller target
estimated / measured torque if available
end-effector pose and velocity
object pose and velocity
contact state
contact force / wrist F/T (training-only privileged channel)
physics parameters (diagnostic / privileged only)
task/probe identity (metadata only)
```

---

# 8. Representation-Learning Variants

Implement three variants before inventing additional objectives.

## V1. Privileged Latent Distillation

### Teacher encoder

Training-only privileged input:

\[
H_t^+ =
\{q,\dot q,u,x_{obj},\dot x_{obj},F/T,contact\}.
\]

\[
z_t^+ = E_{priv}(H_t^+).
\]

The privileged encoder must have a non-trivial anchor. Do not jointly minimize only

\[
\|z-z^+\|^2
\]

because both encoders could collapse.

Use a short-horizon response objective to train the teacher.

### Student encoder

Deployment input:

\[
H_t =
\{RGBD,q,\dot q,u\}.
\]

\[
z_t = E_{student}(H_t).
\]

Distillation objective:

\[
L_{distill}
=
\|
norm(z_t)
-
stopgrad(norm(z_t^+))
\|_2^2.
\]

---

## V2. Action-Response Self-Supervised Context

No privileged teacher.

Past history:

\[
z_t = E_\phi(H_t^-).
\]

Future action segment:

\[
U_t^+ = u_{t:t+\Delta}.
\]

Future physical response:

\[
R_t^+.
\]

The predictor receives the action separately:

\[
\hat r_t = P_\psi(z_t,U_t^+).
\]

A target response encoder produces:

\[
r_t^+ = stopgrad(E_R(R_t^+)).
\]

Use

\[
L_{response}
=
\|
norm(\hat r_t)
-
norm(r_t^+)
\|_2^2.
\]

or a contrastive version.

### Important anti-shortcut rule

The latent should not be able to solve the objective by merely encoding the action distribution.

Therefore:

- future action is explicitly supplied to the predictor,
- interaction pretraining must contain multiple actions under the same physics,
- batches should be balanced across probe primitives,
- optionally use action-matched negatives.

---

## V3. Hybrid — Primary Method Candidate

Use:

\[
L =
L_{response}
+
\lambda_d L_{distill}.
\]

No additional auxiliary losses initially.

Do **not** immediately add:

- explicit mass loss,
- explicit friction loss,
- contact loss,
- wrench reconstruction loss,
- task-invariance loss,
- multiple contrastive losses.

Only add an objective if an ablation identifies a concrete failure mode.

---

# 9. Response Representation

Do not predict raw future RGB-D frames.

The project is not a world-model paper.

Use short-horizon physical response features.

Recommended privileged response vector:

\[
R_t =
[
\Delta v_{ee},
\Delta \omega_{ee},
e_{track},
\tau_{res},
\Delta v_{obj},
\Delta \omega_{obj},
F/T
].
\]

Deployment encoder does not need F/T.

Prediction horizon:

- begin with 50–200 ms,
- tune only after observing whether the signal is informative.

Potential simplified response:

```text
EE velocity change
joint tracking residual
object linear velocity change
object angular velocity change
estimated torque residual
```

---

# 10. Downstream Adapter

Base task policy:

\[
\bar a_t=\pi_k(o_t)
\]

is frozen.

The adapter receives:

```text
nominal task action
dynamics latent
current proprioceptive state
optional current contact state
```

and outputs a small physical correction.

Recommended initial action interface:

\[
A_k(\bar a_t,z_t,s_t)
\rightarrow
[
\Delta x_{ee},
\alpha_v,
K,
D,
F_{grip}
].
\]

Not every task needs every output.

Use masks in the config.

Example:

```yaml
push:
  outputs:
    cartesian_residual: true
    velocity_scale: true
    stiffness: true
    damping: false
    grip_force: false

lift:
  outputs:
    cartesian_residual: true
    velocity_scale: true
    stiffness: true
    damping: true
    grip_force: true
```

---

# 11. Adapter Training

The project should test three levels of transfer.

## Level 1: shared encoder + task-specific small adapter

Primary realistic target.

\[
E_{shared}
+
A_{push},
A_{lift}.
\]

The encoder is frozen.

Only the small adapter is trained.

Measure how much task-specific data is required.

## Level 2: shared encoder + interaction-primitive adapter

Example:

- sliding/contact adapter,
- load-bearing adapter.

Tasks can share adapters when interaction mechanics overlap.

## Level 3: fully shared adapter

Only test after Level 1/2 work.

Do not make this the initial project requirement.

---

# 12. How to Train the Adapter

Preferred order:

## Option A: supervised teacher correction

Use a privileged task execution controller or optimized controller to generate:

```text
nominal action
physics-aware corrected action
desired stiffness/damping
desired grip force
```

Train adapter to predict the correction.

This is easiest to debug.

## Option B: offline RL

Once supervised behavior works, optionally fine-tune the adapter using offline RL.

## Option C: online residual RL

Only after A/B are stable.

Do not use full-policy online RL initially.

The adaptation space should remain small.

---

# 13. Baselines

These are required.

## B0. No physical adaptation

Frozen nominal policy.

## B1. Domain-randomized task policy

Task policy trained directly under mass/friction randomization.

## B2. Explicit system identification

Estimate:

\[
\hat m,\hat \mu
\]

from history.

Feed explicit estimates to the same adapter architecture.

This tests whether the learned latent is actually necessary.

## B3. Per-task RMA

Each task has its own adaptation representation and policy/training pipeline.

## B4. Multi-task RMA

Critical baseline.

Single multi-task policy with task conditioning and RMA-style context.

This addresses the reviewer objection:

> Why not simply train one multi-task RMA?

## B5. TAM-like adaptation

Low-level residual execution adaptation.

If official implementation is practical, integrate it.

Otherwise build the closest faithful reference baseline and clearly label the difference.

## B6. Privileged oracle context

Adapter receives true simulator physics parameters / privileged context.

This gives an upper bound.

---

# 14. Core Evaluation: Adaptation Data Efficiency

This must be the main experiment.

For each downstream task, vary the amount of physics-randomized task-specific adaptation data.

Example fractions:

```text
1%
5%
10%
20%
50%
100%
```

For each method plot:

- x-axis: task-specific physical adaptation data,
- y-axis: OOD task success.

Primary desired observation:

> The pretrained shared dynamics representation reaches comparable OOD success using significantly less downstream adaptation data.

Do not define success purely as a fixed numerical reduction before experiments.

Report the actual curve.

---

# 15. Representation Diagnostics

Do not rely only on t-SNE.

Implement:

## D1. Linear physics probe

Freeze encoder.

Train linear heads:

\[
z\rightarrow m
\]

\[
z\rightarrow \mu
\]

for diagnostic purposes only.

Do not backpropagate these probes into the encoder.

## D2. Task/probe-ID probe

Train:

\[
z\rightarrow task/probe\ ID.
\]

High accuracy may indicate behavior leakage.

Interpret carefully: low task-ID accuracy is not automatically sufficient for good physics representations.

## D3. Cross-policy retrieval

Same physics configuration but different behavior policy:

- measure latent similarity,
- compare against different-physics samples.

## D4. Held-out action-response prediction

Use an action distribution not present in training and evaluate response prediction.

---

# 16. Required Ablations

Keep ablations small and interpretable.

1. response-only representation,
2. distillation-only representation,
3. hybrid response + distillation,
4. hybrid without action-matched sampling / anti-shortcut design,
5. proprioception only,
6. RGB-D + proprioception,
7. shared encoder from scratch vs pretrained shared encoder.

Optional only if needed:

8. shared adapter vs task adapter.

Do not create a large multi-loss ablation grid unless necessary.

---

# 17. Main Metrics

## Task metrics

- success rate,
- completion time,
- final task error.

## Physical-interaction metrics

- peak contact force,
- RMS contact force,
- drop rate,
- slip rate,
- excessive-force violation rate,
- adaptation latency.

## Scalability metrics

- task-specific adaptation trajectories,
- task-specific gradient steps,
- task-specific wall-clock training,
- number of trainable task-specific parameters.

The paper should emphasize **adaptation cost**, not only peak success.

---

# 18. Real-Robot Plan

Do not start real experiments before simulation passes the first hypothesis test.

Minimum real tasks:

## Real Task 1: Push

Use visually identical boxes.

Vary:

- hidden internal weight,
- table surface material.

## Real Task 2: Lift

Use visually identical boxes.

Vary:

- hidden internal weight,
- gripper-object interface material.

Use F/T only for:

- teacher data if needed,
- evaluation metrics,
- diagnostics.

The deployment encoder should use only the intended deployable observations.

---

# 19. Suggested Repository Structure

```text
project_root/
├── README.md
├── pyproject.toml
├── configs/
│   ├── env/
│   │   ├── push.yaml
│   │   └── lift.yaml
│   ├── physics/
│   │   ├── train.yaml
│   │   └── ood.yaml
│   ├── encoder/
│   │   ├── response.yaml
│   │   ├── distill.yaml
│   │   └── hybrid.yaml
│   ├── adapter/
│   │   ├── push.yaml
│   │   └── lift.yaml
│   └── experiment/
│       ├── data_efficiency.yaml
│       ├── representation_ablation.yaml
│       └── ood_eval.yaml
│
├── src/
│   ├── envs/
│   │   ├── make_env.py
│   │   ├── push_env.py
│   │   ├── lift_env.py
│   │   └── physics_randomization.py
│   │
│   ├── data/
│   │   ├── interaction_collector.py
│   │   ├── rollout_collector.py
│   │   ├── sequence_dataset.py
│   │   └── normalization.py
│   │
│   ├── models/
│   │   ├── history_encoder.py
│   │   ├── privileged_encoder.py
│   │   ├── response_encoder.py
│   │   ├── response_predictor.py
│   │   ├── physical_adapter.py
│   │   └── task_policy_wrapper.py
│   │
│   ├── controllers/
│   │   ├── osc_wrapper.py
│   │   ├── impedance_interface.py
│   │   └── adapter_action_mapper.py
│   │
│   ├── training/
│   │   ├── train_base_policy.py
│   │   ├── train_response_encoder.py
│   │   ├── train_privileged_encoder.py
│   │   ├── train_student_encoder.py
│   │   ├── train_hybrid_encoder.py
│   │   └── train_adapter.py
│   │
│   ├── baselines/
│   │   ├── no_adaptation.py
│   │   ├── explicit_sysid.py
│   │   ├── rma.py
│   │   ├── multitask_rma.py
│   │   └── tam_like.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_task.py
│   │   ├── evaluate_ood.py
│   │   ├── adaptation_curve.py
│   │   ├── representation_probe.py
│   │   └── force_metrics.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       ├── checkpoints.py
│       └── geometry.py
│
├── scripts/
│   ├── collect_interactions.sh
│   ├── train_base_policies.sh
│   ├── pretrain_encoders.sh
│   ├── train_adapters.sh
│   └── run_all_evaluations.sh
│
└── tests/
    ├── test_physics_randomization.py
    ├── test_sequence_dataset.py
    ├── test_encoder_shapes.py
    ├── test_adapter_bounds.py
    └── test_deterministic_eval.py
```

---

# 20. Model Interface Contracts

Codex should preserve explicit interfaces between modules.

## History encoder

```python
class HistoryEncoder(nn.Module):
    def forward(
        self,
        rgbd_history,
        proprio_history,
        action_history,
        history_mask=None,
    ) -> torch.Tensor:
        """Return dynamics latent z: [B, latent_dim]."""
```

## Privileged encoder

```python
class PrivilegedEncoder(nn.Module):
    def forward(
        self,
        privileged_history,
    ) -> torch.Tensor:
        """Training-only privileged latent."""
```

## Response predictor

```python
class ResponsePredictor(nn.Module):
    def forward(
        self,
        dynamics_latent,
        future_action_sequence,
    ) -> torch.Tensor:
        """Predict future physical-response embedding."""
```

## Physical adapter

```python
class PhysicalAdapter(nn.Module):
    def forward(
        self,
        nominal_action,
        dynamics_latent,
        proprio_state,
    ) -> dict:
        """
        Return bounded correction parameters:
        - cartesian_residual
        - velocity_scale
        - stiffness
        - damping
        - grip_force
        """
```

---

# 21. Safety / Stability Constraints in the Adapter

The learned adapter must not output unconstrained controller gains.

Parameterize outputs through bounded mappings.

Example:

```python
velocity_scale = v_min + sigmoid(raw_v) * (v_max - v_min)
stiffness = k_min + sigmoid(raw_k) * (k_max - k_min)
damping = d_min + sigmoid(raw_d) * (d_max - d_min)
cartesian_residual = residual_max * tanh(raw_dx)
```

All bounds belong in configuration files.

Use controller-side clipping as a second safety layer.

---

# 22. Experiment Reproducibility Requirements

Every experiment must log:

```text
git commit
random seed
environment version
physics configuration
dataset version/hash
base-policy checkpoint
encoder checkpoint
adapter checkpoint
normalization statistics
evaluation split
```

Use at least 3 seeds in simulation for all main results.

Do not report only best seeds.

---

# 23. Implementation Milestones

## Milestone 0 — Infrastructure

- robosuite environment boots,
- physics randomization validated,
- deterministic seed test,
- RGB-D/proprio/action sequence collection works.

## Milestone 1 — Nominal task policies

- Push base policy succeeds in nominal physics,
- Lift base policy succeeds in nominal physics,
- both fail measurably under large physical shifts.

If nominal policies do not fail under the chosen variation, the benchmark is not informative.

## Milestone 2 — Representation sanity test

Implement V1, V2, V3.

Check:

- response-prediction error,
- mass/friction linear probe,
- cross-policy retrieval,
- task-ID leakage.

Do not select the representation only by probe accuracy.

## Milestone 3 — Downstream adaptation

Train small adapters with frozen encoders and frozen task policies.

Generate adaptation-data curves.

## Milestone 4 — Critical baselines

Implement:

- per-task RMA,
- multi-task RMA,
- explicit SysID,
- TAM-like baseline.

## Milestone 5 — OOD protocol

Run:

- ID,
- OOD mass,
- OOD friction,
- OOD composition,
- policy shift.

## Milestone 6 — Real robot

Only after simulation demonstrates meaningful adaptation-data advantage.

---

# 24. Kill / Pivot Criteria

These criteria are intentionally strict.

## K1. Shared representation provides no downstream data advantage

If the pretrained shared encoder does not reduce downstream adapter-training data compared with task-specific representation training:

> Drop the strong shared-representation claim.

## K2. Multi-task RMA matches the proposed method

If multi-task RMA achieves similar:

- OOD success,
- adaptation data efficiency,
- task-specific parameter cost,

then the current contribution is weak.

Do not hide this baseline.

## K3. Shared adapter fails

This is not fatal.

Fallback:

\[
E_{shared} + A_{task}^{small}.
\]

The main claim becomes representation reuse / data efficiency.

## K4. Physics representation fails across push and lift

If even the encoder itself does not transfer:

> Reconsider the task-decoupled representation hypothesis before adding more losses.

Do not rescue the project by adding many auxiliary objectives.

## K5. RGB-D adds no benefit

If proprioception-only performs equally well:

> simplify the method and report that result.

Do not force RGB-D into the architecture for narrative reasons.

---

# 25. Highest-Priority Research Questions

Codex implementation decisions should serve these questions in order:

1. Does a task-free interaction dataset contain reusable information about physical dynamics?
2. Which representation objective transfers best:
   - privileged latent distillation,
   - action-response self-supervision,
   - hybrid?
3. Does pretraining reduce downstream task-specific adaptation data?
4. Does the advantage survive against multi-task RMA?
5. Does the representation generalize across:
   - mass shifts,
   - friction shifts,
   - unseen combinations,
   - policy shifts?
6. Is a task-specific small adapter enough?
7. Can adapters be shared at the interaction-primitive level?
8. Only after these succeed:
   - can the same module be attached to a VLA/generalist policy?

---

# 26. Expected Paper-Level Story If the Project Succeeds

A credible final story would be:

1. Current adaptive manipulation methods often learn dynamics context together with each task policy.
2. We instead pretrain an action-response dynamics representation from heterogeneous, task-free interaction data.
3. The encoder is frozen and attached to independently trained manipulation policies.
4. Only a small physical adapter is trained for each downstream task or interaction primitive.
5. Across prehensile and non-prehensile manipulation under large mass/friction shifts, the pretrained representation reduces task-specific adaptation data while maintaining OOD performance.
6. The advantage remains against:
   - explicit system identification,
   - per-task RMA,
   - multi-task RMA,
   - low-level torque adaptation.
7. Real-robot experiments confirm the effect with visually identical objects under hidden mass/friction variation.

This is substantially more defensible than claiming a universal physical foundation layer from the outset.

---

# 27. Instructions to Codex

When implementing this project:

1. **Do not over-engineer the method before the minimal baseline works.**
2. Build experiment infrastructure before complex neural architectures.
3. Keep modules independently replaceable:
   - task policy,
   - history encoder,
   - privileged encoder,
   - response objective,
   - adapter,
   - controller.
4. Prefer configuration-driven experiments.
5. Every new loss must have a clearly identified failure mode it fixes.
6. Do not add a new loss merely because representation metrics look visually better.
7. Keep the frozen-base-policy assumption explicit in code.
8. Avoid accidental leakage of:
   - ground-truth mass,
   - ground-truth friction,
   - privileged object state,
   - F/T signals
   into deployment observations.
9. Add automated assertions that privileged channels are unavailable in deployment evaluation.
10. Prioritize implementation of **multi-task RMA** early, not after the proposed method is complete.
11. Produce adaptation-data curves automatically from the evaluation code.
12. Keep real-robot interfaces compatible with the same history-encoder and adapter APIs.

---

# 28. First Concrete Coding Sprint

Codex should begin with the following tasks, in this order:

### Step 1
Create project skeleton and configuration system.

### Step 2
Implement `Push` and `Lift` environments with configurable mass/friction.

### Step 3
Create a script that sweeps physics configurations and verifies that the intended MuJoCo parameters actually change.

### Step 4
Create interaction data collector with fixed-format sequence storage.

### Step 5
Train / load nominal frozen policies.

### Step 6
Verify that nominal policies fail under sufficiently strong OOD mass/friction shifts.

### Step 7
Implement the action-response dataset interface.

### Step 8
Implement:
- `HistoryEncoder`,
- `ResponseEncoder`,
- `ResponsePredictor`.

### Step 9
Train the response-only representation.

### Step 10
Add privileged encoder and latent distillation.

### Step 11
Implement hybrid training.

### Step 12
Implement representation diagnostics.

### Step 13
Implement small task adapters.

### Step 14
Generate first adaptation-data curve.

### Step 15
Only then implement RMA / multi-task RMA / TAM-like baselines.

---

# 29. Immediate Definition of Done

The first meaningful internal result is **not** a polished benchmark.

It is:

> On Push and Lift, a frozen shared encoder pretrained from separate interaction data allows a small task adapter to reach better OOD mass/friction performance than the same adapter trained with a randomly initialized or task-specific-from-scratch encoder at the same downstream data budget.

If this does not occur, investigate the representation hypothesis before expanding the project.

---

# 30. Summary

The project should be implemented around the following separation:

\[
\boxed{
D_{interaction}
\rightarrow
E_{dyn}
}
\]

learn reusable physical dynamics context,

\[
\boxed{
D_{demo}^{task}
\rightarrow
\pi_{task}
}
\]

learn nominal behavior independently,

and

\[
\boxed{
(\pi_{task},E_{dyn})
\rightarrow
A_{physical}
}
\]

learn a small amount of downstream physical adaptation.

The project succeeds only if this separation produces a measurable reduction in task-specific physical adaptation cost relative to strong alternatives such as per-task and multi-task RMA.

The architecture is therefore a **testable hypothesis about reusable physical adaptation**, not an assumption that a universal physics latent already exists.

---

# 31. Multi-Agent Critic Implementation Protocol

The implementation process itself should follow a **role-separated multi-agent verification loop**.

The purpose is not to maximize the number of agents. The purpose is to prevent the implementation from drifting toward a complex method that merely confirms the original hypothesis.

If the coding environment supports independent sub-agents, assign these roles to separate agents. If it does not, execute the same roles sequentially as explicit review passes with isolated prompts and separate outputs.

The implementation loop is:

```text
Research Spec
    ↓
Planner
    ↓
Primary Implementer
    ↓
Independent Implementer / Reproducer
    ↓
Experiment Critic
    ↓
Method Critic
    ↓
Data-Leakage / Evaluation Auditor
    ↓
Baseline Auditor
    ↓
Failure-Case / Edge-Case Reviewer
    ↓
Final Judge
    ↓
Merge / Reject / Revise
```

A change must not be merged merely because the main implementation runs.

---

## 31.1 Planner Agent

The Planner converts each research milestone into a small, falsifiable engineering task.

For every task, output:

```yaml
goal:
research_question:
minimum_implementation:
required_inputs:
expected_outputs:
baselines_affected:
tests_required:
failure_conditions:
files_expected_to_change:
```

The Planner must explicitly answer:

1. What scientific question does this implementation enable us to test?
2. What is the simplest implementation that can answer that question?
3. Which parts are research-essential and which are engineering convenience?
4. What result would falsify the intended hypothesis?
5. What information could accidentally leak privileged physics into deployment?

The Planner must reject tasks such as:

> "Add another auxiliary loss because representation quality may improve."

unless the task is connected to a documented failure mode.

---

## 31.2 Primary Implementer Agent

The Primary Implementer writes the main code.

Responsibilities:

- implement only the approved minimal scope,
- preserve module interfaces,
- write tests together with the implementation,
- keep train-time privileged channels separate from deployment observations,
- avoid silently changing benchmark definitions,
- avoid using evaluation OOD configurations during training.

Every implementation PR / patch should include a machine-readable summary:

```yaml
change:
research_reason:
assumptions:
new_hyperparameters:
privileged_inputs_used:
deployment_inputs_used:
expected_failure_modes:
tests_added:
```

The implementer should not decide that the method "works" based on training loss alone.

---

## 31.3 Independent Implementer / Reproducer Agent

For critical components, a second agent independently checks or reimplements the logic without copying the Primary Implementer's reasoning.

High-priority components for independent reproduction:

- physics randomization,
- OOD split generation,
- sequence slicing,
- history/action temporal alignment,
- privileged/student observation separation,
- adaptation-data budget accounting,
- task success computation,
- force/slip metrics,
- RMA and multi-task RMA baselines.

The Independent Implementer should attempt to reproduce the same result from:

- the written specification,
- public APIs,
- configuration files,

rather than trusting the first implementation.

If two implementations disagree, do not average the outputs. Identify the violated assumption.

---

# 32. Scientific Critic Roles

## 32.1 Method Critic

Assume the proposed method is unnecessary.

For every new module or loss, ask:

1. Can the same gain be obtained by a larger task policy?
2. Can multi-task RMA obtain the same gain?
3. Can explicit system identification obtain the same gain?
4. Can proprioception-only obtain the same gain?
5. Does the adapter simply memorize task identity?
6. Does the latent encode object/task identity rather than dynamics?
7. Does the method improve only ID performance?
8. Is the proposed component needed at the same downstream data budget?

The Method Critic should actively recommend deleting components.

A proposed component survives only if a controlled experiment identifies an advantage.

---

## 32.2 Experiment Critic

Assume the experiment is designed to make the proposed method look good.

Check:

- whether baselines receive equal observation history,
- whether baselines receive equal task-specific data budgets,
- whether baseline network capacity is comparable,
- whether all methods are tuned on the same validation protocol,
- whether OOD parameters are truly outside training support,
- whether the nominal policy is artificially weak,
- whether the physics shift is large enough to matter,
- whether the physics shift is unrealistically extreme,
- whether success metrics hide high force or unsafe behavior,
- whether reported adaptation cost ignores pretraining cost.

The critic must distinguish:

```text
pretraining cost
task-specific adaptation cost
inference cost
real-robot data cost
```

Do not claim general efficiency while reporting only one of these.

---

## 32.3 Baseline Auditor

This role is mandatory before claiming improvement.

For each baseline, create a checklist:

```yaml
baseline_name:
official_source_or_reference:
observation_space:
action_space:
history_length:
task_conditioning:
physics_randomization:
training_data_budget:
trainable_parameters:
optimization_steps:
hyperparameter_search_budget:
evaluation_split:
known_deviation_from_original_method:
```

Priority baselines:

1. no adaptation,
2. domain-randomized nominal policy,
3. explicit SysID + same adapter,
4. per-task RMA,
5. multi-task RMA,
6. TAM or TAM-like adaptation,
7. privileged/oracle context.

If a baseline cannot be faithfully reproduced, document the deviation instead of labeling an approximation as the original method.

---

## 32.4 Data-Leakage Auditor

This agent specifically attempts to break the train/deployment separation.

Search for accidental access to:

- true mass,
- true friction,
- object body IDs correlated with physics,
- material IDs,
- simulator contact parameters,
- privileged object pose,
- F/T signals,
- contact labels,
- OOD split labels,
- task IDs,
- probe IDs,
- episode filenames encoding physics.

The deployment path should fail loudly if privileged keys are present.

Add tests such as:

```python
def test_student_observation_contains_no_privileged_keys():
    ...

def test_ood_physics_values_are_not_in_training_dataset():
    ...

def test_object_identity_does_not_deterministically_encode_mass():
    ...

def test_eval_metadata_is_not_passed_to_policy():
    ...
```

Also randomize metadata and object IDs where practical to test for shortcut learning.

---

## 32.5 Temporal-Alignment Auditor

This project depends critically on action-response relationships.

Therefore explicitly test that:

\[
H_t^- \rightarrow U_t^+ \rightarrow R_t^+
\]

is temporally correct.

Check for:

- off-by-one errors,
- controller-rate vs policy-rate mismatch,
- delayed sensor timestamps,
- future information leaking into the history encoder,
- response windows overlapping incorrectly,
- action chunks being indexed at different rates.

Add synthetic tests where a known impulse occurs at a known timestep and confirm the sequence dataset returns the correct past/action/future windows.

This is a high-priority audit.

---

## 32.6 Representation Critic

Assume the learned latent is not a dynamics representation.

Test alternative explanations:

### Task leakage

Train:

\[
z \rightarrow \text{task/probe ID}.
\]

### Object identity leakage

Train:

\[
z \rightarrow \text{object ID}.
\]

### Action-style leakage

Train:

\[
z \rightarrow \text{behavior policy ID}.
\]

### Physics information

Train diagnostic probes:

\[
z \rightarrow m,
\qquad
z \rightarrow \mu.
\]

### Cross-policy consistency

For the same physical configuration but different behavior policies, measure whether latent distances are smaller than for different physical configurations.

### Counterfactual action test

Hold the inferred latent fixed and change the future action supplied to the response predictor.

The predicted response should change appropriately.

This helps verify that the predictor is not ignoring the explicit action input.

---

# 33. Counterexample and Edge-Case Agent

This role intentionally constructs cases where the hypothesis should fail.

Required edge cases:

## E1. Unobservable physics

Create two physics configurations that produce nearly identical response histories under a weak probe.

Expected behavior:

- encoder uncertainty or indistinguishability,
- no artificial claim of exact identification.

If the deterministic encoder is used, document that the latent cannot resolve observationally equivalent dynamics.

## E2. Same mass, different friction

Verify latent/adaptation changes when only contact friction changes.

## E3. Same friction, different mass

Verify the opposite axis.

## E4. Different physics, similar short-horizon response

Test whether a longer history or additional excitation is needed.

## E5. Different policy, same physics

Test policy leakage.

## E6. Same task, different controller gain

Determine whether the encoder treats robot-controller mismatch as dynamics context.

This may be useful, but it must be explicitly characterized.

## E7. Contact-mode transition

Test:

```text
no contact → contact
stick → slip
slip → re-stick
```

Do not assume one smooth latent/operator explains all modes.

## E8. Adapter saturation

Force the latent or nominal action toward extreme conditions and verify bounded adapter outputs.

---

# 34. Final Judge Agent

The Final Judge decides whether a change advances the research project.

It must not use majority vote.

A change should be accepted only if:

1. the implementation is correct,
2. the evaluation is fair,
3. privileged information is contained,
4. the result addresses a stated scientific question,
5. a strong alternative explanation has been tested,
6. the added complexity is justified.

The Final Judge produces one of:

```text
ACCEPT
ACCEPT_WITH_LIMITED_CLAIM
REVISE
REJECT
PIVOT
```

Example:

```yaml
decision: ACCEPT_WITH_LIMITED_CLAIM

supported_claim:
  "The pretrained encoder improves Push adaptation at a fixed data budget."

unsupported_claims:
  - "The representation is task-independent."
  - "The method generalizes across interaction regimes."

required_next_test:
  "Repeat on Lift with identical adaptation budget and multi-task RMA baseline."
```

This prevents one positive experiment from silently expanding into a broader contribution.

---

# 35. Multi-Agent Review Loop for Every Major Milestone

For each major milestone use:

```text
1. Planner defines the falsifiable question.
2. Primary Implementer writes code.
3. Unit/integration tests run.
4. Independent Reproducer checks critical logic.
5. Data-Leakage Auditor checks observation boundaries.
6. Experiment Critic checks fairness.
7. Baseline Auditor verifies comparison conditions.
8. Representation Critic tests shortcut hypotheses.
9. Edge-Case Agent constructs failure cases.
10. Final Judge accepts, narrows, rejects, or requests revision.
```

If a major issue is discovered:

```text
Revision
    ↓
re-run relevant critics
    ↓
Final Judge
```

Do not re-run every reviewer for cosmetic changes.

---

# 36. Multi-Agent Workflow for Representation Objectives

Each representation variant must be evaluated independently.

## Candidate V1 — Distillation

Question:

> Does privileged latent distillation improve downstream adaptation efficiency?

Critic focus:

- teacher latent collapse,
- privileged information not inferable from student history,
- latent MSE improving while downstream task performance does not.

## Candidate V2 — Response Prediction

Question:

> Does action-conditioned response prediction learn reusable dynamics context without privileged teacher input?

Critic focus:

- task/action leakage,
- response predictor ignoring latent,
- latent memorizing trajectory phase.

## Candidate V3 — Hybrid

Question:

> Does combining response anchoring and latent distillation provide a meaningful gain beyond either component?

Critic focus:

- gain caused only by increased model size,
- sensitivity to \(\lambda_d\),
- unnecessary complexity.

The Hybrid becomes the main method only if it consistently beats the simpler variants.

Otherwise publish the simpler method.

---

# 37. Multi-Agent Workflow for Baseline Fairness

Before any main result table is generated, require a baseline review artifact:

```text
reports/baseline_audit/<baseline>.md
```

Each file should include:

```yaml
paper_or_repo_reference:
implementation_source:
changes_from_original:
training_budget:
observation_history:
action_interface:
network_capacity:
hyperparameters:
validation_method:
known_limitations:
```

No baseline result should enter the main table without this audit.

---

# 38. Automated Experiment Sanity Checks

Before expensive training, run automated checks.

## Physics separation

Verify randomization produces expected MuJoCo values.

## Observation ablation sanity

Check that removing physics-sensitive history causes expected degradation in a controlled synthetic test.

## Response predictability

Before training the full encoder, estimate whether the selected response target has meaningful variation across mass/friction conditions.

## Nominal policy failure

Verify the nominal policy:

- succeeds in nominal physics,
- degrades under target OOD shifts.

If it succeeds everywhere, adaptation is unnecessary.

If it fails everywhere, the task policy is inadequate.

## Oracle upper bound

Use true physics context with the same adapter.

If the oracle adapter cannot improve the task, the proposed latent cannot solve the problem either.

This is one of the earliest required sanity checks.

---

# 39. Oracle-First Development Rule

Before investing heavily in representation learning, test:

\[
A_k(\bar a_t,\theta^{GT})
\]

where the adapter receives true physics parameters or privileged physical state.

Ask:

> If the adapter knew the physics perfectly, could it actually recover task performance?

If the answer is no, the bottleneck is not context estimation.

Possible causes:

- nominal trajectory is unrecoverable,
- adapter action space is insufficient,
- controller is insufficient,
- task requires global re-planning.

Only proceed to sophisticated latent learning if the oracle-context adapter demonstrates a meaningful upper bound.

This rule is mandatory because it prevents representation learning from being blamed for an inadequate control interface.

---

# 40. Critic-Gated Development Order

The implementation order should therefore be revised to:

```text
1. Environment + physics randomization
2. Nominal policy
3. OOD failure verification
4. Oracle-context adapter
5. Explicit-SysID baseline
6. Per-task RMA
7. Multi-task RMA
8. Interaction dataset
9. Response-only encoder
10. Distillation encoder
11. Hybrid encoder
12. Adaptation-data curves
13. Representation diagnostics
14. Optional primitive-shared adapter
15. Real robot
```

This order is intentionally different from a proposal-first implementation.

The strongest baselines and oracle should appear early so that the project can be killed or pivoted quickly.

---

# 41. Multi-Agent Output Artifacts

Create the following project directories:

```text
reports/
├── plans/
├── implementation_reviews/
├── baseline_audit/
├── leakage_audit/
├── representation_audit/
├── experiment_reviews/
├── failure_cases/
└── final_judgments/
```

For each major experiment, preserve:

```text
plan.md
implementation_review.md
baseline_audit.md
leakage_audit.md
critic_report.md
final_judgment.md
```

The reports should be concise and decision-oriented.

Do not store hidden chain-of-thought.

Store only:

- conclusions,
- evidence,
- identified issues,
- tests performed,
- required fixes,
- final decisions.

---

# 42. Example Agent Prompts

## Planner prompt

```text
You are the Planner for a robotics research implementation.
Convert the requested milestone into the smallest falsifiable experiment.
Identify required code changes, baselines, tests, leakage risks, and kill criteria.
Do not assume the proposed method is correct.
```

## Method Critic prompt

```text
Assume the proposed method is unnecessary.
Find simpler explanations and stronger baselines that could reproduce the reported gain.
Focus on multi-task RMA, explicit SysID, proprioception-only representations, and controller differences.
Recommend deleting any component not supported by controlled evidence.
```

## Experiment Critic prompt

```text
Assume the experiment was unintentionally designed in favor of the proposed method.
Check observation spaces, data budgets, network capacity, tuning budgets, OOD splits, success metrics, and compute accounting.
List every unfair comparison and the concrete repair required.
```

## Data-Leakage Auditor prompt

```text
Attempt to infer ground-truth physics from implementation metadata rather than sensor history.
Search for mass, friction, object ID, simulator state, F/T, contact labels, task ID, filenames, configuration values, and future information leaking into deployment inputs.
Report exact files and fields that must be removed or guarded.
```

## Representation Critic prompt

```text
Assume the latent does not represent dynamics.
Test whether it instead represents task identity, action distribution, object identity, trajectory phase, or policy identity.
Specify diagnostics and counterfactual tests that distinguish these hypotheses.
```

## Final Judge prompt

```text
Judge the evidence, not the number of reviewers agreeing.
Accept only claims directly supported by fair experiments.
Return ACCEPT, ACCEPT_WITH_LIMITED_CLAIM, REVISE, REJECT, or PIVOT.
Explicitly state which claims are supported, unsupported, and what decisive experiment comes next.
```

---

# 43. Final Implementation Principle

The coding agent should optimize for:

\[
\boxed{
\text{fast falsification}
>
\text{method complexity}
}
\]

The goal is not to implement the full conceptual architecture.

The goal is to discover as early as possible whether:

\[
\boxed{
\text{task-free action-response pretraining}
}
\]

actually creates reusable physical information that improves downstream adaptation under a controlled and fair comparison.

Every implementation decision should therefore survive two questions:

1. **What alternative explanation could produce the same result?**
2. **What is the cheapest experiment that could disprove our interpretation?**

Only components that survive this critic loop should remain in the final method.



Source Codes for the baseline methods

TAM :Torque Adaptation Module for Robust Motion Transfer in Manipulation
https://github.com/Dongwon-Son/TAM

RMA : Rapid Motor Adaptation for Robotic Manipulator Arms
https://github.com/yichao-liang/rma4rma
RMA-variant(Contact-Rich) CoRMA: Contrastive RMA for Contact-Rich Meta-Adaptation
https://arxiv.org/pdf/2605.22082v1
