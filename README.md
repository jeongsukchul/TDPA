# Task-Decoupled Physical Adaptation (TDPA)

This repository is a minimal, runnable research pipeline for testing whether a dynamics
representation learned from task-free action-response data reduces downstream physical
adaptation cost. It implements the interfaces, leakage guards, physics splits, representation
objectives, adapters, early baselines, diagnostics, and experiment accounting specified in
`Project_Pipeline.md`.

The default backend is a deterministic, low-cost manipulation surrogate. It makes the complete
pipeline and its scientific sanity checks runnable without MuJoCo, robosuite, demonstrations, or
a GPU. It is an infrastructure and hypothesis-debugging backend, not evidence for a robotics
claim. The environment factory keeps the backend boundary explicit so robosuite tasks can replace
it without changing datasets, encoders, adapters, or evaluation code.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .

python -m tdpa.envs.verify_physics --task push
python -m tdpa.data.interaction_collector --task push --episodes 24 --output artifacts/push.npz
python -m tdpa.data.interaction_collector --task lift --episodes 24 --output artifacts/lift.npz
python -m tdpa.training.train_encoder --variant response --datasets artifacts/push.npz artifacts/lift.npz --epochs 2 --output artifacts/response.pt
python -m tdpa.training.train_adapter --task push --encoder artifacts/response_student.pt --episodes 24 --epochs 2 --output artifacts/push_adapter.pt
python -m tdpa.evaluation.evaluate_ood --task push --encoder artifacts/response_student.pt --adapter artifacts/push_adapter.pt
```

Install the development extra and run the test suite with:

```bash
pip install -e ".[dev]"
pytest -q
```

The shell entry points in `scripts/` run the same stages. Outputs under `artifacts/` and `runs/`
are ignored by git.

## Scientific boundaries

- Deployment models receive only RGB-D, proprioception, and action history.
- Physics values, object state, contact state, and force/torque are stored as privileged arrays
  and are rejected by the deployment-policy wrapper.
- Physics metadata is used only for dataset stratification, diagnostics, oracle adapters, and
  evaluation.
- ID and OOD supports are validated before sampling.
- The frozen-policy and frozen-pretrained-encoder assumptions are asserted in code.
- Results from the surrogate backend are smoke tests. Paper claims require robosuite/MuJoCo and,
  after passing the kill criteria, real-robot validation.

## Layout

Configuration lives in `configs/`; code lives in `src/tdpa/`; tests cover physics splits,
temporal alignment, leakage, model interfaces, bounded outputs, and deterministic evaluation.
Decision-oriented critic artifacts live in `reports/`.

The earliest decisive experiment is oracle-first: if an adapter given true mass and friction
cannot recover OOD task performance, representation learning cannot solve the bottleneck.

## Implementation status

The synthetic backend currently runs the complete smoke path:

1. validated ID/OOD physics manifests and parameter readback;
2. frozen visual-servo Push and Lift policies;
3. an engineering oracle-context gate on identical episode manifests;
4. balanced task-free interaction collection and causal sequence slicing;
5. response-only, privileged-distillation, and hybrid encoder training;
6. frozen-encoder supervised task-adapter training;
7. five-way OOD evaluation, representation probes, and adaptation-curve generation.

B0 and the engineering oracle are executable comparisons. Explicit SysID, domain-randomized,
per-task RMA, multi-task RMA, and TAM-like architectures are scaffolded but do not yet have
equal-budget training runners, so they are intentionally ineligible for a result table. Exact
fidelity gaps are recorded under `reports/baseline_audit/`.

Run a short end-to-end integration check after installation with:

```bash
./scripts/run_smoke_pipeline.sh
```

The critic-gated order and current supported/unsupported claims are preserved in `reports/`.
