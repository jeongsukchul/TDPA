# ACCEPT_WITH_LIMITED_CLAIM

## Scope of acceptance

Accept the repository as a runnable synthetic research-infrastructure MVP. Do not treat this as completion of the publication-grade robotics project or as evidence for the shared-representation hypothesis. The current complexity is justified as modular experiment scaffolding; the hybrid representation, RGB-D path, and learned adaptation method are not yet empirically justified as the preferred method.

Independent checks on the current filesystem found 64 passing tests, a clean Ruff run, a clean dependency check, and a reproducible in-memory 1,680-rollout oracle gate with manifest hash `ad8141bdb5b0971f08fc476ec1ff776339208297d4e2a53dd5170cdedc186248`. The latest temporal, command-history, non-finite telemetry, contact-transition, policy-shift, response-normalization, and student-only checkpoint fixes are present.

## Supported claims

- The deterministic synthetic Push/Lift backend, configuration system, collection path, V1/V2/V3 training interfaces, bounded adapter, evaluation entry points, and smoke workflow are runnable.
- Train, ID, OOD-mass, OOD-friction, OOD-composition, and paired policy-shift supports are generated deterministically and validated; the policy-shift path changes executed commands rather than only metadata.
- History windows use strict-past observations and executed seven-dimensional commands, pair future actions with their causal responses, and match online warm-up semantics.
- The deployable learned path structurally loads a student-only encoder and receives RGB-D with the goal channel masked, proprioception, and executed-command history. No direct mass, friction, privileged object state, contact label, F/T, split, task ID, or probe ID path into the student or adapter was found.
- On the synthetic surrogate, the frozen hand-coded nominal controller is competent at nominal physics, degrades on selected OOD shifts, and a bounded hand-designed perfect-context engineering oracle improves average OOD success on identical manifests without reported saturation or force-limit violations.
- These results validate synthetic experiment plumbing and the existence of a surrogate control-interface sanity check only.

## Unsupported claims

- Task-free action-response pretraining reduces task-specific physical-adaptation data.
- The learned latent is usefully physics-sensitive, task/probe/policy invariant, or transferable across Push and Lift, behavior policies, MuJoCo environments, or robots.
- V1, V2, or V3 is better than the others; hybrid complexity is necessary; or RGB-D improves over proprioception alone.
- The learned adapter uses the latent rather than nominal action, proprioception, trajectory phase, controller commands, or other shortcuts.
- The method matches or outperforms explicit SysID, domain randomization, per-task RMA, multi-task RMA, TAM/TAM-like adaptation, or a learned privileged oracle at equal data, capacity, tuning, and control authority.
- The synthetic force/contact model establishes physical safety, MuJoCo/robosuite benchmark validity, real-robot robustness, uniform low/high OOD recovery, or general data/compute/deployment efficiency.
- The repository is a completed publication-grade realization of `Project_Pipeline.md`.

## Remaining blockers

1. `make_env` rejects the robosuite backend. Physics “readback” currently reads the same synthetic parameters used to construct hand-written dynamics, so the mandatory MuJoCo oracle-first gate has not occurred.
2. The nominal policies are fixed visual servos over one deterministic reset, not frozen BC/Diffusion policies validated over seeded object, robot, target, and nuisance variation.
3. Only B0 and the hand-designed engineering oracle are end-to-end. SysID, domain-randomized policy, per-task RMA, multi-task RMA, and TAM-like code remain unevaluated scaffolds; learned B6 is absent. Consequently, observation, data, capacity, tuning, and controller-authority fairness is not established.
4. `adaptation_curve.py` runs only `pretrained_shared`; it does not use one immutable set of nested task-data subsets across the required methods. Pretraining, privileged-label collection, optimizer-step, wall-time, inference-latency, tuning, and task-specific parameter ledgers remain incomplete.
5. Checked smoke evidence does not show useful latent dependence: correct, shuffled, and zero-latent response errors are nearly equal, and no downstream zero-latent or physics-mismatched-latent control exists. The available V1/V3 artifacts are not converged comparative results.
6. Representation probes are still partly resubstitution diagnostics, and no three-seed held-out physics/action result with uncertainty and predeclared thresholds establishes mass/friction information, cross-policy consistency, or task invariance.
7. Adapter labels come from oracle-controlled histories while deployment is learned-adapter closed loop, leaving covariate shift untested. The Push loss also includes fixed disabled-output terms, which pollutes its reported loss even though those terms do not train the disabled heads.
8. Directional low/high OOD results, confidence intervals, locked validation/tuning manifests, and complete manifest/config/checkpoint provenance are missing. The workspace is not a Git repository and artifacts report `git_commit: unversioned`.
9. The expanded stiffness/damping/grip interface was introduced after the narrower MVP plan, and Push recovery depends on stiffness. That dependence and the failed smaller interface must remain explicit rather than being presented as predeclared evidence.

## Decisive next test

Implement audited robosuite/MuJoCo Push and Lift tasks with direct body/geom mass and both-side friction readback, randomized but method-paired initial conditions, and strong frozen nominal policies. Lock one three-seed, 20-episode-per-cell manifest and rerun B0 against the bounded perfect-context oracle, reporting low/high directions, paired uncertainty, force/saturation traces, and full provenance. If nominal competence, informative OOD degradation, and safe oracle recovery do not all hold for both tasks, pivot the benchmark or correction interface before any further representation work. If they do hold, the next research-claim gate is the equal-budget nested adaptation curve against shared-scratch, learned oracle, SysID, per-task RMA, and multi-task RMA with zero/shuffled-latent controls.
