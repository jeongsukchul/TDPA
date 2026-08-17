# MVP counterexample and edge-case review

Date: 2026-08-17  
Scope: settled synthetic MVP code, tests, and checked-in response-only smoke encoder/Push adapter.  
Decision: **REVISE before edge-case sign-off or any reusable-physics claim.** The synthetic
plumbing is usable, but non-finite fallback telemetry is incorrect and the checked-in smoke
representation does not robustly separate the required Lift physics axes.

## Evidence run

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q -p no:cacheprovider`: **61 passed**.
- Ran an in-memory, fixed-seed counterexample harness using
  `artifacts/smoke_encoder_student.pt` and, where applicable,
  `artifacts/smoke_push_adapter.pt`. It varied one factor at a time while keeping task,
  excitation, seed, and the other physics parameter fixed.
- The numeric latent results below are diagnostics of a one-epoch smoke checkpoint, not
  scientific estimates. They are useful here because they expose cases the current tests do
  not gate.

## Required cases

| Case | Observed result | Verdict and classification |
|---|---|---|
| **E1 — unobservable physics** | Under an eight-step no-contact `hold` history, Push with `(mass, friction)=(0.4, 0.1)` and `(2.4, 1.2)` produced exactly equal RGB-D, proprioception, response histories, and latent (`L2=0`, cosine approximately `1`). This follows the intended deployment boundary: physics is absent from observations until it affects motion (`src/tdpa/envs/base.py:86-112,123-133`). | **PASS with documented limitation / future work.** `HistoryEncoder` is deterministic and emits one point latent (`src/tdpa/models/history_encoder.py:73-104`), so it cannot express uncertainty and must not be described as identifying observationally equivalent physics. This report supplies the required limitation; an uncertainty head is optional unless calibrated identification is claimed. |
| **E2 — same mass, different friction** | A fixed Push probe at mass `1.0`, friction `0.2` versus `1.0` changed physical response (`max absolute difference 8.7409`) and the smoke latent (`L2=0.1028`, cosine `0.9871`). With nominal/proprio inputs held at zero, the Push adapter changed only negligibly: maximum head differences were residual `1.22e-4`, velocity scale `4.85e-4`, and stiffness `0.00345`. In Lift, mass `0.8`, friction `0.2` versus `1.0` flipped `ever_grasped` from false to true and changed response by `18.0`, but latent `L2` was only `2.20e-4` with cosine `0.9999999`. | **CLAIM-BLOCKING.** The simulator exposes the axis, but the checked-in representation/adapter does not demonstrate useful friction-conditioned Lift adaptation. Add paired held-out tests for both tasks and assert a predeclared response/latent-use or downstream-performance criterion. This is not a blocker to merging code explicitly labeled synthetic smoke plumbing. |
| **E3 — same friction, different mass** | A fixed Push probe at friction `0.55`, mass `0.4` versus `2.2` changed response (`10.3984`) and latent (`L2=0.1485`, cosine `0.9737`), but isolated adapter-head changes remained negligible (residual `1.68e-4`, velocity `6.95e-4`, stiffness `0.00353`). In Lift, the same mass pair at friction `0.55` flipped `ever_grasped` true-to-false and changed response by `18.0`, while latent `L2` was again only `2.20e-4` with cosine `0.9999999`. | **CLAIM-BLOCKING.** Same repair as E2. A physics-sensitive environment is not evidence that the frozen latent or adapter uses mass. |
| **E4 — different physics, similar short response** | For the E2 Push pair, all pre-contact observations and the first-contact response at step 8 were exactly equal. The next response diverged (`L2=0.3865`). This is expected from the dynamics: friction drag is zero while object velocity is initially zero and becomes informative only after motion (`src/tdpa/envs/push_env.py:35-52`). | **PASS as a counterexample / future test.** At least one additional transition or stronger excitation is required. Add a regression test and report performance as a function of post-contact history/excitation; do not claim exact inference from the first short response. |
| **E5 — different policy, same physics** | The held-out chirp changes the executed command trace at identical physics; the existing paired trace test passes. In the focused Push pair, same-physics cross-policy latent distance was `0.0306` (cosine `0.9988`), below the different-physics/same-policy distance `0.1487` (cosine `0.9736`). The full script now collects paired ID/policy-shift archives and invokes cross-policy retrieval with held-out-archive checking (`scripts/collect_interactions.sh:6-8`; `scripts/run_all_evaluations.sh:14-22`). | **PARTIAL PASS / CLAIM-BLOCKING until aggregate evidence exists.** One pair is encouraging, but no checked-in multi-seed diagnostic result or acceptance threshold establishes policy invariance. The diagnostic runner provides the mechanism; exercise it on paired held-out physics and compare against different-physics controls before making the claim. |
| **E6 — same task, different controller gain** | With identical Push motion actions and physics, stiffness `40` versus `180` changed the seven-dimensional execution command, physical response (`max absolute difference 14.5550`), and latent (`L2=0.1029`, cosine `0.9869`). Controller targets are deliberately included in encoder action history. | **CHARACTERIZED, but CLAIM-BLOCKING as an alternative explanation.** The encoder treats commanded controller gain as context, at a distance comparable to the tested physics shifts. This can be useful, but current data do not separate commanded gain, realized gain, and hidden controller mismatch. Add a gain-randomized diagnostic and a hidden realized-gain mismatch case; monitor the adapter-gain-to-latent feedback loop. |
| **E7 — contact-mode transition** | A controlled Lift state-machine probe produced `no contact/ungrasped/not slipped -> contact/grasped/not slipped -> contact/ungrasped/slipped -> contact/grasped/slipped` under high/low/high grip. Thus contact-to-grasp, slip/release, and re-grasp are possible. However, `slipped` is an episode-sticky flag (`src/tdpa/envs/lift_env.py:45-56`), so re-stick is not represented as a current mode; Push has no stick/slip state. No test evaluates latent or adapter behavior across these boundaries, and the deployment adapter has no explicit mode input. | **MERGE-BLOCKING for the required E7 sign-off; future work for smoke-only plumbing.** Add public-action transition tests for no-contact/contact, stick/slip/re-stick, distinguish current mode from ever-slipped metrics, and evaluate latent/command discontinuity and stability around each transition. Do not infer contact-rich robustness from the current smooth GRU/MLP path. |
| **E8 — adapter saturation** | Finite inputs of magnitude `1e20` produced finite neural-adapter outputs within configured bounds. The mapper clipped finite gain overflow and reported `saturated=true`; the combined repository test passes. | **PARTIAL PASS.** Finite saturation is covered. Non-finite handling below is a merge blocker. |

## Non-finite commands

The second safety layer successfully converted all tested non-finite nominal actions and adapter
outputs to finite, bounded execution commands (`src/tdpa/controllers/adapter_action_mapper.py:59-88`).
The neural adapter itself returned non-finite values for non-finite nominal/latent inputs because
the bounded sigmoid/tanh mappings do not sanitize their inputs
(`src/tdpa/models/physical_adapter.py:43-54`).

More importantly, each isolated fallback was misreported:

- non-finite nominal action only: finite mapped command, `saturated=false`;
- non-finite residual/velocity/gain corrections only: finite mapped command, `saturated=false`;
- wholly non-finite neural-adapter output: finite mapped command, `saturated=false`;
- finite out-of-range gains: finite clipped command, `saturated=true`.

The mapper sanitizes raw values before computing `saturated`, so it loses whether a fallback
occurred; nominal gripper clipping is also absent from the saturation expression. The existing
test combines non-finite fields with a finite `-999` residual and finite gain overflows, so its
`assert applied.saturated` does not isolate this failure (`tests/test_adapter_bounds.py:28-47`).

**Classification: MERGE-BLOCKING correctness/safety accounting.** Track raw finiteness and every
clipped channel, including nominal gripper, before sanitization. A non-finite fallback must either
fail closed or set an explicit invalid/saturated flag. Add isolated tests for nominal action,
residual, velocity, stiffness, damping, grip force, and non-finite model outputs. Until repaired,
the oracle gate's zero-saturation safety check can silently pass a fallback event.

## Empty and partial histories

- `DeploymentHistory.tensors()` on a new buffer fails loudly with
  `RuntimeError: History is empty` (`src/tdpa/data/history_buffer.py:27-29`). The adapted rollout
  safely bypasses the encoder at step zero and appends the first executed command before using
  history (`src/tdpa/evaluation/evaluate_task.py:47-66`).
- A one-step history was left-padded to eight steps with mask
  `[false, false, false, false, false, false, false, true]` and produced a finite latent. The
  packed-GRU implementation ignores masked padding, and the existing padding-invariance test
  passes (`src/tdpa/models/history_encoder.py:20-33`; `tests/test_encoder_shapes.py:86-99`).
- An all-false mask fails loudly with `ValueError: Every sequence needs at least one valid
  timestep`.
- `SequenceDataset` rejects an empty episode list, requires at least one past element and one
  causal action-response pair, and uses the joint mask for partial future windows
  (`src/tdpa/data/sequence_dataset.py:139-159,187-200,249-278`). Its partial-window and online/offline
  alignment tests pass.
- `DeploymentHistory(0)` is accepted by the constructor, silently discards appended elements via
  `deque(maxlen=0)`, and later raises `History is empty`.

**Classification:** partial-history behavior **passes**. Empty/all-masked failure behavior is
acceptable. Positive validation of `DeploymentHistory.history_length` is **future hardening** and
should mirror `SequenceDataset`; promote it to merge-blocking only if history length becomes a
user-supplied runtime setting.

## Blocking summary

### Merge-blocking defects

1. Non-finite mapper fallbacks and nominal gripper clipping are not reflected in saturation/safety
   telemetry.
2. E7 lacks a current contact-mode model and automated transition/stability coverage required for
   edge-case sign-off.

### Evidence gates before research claims

1. E2/E3 must demonstrate useful one-axis separation and downstream latent use on both Push and
   Lift; the checked-in Lift smoke result is effectively invariant despite a grasp-outcome flip.
2. E5 needs aggregate paired cross-policy retrieval with uncertainty and a predeclared criterion.
3. E6 must be separated from object physics and tested for closed-loop gain/latent feedback.

### Non-blocking future work

- uncertainty modeling for E1 if calibrated identification is desired;
- history/excitation sensitivity for E4;
- constructor validation for zero-length deployment history.

## Focused blocker re-audit — 2026-08-17

Re-ran `tests/test_adapter_bounds.py`, `tests/test_contact_transitions.py`, and the full suite:
**7 focused tests passed; 64 total tests passed.**

1. **Non-finite fallback telemetry: RESOLVED.** The mapper now detects raw non-finite nominal
   and correction values before sanitization (`src/tdpa/controllers/adapter_action_mapper.py:48-54,68-77`)
   and includes that condition in `saturated` (`:96-102`). Focused runtime checks confirmed
   finite mapped commands and `saturated=true` independently for a non-finite nominal action,
   velocity scale, residual, stiffness, damping, and grip force. The isolated regression test
   covers representative nominal, velocity, residual, and grip-force cases
   (`tests/test_adapter_bounds.py:59-65`). **Residual narrow issue:** a finite nominal gripper
   command outside `[-1, 1]` is clipped but still reports `saturated=false`, because nominal
   gripper clipping is not part of the saturation expression. This no longer blocks the stated
   non-finite repair, but remains safety-telemetry hardening before claiming every clipped channel
   is counted.

2. **E7 contact-transition coverage: RESOLVED for synthetic edge-case sign-off.** Lift now has
   an explicit current `contact_mode` with `no_contact`, `stick`, and `slip` states, separate from
   the episode-sticky `slipped` metric (`src/tdpa/envs/lift_env.py:22-27,47-62`). Automated tests
   exercise Push `free-space -> contact -> free-space` and Lift `stick -> slip -> re-stick`
   (`tests/test_contact_transitions.py:9-33`). Latent/adapter stability across mode boundaries is
   still a **research-evidence gate**, but not a remaining code merge blocker for smoke plumbing.

**Re-audit decision:** the two requested blockers are cleared for the explicitly limited synthetic
MVP. Preserve the finite nominal-gripper telemetry gap as follow-up work and retain all earlier
E2/E3/E5/E6 claim restrictions.
