# MuJoCo / robosuite smoke sprint

## Outcome and boundary

Add an explicitly selected `robosuite` backend for Push and Lift that can be installed in a
reproducible Conda environment and exercised without training. The sprint must prove that both
tasks compile, reset, render, step, make contact, expose only the existing deployment observation
keys, set real MuJoCo model parameters, read those parameters back by resolved body/geom ID, and
reproduce a logged initial state from `(task, seed, episode_index)`.

This is a backend smoke gate, not the physical oracle-performance gate. Keep `synthetic` as the
default backend and preserve the existing synthetic result. Do not run or claim BC/Diffusion
training, encoder/adapter training, learned baselines, OOD recovery, or adaptation-data efficiency.

## Implementation order and exact integration points

1. **Pin the runnable environment.** Add `environment.yml` with `name: TDPA`, Python 3.11, pip,
   and an editable install of `.[dev,simulation]`. Pin the simulation API used by the code
   (`robosuite==1.5.1`, whose package requires `mujoco>=3.2.3`; constrain MuJoCo to one tested
   minor range rather than leaving both unbounded). Keep heavy simulation dependencies optional
   in `pyproject.toml`; importing `tdpa` or using the synthetic backend must not import robosuite.
   Record the resolved Python, robosuite, MuJoCo, NumPy, and renderer versions in smoke output.

2. **Define one backend-neutral contract.** In `src/tdpa/envs/base.py`, add a structural
   `ManipulationEnv` protocol and immutable `PhysicsReadback` / `ResetState` records. The protocol
   must cover the members already consumed by callers: `config`, `horizon`, `image_size`,
   `force_limit`, `reset()`, `step(action, controller)`, `metrics()`, `read_physics()`, and
   `close()`. Add the same `read_physics()` and no-op `close()` methods to
   `SyntheticManipulationEnv`; do not weaken its tests or observation contract.

3. **Build fixed-topology robosuite tasks.** Add
   `src/tdpa/envs/robosuite_tasks.py` containing a shared single-Panda tabletop task plus
   `TDPAPush` and `TDPALiftTransport`. Use robosuite 1.5.1's `ManipulationEnv`, `TableArena`, a
   fixed-size `BoxObject`, `ManipulationTask`, and a fixed target site. Do not rely on the stock
   `Lift` object's randomized size or its lift-only success criterion. Use the same object XML
   topology, names, material, texture, geom order, and visual parameters at every mass/friction
   value. `TDPAPush` succeeds on planar target error; `TDPALiftTransport` requires the object to
   be grasped/lifted and within the 3-D transport target tolerance.

4. **Adapt robosuite to the TDPA API.** Add `src/tdpa/envs/robosuite_backend.py` with
   `RobosuiteManipulationEnv`. Instantiate the custom task directly (no global registration is
   required) using a Panda and robosuite's `OSC_POSITION` part-controller config; assert the
   resulting normalized action dimension is exactly four: Cartesian xyz plus gripper. Configure
   `initialization_noise=None`, `hard_reset=False`, `control_freq` and `horizon` from YAML,
   `use_object_obs=False`, and headless offscreen RGB-D from `agentview`.

   Convert robosuite observations into exactly:

   - `rgbd`: float32 CHW, three normalized RGB channels plus one metric/normalized depth channel;
   - `proprio`: the existing 10-D deployable schema
     `[eef_position(3), finite-difference eef_velocity(3), gripper_width(1), last_xyz_command(3)]`.

   Object pose, target pose, contact state, force/torque, physics, names, IDs, task/split labels,
   and reset metadata must never be returned by `reset()` or `step()`. Privileged values may
   appear only in the explicit smoke/readback record. Convert robosuite's 4-tuple step result to
   the repository's `(observation, reward, terminated, truncated, info)` convention. Compute
   contact force from MuJoCo contacts involving the declared task geoms with
   `mujoco.mj_contactForce`, and keep physics out of `info`.

5. **Set and independently read real physics.** Resolve the object's root body through
   `sim.model.body_name2id(task_object.root_body)` and resolve every declared contact geom through
   `sim.model.geom_name2id(name)`. Never assume numeric IDs or use substring-only matching.
   After compilation, mutate:

   - object `body_mass` to requested mass and scale `body_inertia` by the same mass ratio;
   - Push sliding friction on every object collision geom **and** the table collision geom;
   - Lift sliding friction on every object collision geom **and** both finger-pad collision geoms.

   Preserve fixed torsional/rolling friction components unless the config explicitly declares
   them. Recompute MuJoCo constants / forward state after mutation using the underlying
   `sim.model._model` and `sim.data._data` APIs required by robosuite 1.5.1. Fail construction if
   any expected body or either side of a contact pair resolves to an empty/duplicate set.

   `read_physics()` must read fresh values from `body_mass`, `body_inertia`, and `geom_friction`;
   it must not return the requested `Physics` object. Its record must include requested values,
   actual values, body/geom names and IDs, all three friction components for each geom, and a
   stable asset/topology signature. The verifier compares requested and actual values with a
   declared numerical tolerance and fails nonzero on mismatch. Reapply and re-verify after any
   operation that recompiles the model.

6. **Make resets index-addressable and paired.** Add
   `src/tdpa/envs/reset_randomization.py`. Derive a local NumPy generator with `SeedSequence`
   from stable integer task codes, `seed`, and `episode_index`; never use Python's randomized
   `hash()` or rely on global `np.random` call order. Resolve and log robot joint qpos, zero qvel,
   object position/quaternion, and target position before constructing either method. Keep the
   sampling bounds task-specific and independent of mass, friction, split, method, or construction
   order. Apply the resolved state after robosuite reset, call forward without a settling step,
   and expose a privileged `reset_fingerprint()` for verification only.

   Extend `make_env()` in `src/tdpa/envs/make_env.py` with `episode_index=0`; lazy-import and
   return `RobosuiteManipulationEnv` only for `backend="robosuite"`. Broaden the return annotation
   to the protocol. Existing calls and default YAML behavior must remain synthetic and unchanged.

7. **Configure without changing scientific defaults.** Extend `configs/env/push.yaml` and
   `configs/env/lift.yaml` with a nested `robosuite:` section for robot, controller, table/object
   dimensions, camera, reset bounds, control frequency, fixed torsional/rolling friction, geom
   names, and force limits. Leave top-level `backend: synthetic`. The current shared physics ranges
   may be exercised for write/readback, but label them uncalibrated for these MuJoCo assets; a
   later oracle gate needs task-specific physically plausible ranges.

8. **Add two no-training CLIs.** Update `src/tdpa/tools/verify_physics.py` to accept
   `--backend {synthetic,robosuite}`, pass the backend explicitly, use `read_physics()`, close every
   environment in `finally`, and emit a backend/versioned JSON artifact. Add
   `src/tdpa/tools/smoke_robosuite.py` (and a `tdpa-smoke-robosuite` entry point) that runs both
   tasks over two physics values and two reset indices, checks observation/action/metric finiteness,
   executes a short bounded task-specific probe, requires at least one relevant contact in the
   contact probe, verifies readback before and after stepping, verifies replayed reset fingerprints,
   and writes `artifacts/robosuite_smoke.json`. Probe object state is privileged smoke scaffolding;
   it is not a deployable policy or result-table method.

9. **Cover the integration.** Add `tests/test_robosuite_backend.py`, marked `simulation`, with:

   - Push and Lift construction/reset/step/close and exact `{rgbd, proprio}` deployment keys;
   - finite shapes/dtypes and a four-dimensional action-space assertion;
   - requested-versus-live body mass and every relevant geom's three-component friction readback;
   - non-empty object/table and object/left-pad/right-pad ID sets;
   - unchanged asset/topology signature across physics values;
   - identical reset state/fingerprint for equal `(task, seed, index)` across physics and
     construction order, and a changed fingerprint for a different index;
   - identical initial RGB-D/proprio for equal resolved reset state at different physics values
     (exact state equality; image equality or a documented renderer tolerance);
   - a relevant MuJoCo contact and finite contact force after the smoke probe;
   - a negative test proving physics/readback metadata is rejected by the frozen deployment-policy
     guard.

   Use `pytest.importorskip("robosuite")` so the normal lightweight test suite remains valid, but
   run the simulation marker as a required test inside `TDPA`.

## Smoke acceptance gate

The sprint passes only if all of the following hold in the `TDPA` environment:

- the full existing suite still passes;
- both custom tasks compile, reset, render RGB-D, execute at least ten finite actions, and close;
- every requested mass and every geom-side sliding friction value is read from the live MuJoCo
  model within tolerance before and after stepping;
- object/table contact is observed for Push and object/both-finger contact is observed for Lift;
- repeated `(task, seed, episode_index)` produces identical resolved robot/object/target state,
  independent of physics and environment construction order;
- different reset indices produce variation in at least object or target state;
- initial deployment observations do not reveal mass/friction under a paired reset;
- the smoke JSON records versions, renderer, config hashes, resolved reset states, resolved
  body/geom names and IDs, requested/read-back physics, steps, contact counts, and any failure;
- no training command or checkpoint is created.

## Commands to hand to the user

```bash
conda env update -n TDPA -f environment.yml --prune
conda activate TDPA

python -m pytest -q
MUJOCO_GL=egl python -m pytest -q -m simulation tests/test_robosuite_backend.py

MUJOCO_GL=egl python -m tdpa.tools.verify_physics \
  --backend robosuite --task push --seed 7 --count 2 \
  --output artifacts/robosuite_push_physics.json
MUJOCO_GL=egl python -m tdpa.tools.verify_physics \
  --backend robosuite --task lift --seed 7 --count 2 \
  --output artifacts/robosuite_lift_physics.json

MUJOCO_GL=egl python -m tdpa.tools.smoke_robosuite \
  --task all --seed 7 --steps 10 --output artifacts/robosuite_smoke.json
```

If EGL is unavailable on a CPU-only Linux host, repeat only the rendering commands with
`MUJOCO_GL=osmesa`; do not silently disable RGB-D to obtain a pass.

## Must remain unsupported after this sprint

- The existing `FrozenNominalPolicy` on robosuite RGB-D: it locates synthetic marker channels and
  is not a BC/Diffusion checkpoint.
- `oracle_gate.py --backend robosuite` and any OOD recovery claim: the current engineering oracle
  assumes synthetic response equations.
- Runtime stiffness, damping, and physical grip-force targets: robosuite's fixed
  `OSC_POSITION`/normalized gripper action does not implement the current controller dictionary.
  The backend must reject non-nominal requests rather than silently ignore them.
- Interaction-archive collection and encoder/adapter training on this backend until the MuJoCo
  privileged/response schemas, units, causal timing, and command semantics receive their own
  tests and version bump.
- A grasp-lift-transport performance claim from the privileged smoke probe, any force-safety claim,
  calibrated OOD ranges, learned-baseline comparison, representation transfer claim, or paper
  result.

The next gate after this sprint is to train or import strong frozen nominal BC/Diffusion policies,
implement a real bounded controller interface (including explicit gain/grip semantics), and only
then run the paired 3-seed x 20-episode MuJoCo B0-versus-oracle evaluation.
