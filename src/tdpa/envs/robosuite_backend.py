from __future__ import annotations

import hashlib
import json
from importlib import metadata
from typing import Any

import mujoco
import numpy as np
import robosuite as suite

from tdpa.envs.base import GeomPhysicsReadback, Physics, PhysicsReadback
from tdpa.envs.reset_randomization import ResetState, sample_reset_state
from tdpa.envs.robosuite_tasks import TDPALiftTransport, TDPAPush


def _real_depth_map(sim: Any, depth_map: np.ndarray) -> np.ndarray:
    """Convert MuJoCo's normalized depth buffer without importing optional h5py utilities."""
    depth = np.asarray(depth_map)
    if np.any(depth < 0.0) or np.any(depth > 1.0):
        raise RuntimeError("MuJoCo depth buffer is outside [0, 1]")
    extent = float(sim.model.stat.extent)
    far = float(sim.model.vis.map.zfar) * extent
    near = float(sim.model.vis.map.znear) * extent
    return near / (1.0 - depth * (1.0 - near / far))


class RobosuiteManipulationEnv:
    """TDPA's narrow, deployment-safe wrapper around a live MuJoCo model."""

    def __init__(
        self,
        task: str,
        config: dict[str, Any],
        physics: Physics,
        *,
        seed: int = 0,
        episode_index: int = 0,
    ) -> None:
        if task not in {"push", "lift"}:
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        self.config = config
        self.physics = physics
        self.seed = int(seed)
        self.episode_index = int(episode_index)
        rs_config = config.get("robosuite")
        if not isinstance(rs_config, dict):
            raise TypeError(f"configs/env/{task}.yaml is missing a robosuite section")
        self.robosuite_config = rs_config
        self.horizon = int(config.get("episode_length", 64))
        self.image_size = int(rs_config.get("image_size", 64))
        self.force_limit = float(config.get("force_limit", 0.0))
        self.dt = 1.0 / float(rs_config.get("control_frequency", 20))
        self.t = 0
        self.last_action = np.zeros(4, dtype=np.float32)
        self._last_eef_position: np.ndarray | None = None
        self._last_observation: dict[str, np.ndarray] | None = None
        self._reset_state: ResetState | None = None

        controller = suite.load_composite_controller_config(
            robot=str(rs_config.get("robot", "Panda"))
        )
        arm = controller["body_parts"]["right"]
        arm["type"] = "OSC_POSITION"
        arm["input_min"] = -1
        arm["input_max"] = 1
        delta = float(rs_config.get("position_delta_limit", 0.05))
        arm["output_min"] = [-delta, -delta, -delta]
        arm["output_max"] = [delta, delta, delta]
        arm["impedance_mode"] = "fixed"
        execution_config = rs_config.get("execution", {})
        execution_nominal = (
            execution_config.get("nominal", {}) if isinstance(execution_config, dict) else {}
        )
        nominal_stiffness = float(execution_nominal.get("stiffness", 100.0))
        nominal_damping = float(execution_nominal.get("damping", 15.0))
        arm["kp"] = nominal_stiffness
        arm["damping_ratio"] = nominal_damping / (2.0 * np.sqrt(nominal_stiffness))

        task_type = TDPAPush if task == "push" else TDPALiftTransport
        target = tuple(float(value) for value in rs_config["target_position"])
        half_size = tuple(float(value) for value in rs_config.get("object_half_size", [0.025] * 3))
        friction = tuple(
            float(value) for value in rs_config.get("base_friction", [1.0, 0.005, 0.0001])
        )
        self.raw = task_type(
            robots=str(rs_config.get("robot", "Panda")),
            controller_configs=controller,
            initialization_noise=None,
            table_full_size=tuple(rs_config.get("table_full_size", [0.8, 0.8, 0.05])),
            table_friction=friction,
            use_camera_obs=True,
            use_object_obs=False,
            reward_scale=1.0,
            reward_shaping=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_gpu_device_id=int(rs_config.get("render_gpu_device_id", -1)),
            control_freq=int(rs_config.get("control_frequency", 20)),
            lite_physics=False,
            horizon=self.horizon,
            ignore_done=False,
            hard_reset=False,
            camera_names=str(rs_config.get("camera_name", "agentview")),
            camera_heights=self.image_size,
            camera_widths=self.image_size,
            camera_depths=True,
            seed=self.seed,
            target_position=target,
            success_tolerance=float(rs_config.get("success_tolerance", 0.06)),
            object_half_size=half_size,
        )
        if int(self.raw.action_dim) != 4:
            self.raw.close()
            raise RuntimeError(
                f"OSC_POSITION backend must expose 4 actions, got {self.raw.action_dim}"
            )
        # Construction performs one noise-free reset, yielding Panda's declared
        # nominal configuration. Preserve it before deterministic resets bypass
        # robosuite's robot initializer.
        self._nominal_robot_qpos = self._robot_joint_qpos()
        self.raw.deterministic_reset = True
        self._object_half_height = half_size[2]
        self._resolve_model_ids()
        model = self.raw.sim.model
        self._base_mass = float(model.body_mass[self._body_id])
        self._base_inertia = np.asarray(model.body_inertia[self._body_id], dtype=np.float64).copy()
        self._setup_execution_controller()
        self._topology_signature = self._compute_topology_signature()
        self._apply_physics()
        self._apply_execution_controller(None)

    @property
    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.raw.action_spec
        return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)

    def _resolve_names(self, names: list[str] | tuple[str, ...], *, kind: str) -> tuple[int, ...]:
        if not names or len(set(names)) != len(names):
            raise RuntimeError(f"Expected a non-empty, unique {kind} geom list, got {names}")
        ids = tuple(int(self.raw.sim.model.geom_name2id(name)) for name in names)
        if any(geom_id < 0 for geom_id in ids) or len(set(ids)) != len(ids):
            raise RuntimeError(f"Could not uniquely resolve all {kind} geoms: {names}")
        return ids

    def _resolve_model_ids(self) -> None:
        model = self.raw.sim.model
        self._body_name = str(self.raw.cube.root_body)
        self._body_id = int(model.body_name2id(self._body_name))
        if self._body_id < 0:
            raise RuntimeError(f"Could not resolve object body {self._body_name}")
        self._object_geom_names = tuple(str(name) for name in self.raw.cube.contact_geoms)
        self._object_geom_ids = self._resolve_names(self._object_geom_names, kind="object")
        if self.task == "push":
            self._counterpart_geom_names = tuple(
                str(name)
                for name in self.robosuite_config.get("table_contact_geoms", ["table_collision"])
            )
        else:
            gripper = self.raw.robots[0].gripper["right"]
            left = list(gripper.important_geoms["left_fingerpad"])
            right = list(gripper.important_geoms["right_fingerpad"])
            if not left or not right:
                raise RuntimeError("Both Panda finger-pad geom groups are required")
            self._counterpart_geom_names = tuple(str(name) for name in left + right)
        self._counterpart_geom_ids = self._resolve_names(
            self._counterpart_geom_names, kind="contact-counterpart"
        )
        if set(self._object_geom_ids) & set(self._counterpart_geom_ids):
            raise RuntimeError("Object and counterpart contact geom sets must be disjoint")
        self._target_site_id = int(model.site_name2id("tdpa_target"))
        if self._target_site_id < 0:
            raise RuntimeError("Could not resolve tdpa_target site")

    def _compute_topology_signature(self) -> str:
        model = self.raw.sim.model
        topology = {
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "bodies": [
                (model.body_id2name(index), int(model.body_parentid[index]))
                for index in range(model.nbody)
            ],
            "geoms": [
                (
                    model.geom_id2name(index),
                    int(model.geom_type[index]),
                    int(model.geom_bodyid[index]),
                    np.asarray(model.geom_size[index]).round(12).tolist(),
                )
                for index in range(model.ngeom)
            ],
        }
        encoded = json.dumps(topology, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _setup_execution_controller(self) -> None:
        execution = self.robosuite_config.get("execution")
        if not isinstance(execution, dict):
            raise TypeError("robosuite.execution configuration is required")
        nominal = execution.get("nominal")
        bounds = execution.get("bounds")
        if not isinstance(nominal, dict) or not isinstance(bounds, dict):
            raise TypeError("robosuite.execution requires nominal and bounds mappings")
        keys = ("velocity_scale", "stiffness", "damping", "grip_force")
        self._execution_nominal: dict[str, float] = {}
        self._execution_bounds: dict[str, tuple[float, float]] = {}
        for key in keys:
            interval = np.asarray(bounds.get(key), dtype=np.float64)
            if interval.shape != (2,) or not np.isfinite(interval).all():
                raise ValueError(f"Execution bound {key} must be two finite values")
            low, high = float(interval[0]), float(interval[1])
            value = float(nominal.get(key, np.nan))
            if low > high or not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid nominal/bounds for execution field {key}")
            self._execution_bounds[key] = (low, high)
            self._execution_nominal[key] = value
        osc = self.raw.robots[0].part_controllers["right"]
        stiffness_low, stiffness_high = self._execution_bounds["stiffness"]
        if stiffness_low < float(np.max(osc.kp_min[:3])) or stiffness_high > float(
            np.min(osc.kp_max[:3])
        ):
            raise ValueError("Configured stiffness bounds exceed robosuite OSC limits")
        if self._execution_bounds["damping"][0] < 0:
            raise ValueError("Damping bounds must be non-negative")
        self._gripper_actuator_ids = tuple(
            int(index) for index in self.raw.robots[0]._ref_joint_gripper_actuator_indexes["right"]
        )
        if len(self._gripper_actuator_ids) != 2:
            raise RuntimeError("Panda execution expects two gripper actuators")
        self._last_execution_requested = dict(self._execution_nominal)
        self._last_execution_applied = dict(self._execution_nominal)
        self._controller_saturated = False

    @staticmethod
    def _execution_scalar(value: object, default: float) -> tuple[float, bool]:
        try:
            scalar = float(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return default, True
        if not np.isfinite(scalar):
            return default, True
        return scalar, False

    def _apply_execution_controller(self, controller: dict[str, float] | None) -> float:
        requested_input = controller or {}
        unknown = set(requested_input) - set(self._execution_nominal)
        if unknown:
            raise ValueError(f"Unknown execution controller fields: {sorted(unknown)}")
        requested: dict[str, float] = {}
        applied: dict[str, float] = {}
        saturated = False
        for key, default in self._execution_nominal.items():
            value, invalid = self._execution_scalar(requested_input.get(key, default), default)
            low, high = self._execution_bounds[key]
            clipped = float(np.clip(value, low, high))
            requested[key] = value
            applied[key] = clipped
            saturated = saturated or invalid or clipped != value

        osc = self.raw.robots[0].part_controllers["right"]
        if np.asarray(osc.kp).shape != (6,) or np.asarray(osc.kd).shape != (6,):
            raise RuntimeError("Unexpected robosuite OSC gain shape")
        osc.kp[:3] = applied["stiffness"]
        osc.kd[:3] = applied["damping"]
        force = applied["grip_force"]
        ids = np.asarray(self._gripper_actuator_ids, dtype=np.int64)
        self.raw.sim.model.actuator_forcerange[ids, 0] = -force
        self.raw.sim.model.actuator_forcerange[ids, 1] = force
        self._last_execution_requested = requested
        self._last_execution_applied = applied
        self._controller_saturated = saturated
        return applied["velocity_scale"]

    def controller_readback(self) -> dict[str, object]:
        """Explicit command/readback evidence, excluded from deployment observations."""
        osc = self.raw.robots[0].part_controllers["right"]
        ids = np.asarray(self._gripper_actuator_ids, dtype=np.int64)
        return {
            "requested": dict(self._last_execution_requested),
            "applied": dict(self._last_execution_applied),
            "translational_kp": np.asarray(osc.kp[:3], dtype=np.float64).tolist(),
            "translational_kd": np.asarray(osc.kd[:3], dtype=np.float64).tolist(),
            "gripper_force_ranges": np.asarray(
                self.raw.sim.model.actuator_forcerange[ids], dtype=np.float64
            ).tolist(),
            "saturated": self._controller_saturated,
        }

    def _apply_physics(self) -> None:
        model = self.raw.sim.model
        ratio = self.physics.mass / self._base_mass
        model.body_mass[self._body_id] = self.physics.mass
        model.body_inertia[self._body_id] = self._base_inertia * ratio
        for geom_id in self._object_geom_ids + self._counterpart_geom_ids:
            model.geom_friction[geom_id, 0] = self.physics.friction
        mujoco.mj_setConst(model._model, self.raw.sim.data._data)
        mujoco.mj_forward(model._model, self.raw.sim.data._data)

    def read_physics(self) -> PhysicsReadback:
        model = self.raw.sim.model

        def read_geom(name: str, geom_id: int) -> GeomPhysicsReadback:
            values = tuple(float(value) for value in model.geom_friction[geom_id])
            return GeomPhysicsReadback(name=name, geom_id=geom_id, friction=values)

        return PhysicsReadback(
            backend="robosuite",
            requested_mass=self.physics.mass,
            requested_friction=self.physics.friction,
            actual_mass=float(model.body_mass[self._body_id]),
            body_name=self._body_name,
            body_id=self._body_id,
            body_inertia=tuple(float(value) for value in model.body_inertia[self._body_id]),
            object_geoms=tuple(
                read_geom(name, geom_id)
                for name, geom_id in zip(self._object_geom_names, self._object_geom_ids)
            ),
            counterpart_geoms=tuple(
                read_geom(name, geom_id)
                for name, geom_id in zip(self._counterpart_geom_names, self._counterpart_geom_ids)
            ),
            topology_signature=self._topology_signature,
        )

    def _robot_joint_qpos(self) -> np.ndarray:
        indexes = np.asarray(self.raw.robots[0]._ref_joint_pos_indexes, dtype=np.int64)
        return np.asarray(self.raw.sim.data.qpos[indexes], dtype=np.float64).copy()

    def _apply_reset_state(self, state: ResetState) -> None:
        robot = self.raw.robots[0]
        position_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
        velocity_indexes = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)
        self.raw.sim.data.qpos[position_indexes] = np.asarray(state.robot_qpos)
        self.raw.sim.data.qvel[velocity_indexes] = 0.0
        object_qpos = np.asarray(state.object_position + state.object_quaternion)
        self.raw.sim.data.set_joint_qpos(self.raw.cube.joints[0], object_qpos)
        self.raw.sim.data.set_joint_qvel(self.raw.cube.joints[0], np.zeros(6))
        self.raw.sim.model.site_pos[self._target_site_id] = np.asarray(state.target_position)
        mujoco.mj_forward(self.raw.sim.model._model, self.raw.sim.data._data)

    def reset(self) -> dict[str, np.ndarray]:
        self.raw.reset()
        self._apply_physics()
        self._apply_execution_controller(None)
        state = sample_reset_state(
            self.task,
            seed=self.seed,
            episode_index=self.episode_index,
            robot_qpos=self._nominal_robot_qpos,
            table_height=float(self.raw.table_offset[2]),
            object_half_height=self._object_half_height,
            config=self.robosuite_config,
        )
        self._apply_reset_state(state)
        self._reset_state = state
        self.t = 0
        self.last_action[:] = 0.0
        self._last_eef_position = None
        # Prime the offscreen buffer after direct qpos / site mutation. Some EGL
        # drivers otherwise return the previous framebuffer on the first read.
        self.raw._get_observations(force_update=True)
        raw_observation = self.raw._get_observations(force_update=True)
        self._last_observation = self._deployment_observation(raw_observation)
        return {key: value.copy() for key, value in self._last_observation.items()}

    def reset_fingerprint(self) -> str:
        if self._reset_state is None:
            raise RuntimeError("reset() must be called before reset_fingerprint()")
        return self._reset_state.fingerprint()

    def reset_state(self) -> ResetState:
        if self._reset_state is None:
            raise RuntimeError("reset() must be called before reset_state()")
        return self._reset_state

    def _deployment_observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        camera = str(self.robosuite_config.get("camera_name", "agentview"))
        rgb = np.asarray(observation[f"{camera}_image"], dtype=np.float32) / 255.0
        depth = _real_depth_map(self.raw.sim, observation[f"{camera}_depth"])
        depth_limit = float(self.robosuite_config.get("depth_limit", 2.0))
        depth = np.clip(np.asarray(depth, dtype=np.float32) / depth_limit, 0.0, 1.0)
        if depth.ndim == 2:
            depth = depth[..., None]
        rgbd = np.concatenate([rgb, depth], axis=-1).transpose(2, 0, 1).astype(np.float32)

        eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
        if self._last_eef_position is None:
            velocity = np.zeros(3, dtype=np.float32)
        else:
            velocity = (eef - self._last_eef_position) / self.dt
        self._last_eef_position = eef.copy()
        gripper_qpos = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
        gripper_width = np.array([float(np.sum(np.abs(gripper_qpos)))], dtype=np.float32)
        proprio = np.concatenate([eef, velocity, gripper_width, self.last_action[:3]]).astype(
            np.float32
        )
        return {"rgbd": rgbd, "proprio": proprio}

    def observation(self) -> dict[str, np.ndarray]:
        if self._last_observation is None:
            raise RuntimeError("reset() must be called before observation()")
        return {key: value.copy() for key, value in self._last_observation.items()}

    def step(
        self, action: np.ndarray, controller: dict[str, float] | None = None
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (4,):
            raise ValueError(f"Expected action shape (4,), got {action_array.shape}")
        if not np.isfinite(action_array).all():
            raise ValueError("Action must contain only finite values")
        velocity_scale = self._apply_execution_controller(controller)
        action_array = action_array.copy()
        action_array[:3] *= velocity_scale
        low, high = self.action_spec
        clipped_action = np.clip(action_array, low, high)
        self._controller_saturated = self._controller_saturated or bool(
            np.any(clipped_action != action_array)
        )
        action_array = clipped_action
        self.last_action = action_array.copy()
        raw_observation, reward, done, _ = self.raw.step(action_array)
        self.t += 1
        self._last_observation = self._deployment_observation(raw_observation)
        success = bool(self.raw._check_success())
        terminated = success
        truncated = bool(done and not success)
        info: dict[str, Any] = dict(self.metrics())
        return (
            {key: value.copy() for key, value in self._last_observation.items()},
            float(reward),
            terminated,
            truncated,
            info,
        )

    def _contact_details(self) -> tuple[int, float, dict[int, int]]:
        data = self.raw.sim.data
        model = self.raw.sim.model
        objects = set(self._object_geom_ids)
        counterparts = set(self._counterpart_geom_ids)
        counterpart_counts = {geom_id: 0 for geom_id in self._counterpart_geom_ids}
        count = 0
        maximum = 0.0
        force = np.zeros(6, dtype=np.float64)
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            first = int(contact.geom1)
            second = int(contact.geom2)
            if first in objects and second in counterparts:
                counterpart = second
            elif second in objects and first in counterparts:
                counterpart = first
            else:
                continue
            mujoco.mj_contactForce(model._model, data._data, contact_index, force)
            count += 1
            counterpart_counts[counterpart] += 1
            maximum = max(maximum, float(np.linalg.norm(force[:3])))
        return count, maximum, counterpart_counts

    def _contact_summary(self) -> tuple[int, float]:
        count, force, _ = self._contact_details()
        return count, force

    def contact_report(self) -> dict[str, object]:
        """Privileged smoke-only contact evidence; never part of reset/step observations."""
        count, force, counterpart_counts = self._contact_details()
        return {
            "relevant_contact_count": count,
            "maximum_contact_force": force,
            "object_geom_ids": list(self._object_geom_ids),
            "counterpart_geom_ids": list(self._counterpart_geom_ids),
            "counterpart_contact_counts": {
                name: counterpart_counts[geom_id]
                for name, geom_id in zip(self._counterpart_geom_names, self._counterpart_geom_ids)
            },
        }

    def prepare_contact_probe(self) -> np.ndarray:
        """Place the object for a short privileged contact-only simulator probe."""
        if self.task == "lift":
            pad_positions = np.asarray(
                [self.raw.sim.data.geom_xpos[index] for index in self._counterpart_geom_ids]
            )
            center = np.mean(pad_positions, axis=0)
            self.raw.sim.data.set_joint_qpos(
                self.raw.cube.joints[0], np.concatenate([center, [1.0, 0.0, 0.0, 0.0]])
            )
            self.raw.sim.data.set_joint_qvel(self.raw.cube.joints[0], np.zeros(6))
        mujoco.mj_forward(self.raw.sim.model._model, self.raw.sim.data._data)
        return np.array([0.0, 0.0, 0.0, 1.0 if self.task == "lift" else 0.0], dtype=np.float32)

    def metrics(self) -> dict[str, float | bool]:
        position = np.asarray(self.raw.sim.data.body_xpos[self._body_id], dtype=np.float64)
        target = np.asarray(self.raw.sim.data.site_xpos[self._target_site_id], dtype=np.float64)
        error = float(
            np.linalg.norm(position[:2] - target[:2])
            if self.task == "push"
            else np.linalg.norm(position - target)
        )
        _, force = self._contact_summary()
        metrics: dict[str, float | bool] = {
            "success": bool(self.raw._check_success()),
            "final_error": error,
            "completion_time": self.t * self.dt,
            "contact_force": force,
            "force_violation": force > self.force_limit,
            "controller_saturated": self._controller_saturated,
        }
        if self.task == "push":
            metrics["overshoot"] = float(max(0.0, position[0] - target[0]))
        else:
            metrics["drop"] = bool(position[2] < float(self.raw.table_offset[2]) - 0.02)
            metrics["slip"] = False
        return metrics

    def versions(self) -> dict[str, str]:
        return {
            "robosuite": metadata.version("robosuite"),
            "mujoco": metadata.version("mujoco"),
            "numpy": np.__version__,
        }

    def close(self) -> None:
        self.raw.close()
