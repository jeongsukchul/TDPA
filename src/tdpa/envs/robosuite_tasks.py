"""Fixed-topology robosuite tasks used by the no-training simulator gate.

This module is imported lazily by :mod:`tdpa.envs.make_env`, so the synthetic
backend does not require robosuite or MuJoCo.
"""

from __future__ import annotations

import numpy as np
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, new_site
from robosuite.utils.placement_samplers import UniformRandomSampler


class _TDPATabletop(Lift):
    tdpa_task = "tabletop"

    def __init__(
        self,
        *args: object,
        target_position: tuple[float, float, float],
        success_tolerance: float,
        object_half_size: tuple[float, float, float],
        **kwargs: object,
    ) -> None:
        self.tdpa_target_position = np.asarray(target_position, dtype=np.float64)
        self.tdpa_success_tolerance = float(success_tolerance)
        self.tdpa_object_half_size = np.asarray(object_half_size, dtype=np.float64)
        super().__init__(*args, **kwargs)

    def _load_model(self) -> None:
        # Bypass Lift._load_model so size, topology, and target are explicit.
        ManipulationEnv._load_model(self)
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        arena.worldbody.append(
            new_site(
                name="tdpa_target",
                pos=self.tdpa_target_position,
                size=(self.tdpa_success_tolerance,),
                rgba=(0.1, 0.8, 0.1, 0.28),
                type="sphere",
            )
        )

        material = CustomMaterial(
            texture="WoodRed",
            tex_name="tdpa_redwood",
            mat_name="tdpa_redwood_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )
        self.cube = BoxObject(
            name="tdpa_object",
            size=self.tdpa_object_half_size,
            rgba=(1, 0, 0, 1),
            material=material,
            duplicate_collision_geoms=False,
        )
        self.placement_initializer = UniformRandomSampler(
            name="TDPASampler",
            mujoco_objects=self.cube,
            x_range=(-0.01, 0.01),
            y_range=(-0.01, 0.01),
            rotation=0.0,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.0,
            rng=self.rng,
        )
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        self.tdpa_target_site_id = int(self.sim.model.site_name2id("tdpa_target"))
        if self.tdpa_target_site_id < 0:
            raise RuntimeError("Could not resolve tdpa_target site")

    def _current_target_position(self) -> np.ndarray:
        return np.asarray(self.sim.data.site_xpos[self.tdpa_target_site_id])


class TDPAPush(_TDPATabletop):
    tdpa_task = "push"

    def _check_success(self) -> bool:
        position = np.asarray(self.sim.data.body_xpos[self.cube_body_id])
        return bool(
            np.linalg.norm(position[:2] - self._current_target_position()[:2])
            <= self.tdpa_success_tolerance
        )


class TDPALiftTransport(_TDPATabletop):
    tdpa_task = "lift"

    def _check_success(self) -> bool:
        position = np.asarray(self.sim.data.body_xpos[self.cube_body_id])
        grasped = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)
        return bool(
            grasped
            and np.linalg.norm(position - self._current_target_position())
            <= self.tdpa_success_tolerance
        )
