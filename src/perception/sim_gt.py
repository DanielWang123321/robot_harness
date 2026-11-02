"""MuJoCo simulation ground-truth perception provider."""

from __future__ import annotations

import time

import mujoco

from core.types import GRIP_DOWN_QUAT_WXYZ, ObjectInfo, Pose, SceneState
from execution.mujoco_backend import MuJoCoBackend

from .base import PerceptionProvider


class SimGTPerception(PerceptionProvider):
    """Extracts ground-truth object poses from MuJoCo mjData into SceneState."""

    def __init__(self, backend: MuJoCoBackend) -> None:
        self.backend = backend

    def observe(self) -> SceneState:
        model = self.backend.model
        data = self.backend.data
        mujoco.mj_forward(model, data)

        objects: dict[str, ObjectInfo] = {}

        cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if cube_id >= 0:
            pos = data.body(cube_id).xpos.copy()
            cube_pose = Pose.from_xyz(float(pos[0]), float(pos[1]), float(pos[2]), GRIP_DOWN_QUAT_WXYZ)
            objects["red_cube"] = ObjectInfo(
                object_id="red_cube",
                category="graspable",
                pose=cube_pose,
                size_xyz_m=(0.03, 0.03, 0.03),
                color="red",
                confidence=1.0,
            )

        target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_marker")
        if target_geom_id >= 0:
            t_pos = data.geom_xpos[target_geom_id].copy()
            target_pose = Pose.from_xyz(float(t_pos[0]), float(t_pos[1]), float(t_pos[2]), GRIP_DOWN_QUAT_WXYZ)
            objects["green_target"] = ObjectInfo(
                object_id="green_target",
                category="target_zone",
                pose=target_pose,
                size_xyz_m=(0.04, 0.04, 0.03),
                color="green",
                confidence=1.0,
            )

        return SceneState(
            robot=self.backend.get_robot_state(),
            objects=objects,
            timestamp=time.time(),
        )
