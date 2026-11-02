"""Unified data structures for Perception-Planning-Execution loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Default top-down grasping quaternion (w, x, y, z): rotation of 180 deg around X/Y axis
GRIP_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)


@dataclass(frozen=True)
class Pose:
    """3D position (m) and orientation quaternion (w, x, y, z) in robot base frame."""

    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)

    @classmethod
    def from_xyz(
        cls,
        x: float,
        y: float,
        z: float,
        quat_wxyz: np.ndarray | tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0),
    ) -> Pose:
        q = np.asarray(quat_wxyz, dtype=float).reshape(4)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        return cls(
            position=(float(x), float(y), float(z)),
            orientation_wxyz=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        )

    @property
    def pos_np(self) -> np.ndarray:
        return np.array(self.position, dtype=float)

    @property
    def quat_np(self) -> np.ndarray:
        return np.array(self.orientation_wxyz, dtype=float)

    def with_z(self, new_z: float) -> Pose:
        return Pose(
            position=(self.position[0], self.position[1], float(new_z)),
            orientation_wxyz=self.orientation_wxyz,
        )


@dataclass(frozen=True)
class ObjectInfo:
    """Detected or ground-truth object in the workspace."""

    object_id: str
    category: str  # e.g., "graspable", "target_zone"
    pose: Pose
    size_xyz_m: tuple[float, float, float] = (0.03, 0.03, 0.03)
    color: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class RobotState:
    """Current robot telemetry in base frame."""

    joint_positions_rad: tuple[float, ...]
    tcp_pose: Pose
    flange_pose: Pose
    gripper_gap_m: float
    is_grasping: bool = False
    contact_forces_n: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class SceneState:
    """Structured snapshot produced by Perception and consumed by LLM Planner / External Agent."""

    robot: RobotState
    objects: dict[str, ObjectInfo] = field(default_factory=dict)
    timestamp: float = 0.0

    def get_object(self, object_id: str) -> ObjectInfo:
        if object_id not in self.objects:
            available = ", ".join(sorted(self.objects.keys())) or "none"
            raise KeyError(f"Object {object_id!r} not found in SceneState (available: {available})")
        return self.objects[object_id]

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "robot": {
                "tcp_xyz_m": [round(v, 4) for v in self.robot.tcp_pose.position],
                "tcp_xyz_mm": [round(v * 1000.0, 2) for v in self.robot.tcp_pose.position],
                "gripper_gap_m": round(self.robot.gripper_gap_m, 4),
                "gripper_gap_mm": round(self.robot.gripper_gap_m * 1000.0, 2),
                "is_grasping": self.robot.is_grasping,
                "contact_forces_n": [round(f, 3) for f in self.robot.contact_forces_n],
            },
            "objects": {
                obj_id: {
                    "category": obj.category,
                    "color": obj.color,
                    "xyz_m": [round(v, 4) for v in obj.pose.position],
                    "xyz_mm": [round(v * 1000.0, 2) for v in obj.pose.position],
                    "size_xyz_m": [round(v, 4) for v in obj.size_xyz_m],
                }
                for obj_id, obj in self.objects.items()
            },
        }


@dataclass(frozen=True)
class ActionPrimitive:
    """Standardized action primitive emitted by the Planner."""

    name: str  # "pick", "place", "move_home", "open_gripper", "close_gripper", "move_tcp"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkspaceLimits:
    """Cartesian and orientation safety bounds in base frame."""

    x_min_m: float = 0.15
    x_max_m: float = 0.65
    y_min_m: float = -0.45
    y_max_m: float = 0.45
    z_min_m: float = 0.008
    z_max_m: float = 0.55
    max_tilt_deg: float = 10.0
    max_cart_vel_m_s: float = 0.25


@dataclass
class ExecutionReport:
    """Summary of an executed task cycle."""

    instruction: str
    plan: list[ActionPrimitive]
    success: bool
    object_lifted: bool = False
    place_error_mm: float = float("inf")
    final_object_xyz: tuple[float, float, float] | None = None
    target_xyz: tuple[float, float, float] | None = None
    message: str = ""
