"""Pick-Place motion primitive library built on standard XArmAPI methods."""

from __future__ import annotations

from dataclasses import dataclass

from core.backend import RobotBackend
from core.safety import XARM_OK
from core.types import ActionPrimitive, SceneState


@dataclass(frozen=True)
class PickPlaceHeightParams:
    """Task height and speed parameters in flange/TCP frames."""

    flange_grasp_z_m: float = 0.1871
    approach_m: float = 0.10
    lift_z_m: float = 0.30
    flange_to_tcp_z_m: float = -0.1735
    grasp_gap_m: float = 0.022
    cart_v_m_s: float = 0.15

    @property
    def grasp_z(self) -> float:
        return self.flange_grasp_z_m + self.flange_to_tcp_z_m

    @property
    def pre_grasp_z(self) -> float:
        return self.grasp_z + self.approach_m

    @property
    def lift_z(self) -> float:
        return self.lift_z_m + self.flange_to_tcp_z_m


def _require_xarm_ok(code: int, action: str) -> None:
    if code != XARM_OK:
        raise RuntimeError(f"XArmAPI {action} failed with code={code}")


class SkillLibrary:
    """Executes high-level ActionPrimitives using standard XArmAPI calls."""

    def __init__(
        self,
        backend: RobotBackend,
        params: PickPlaceHeightParams | None = None,
    ) -> None:
        self.backend = backend
        self.params = params or PickPlaceHeightParams()

    def execute_primitive(self, primitive: ActionPrimitive, scene: SceneState) -> None:
        name = primitive.name
        p = primitive.params

        if name == "move_home":
            _require_xarm_ok(self.backend.reset(wait=True), "reset")
        elif name == "open_gripper":
            _require_xarm_ok(
                self.backend.set_gripper_g2_position(pos=84, speed=150, force=50, wait=True),
                "set_gripper_g2_position(open)",
            )
        elif name == "close_gripper":
            gap_m = float(p.get("gap_m", self.params.grasp_gap_m))
            self.backend.close_gripper(target_gap_m=gap_m)
        elif name == "move_tcp":
            x_mm = float(p["x"]) * 1000.0
            y_mm = float(p["y"]) * 1000.0
            z_mm = float(p["z"]) * 1000.0
            spd_mm_s = self.params.cart_v_m_s * 1000.0
            _require_xarm_ok(
                self.backend.set_position(
                    x=x_mm,
                    y=y_mm,
                    z=z_mm,
                    roll=180.0,
                    pitch=0.0,
                    yaw=0.0,
                    speed=spd_mm_s,
                    is_radian=False,
                    wait=True,
                ),
                "set_position(move_tcp)",
            )
        elif name == "pick":
            obj_id = str(p["object_id"])
            self.pick(obj_id, scene)
        elif name == "place":
            target_id = str(p["target_id"])
            self.place(target_id, scene)
        else:
            raise ValueError(f"Unsupported action primitive: {name!r}")

    def pick(self, object_id: str, scene: SceneState) -> None:
        """Execute Pre-grasp -> Approach -> Grasp -> Lift waypoint sequence via XArmAPI."""
        obj = scene.get_object(object_id)
        ox_mm = obj.pose.position[0] * 1000.0
        oy_mm = obj.pose.position[1] * 1000.0
        h = self.params
        spd_mm_s = h.cart_v_m_s * 1000.0
        lift_z_mm = h.lift_z * 1000.0
        pre_z_mm = h.pre_grasp_z * 1000.0
        grasp_z_mm = h.grasp_z * 1000.0

        _require_xarm_ok(
            self.backend.set_gripper_g2_position(pos=84, speed=150, force=50, wait=True),
            "open_gripper",
        )
        _require_xarm_ok(
            self.backend.set_position(
                x=ox_mm, y=oy_mm, z=lift_z_mm, roll=180.0, pitch=0.0, yaw=0.0, speed=spd_mm_s, wait=True
            ),
            "pick_above",
        )
        _require_xarm_ok(
            self.backend.set_position(
                x=ox_mm, y=oy_mm, z=pre_z_mm, roll=180.0, pitch=0.0, yaw=0.0, speed=spd_mm_s, wait=True
            ),
            "pick_pre_grasp",
        )
        _require_xarm_ok(
            self.backend.set_position(
                x=ox_mm, y=oy_mm, z=grasp_z_mm, roll=180.0, pitch=0.0, yaw=0.0, speed=spd_mm_s * 0.75, wait=True
            ),
            "pick_grasp_descend",
        )
        self.backend.close_gripper(target_gap_m=h.grasp_gap_m, target_body="cube")
        _require_xarm_ok(
            self.backend.set_position(
                x=ox_mm, y=oy_mm, z=lift_z_mm, roll=180.0, pitch=0.0, yaw=0.0, speed=spd_mm_s, wait=True
            ),
            "pick_lift",
        )

    def place(self, target_id: str, scene: SceneState) -> None:
        """Execute Transport -> Descend -> Release -> Retract waypoint sequence via XArmAPI."""
        target = scene.get_object(target_id)
        tx_mm = target.pose.position[0] * 1000.0
        ty_mm = target.pose.position[1] * 1000.0
        h = self.params
        spd_mm_s = h.cart_v_m_s * 1000.0
        lift_z_mm = h.lift_z * 1000.0
        place_z_mm = h.grasp_z * 1000.0

        _require_xarm_ok(
            self.backend.set_position(
                x=tx_mm, y=ty_mm, z=lift_z_mm, roll=180.0, pitch=0.0, yaw=0.0, speed=spd_mm_s, wait=True
            ),
            "place_above",
        )
        _require_xarm_ok(
            self.backend.set_position(
                x=tx_mm, y=ty_mm, z=place_z_mm, roll=180.0, pitch=0.0, yaw=0.0, speed=spd_mm_s * 0.75, wait=True
            ),
            "place_descend",
        )
        _require_xarm_ok(
            self.backend.set_gripper_g2_position(pos=84, speed=150, force=50, wait=True),
            "place_release",
        )
        if hasattr(self.backend, "settle"):
            self.backend.settle(0.6)
        _require_xarm_ok(
            self.backend.set_position(
                x=tx_mm, y=ty_mm, z=lift_z_mm, roll=180.0, pitch=0.0, yaw=0.0, speed=spd_mm_s, wait=True
            ),
            "place_retract",
        )
