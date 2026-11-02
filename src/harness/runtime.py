"""Stateful RobotHarnessRuntime exposing XArmAPI and high-level skills with rich feedback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from core.backend import RobotBackend
from core.safety import XARM_ERR_INVALID_PARAM, XARM_OK
from core.types import ActionPrimitive
from execution.mujoco_backend import MuJoCoBackend
from perception.base import PerceptionProvider
from perception.orbbec_gemini import OrbbecPerception
from perception.sim_gt import SimGTPerception
from planning.primitives import PickPlaceHeightParams, SkillLibrary

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"

ALLOWED_XARM_METHODS = frozenset(
    {
        "motion_enable",
        "set_mode",
        "set_state",
        "clean_error",
        "clean_warn",
        "emergency_stop",
        "get_state",
        "get_err_warn_code",
        "get_robot_sn",
        "set_tcp_offset",
        "get_position",
        "get_servo_angle",
        "set_position",
        "set_servo_angle",
        "reset",
        "set_gripper_enable",
        "set_gripper_mode",
        "set_gripper_speed",
        "get_gripper_position",
        "set_gripper_position",
        "set_gripper_g2_position",
    }
)


class RobotHarnessRuntime:
    """Unified stateful runtime for external Agent control via XArmAPI and primitives."""

    def __init__(
        self,
        backend: RobotBackend,
        perception: PerceptionProvider | None = None,
        height_params: PickPlaceHeightParams | None = None,
        *,
        runs_dir: str | Path = DEFAULT_RUNS_DIR,
    ) -> None:
        self.backend = backend
        if perception is not None:
            self.perception = perception
        elif isinstance(backend, MuJoCoBackend):
            self.perception = SimGTPerception(backend)
        else:
            self.perception = OrbbecPerception(backend)

        if height_params is None and isinstance(backend, MuJoCoBackend):
            offset_z = backend.measure_flange_to_tcp_offset_z(ref_xy=(0.30, 0.0))
            height_params = PickPlaceHeightParams(flange_to_tcp_z_m=offset_z)
        self.skills = SkillLibrary(backend, height_params)
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def tick_viewer(self) -> None:
        """Keep interactive 3D viewer responsive when idle."""
        if isinstance(self.backend, MuJoCoBackend):
            self.backend.sync_viewer()

    def close(self) -> str | None:
        """Close backend and return saved video path if any."""
        close_fn = getattr(self.backend, "close", None)
        if callable(close_fn):
            res = close_fn()
            if isinstance(res, Path):
                return str(res)
        return None

    def _capture_snapshot(self, camera: str = "overview", filename: str = "latest_view.png") -> str | None:
        out_path = self.runs_dir / filename
        if isinstance(self.backend, MuJoCoBackend):
            saved = self.backend.render_snapshot(out_path, camera=camera)
            return str(saved)
        if isinstance(self.perception, OrbbecPerception):
            bgr, _ = self.perception.capture_aligned_rgbd()
            cv2.imwrite(str(out_path), bgr)
            return str(out_path)
        return None

    def _build_telemetry(
        self,
        *,
        save_image: bool = False,
        camera: str = "overview",
        image_name: str = "latest_view.png",
    ) -> dict[str, Any]:
        scene = self.perception.observe()
        _, pos_mm_deg = self.backend.get_position(is_radian=False)
        _, servo_deg = self.backend.get_servo_angle(is_radian=False)
        _, gripper_counts = self.backend.get_gripper_position()
        _, state_code = self.backend.get_state()
        _, err_warn = self.backend.get_err_warn_code()

        snapshot_path = (
            self._capture_snapshot(camera=camera, filename=image_name) if save_image else None
        )

        return {
            "robot_state": {
                "state": state_code,
                "error_code": err_warn[0],
                "warn_code": err_warn[1],
                "tcp_mm_deg": [round(float(v), 3) for v in pos_mm_deg],
                "servo_deg": [round(float(v), 3) for v in servo_deg] if isinstance(servo_deg, list) else servo_deg,
                "gripper_pos_counts": round(float(gripper_counts), 2),
                "gripper_gap_mm": round(float(scene.robot.gripper_gap_m * 1000.0), 2),
                "is_grasping": bool(scene.robot.is_grasping),
                "contact_forces_n": [round(float(f), 3) for f in scene.robot.contact_forces_n],
            },
            "scene": scene.to_prompt_dict()["objects"],
            "snapshot_path": snapshot_path,
        }

    def observe(
        self,
        *,
        save_image: bool = True,
        camera: str = "overview",
        image_name: str = "latest_view.png",
    ) -> dict[str, Any]:
        """Observe current robot and scene state, optionally saving an RGB snapshot."""
        telemetry = self._build_telemetry(
            save_image=save_image,
            camera=camera,
            image_name=image_name,
        )
        return {
            "code": XARM_OK,
            "action": "observe",
            **telemetry,
        }

    def call_xarm(
        self,
        method: str,
        *,
        save_image: bool = False,
        camera: str = "overview",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke a standard XArmAPI method on the underlying backend and return closed-loop state."""
        if method not in ALLOWED_XARM_METHODS:
            return {
                "code": XARM_ERR_INVALID_PARAM,
                "error": f"Unsupported XArmAPI method {method!r}. Allowed: {sorted(ALLOWED_XARM_METHODS)}",
            }

        fn = getattr(self.backend, method)
        raw_ret = fn(**kwargs)
        if isinstance(raw_ret, tuple) and len(raw_ret) == 2:
            code, data = int(raw_ret[0]), raw_ret[1]
        elif raw_ret is None:
            code, data = XARM_OK, None
        else:
            code, data = int(raw_ret), None

        telemetry = self._build_telemetry(
            save_image=save_image,
            camera=camera,
            image_name=f"after_{method}.png",
        )
        return {
            "code": code,
            "method": method,
            "data": data,
            **telemetry,
        }

    def act(
        self,
        primitive_name: str,
        *,
        save_image: bool = True,
        camera: str = "overview",
        **params: Any,
    ) -> dict[str, Any]:
        """Execute a high-level ActionPrimitive (pick, place, move_home, etc.) and verify result."""
        scene = self.perception.observe()
        prim = ActionPrimitive(name=primitive_name, params=dict(params))
        try:
            self.skills.execute_primitive(prim, scene)
            code = XARM_OK
            error_msg = None
        except Exception as exc:  # noqa: BLE001
            code = -1
            error_msg = str(exc)

        telemetry = self._build_telemetry(
            save_image=save_image,
            camera=camera,
            image_name=f"after_{primitive_name}.png",
        )
        result: dict[str, Any] = {
            "code": code,
            "action": primitive_name,
            "params": params,
            **telemetry,
        }
        if error_msg is not None:
            result["error"] = error_msg
        return result

    def reset_scene(
        self,
        *,
        cube_x: float = 0.30,
        cube_y: float = 0.00,
        target_x: float = 0.30,
        target_y: float = 0.30,
        save_image: bool = True,
    ) -> dict[str, Any]:
        """Reset simulation scene props & robot arm, or move real arm home."""
        if isinstance(self.backend, MuJoCoBackend):
            self.backend.reset_scene()
            self.backend.set_cube_xy(float(cube_x), float(cube_y))
            self.backend.set_target_xy(float(target_x), float(target_y))
            code = XARM_OK
        else:
            code = self.backend.reset(wait=True)

        telemetry = self._build_telemetry(
            save_image=save_image,
            camera="overview",
            image_name="after_reset.png",
        )
        return {
            "code": code,
            "action": "reset_scene",
            **telemetry,
        }

