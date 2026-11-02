"""Orbbec Gemini 305 Eye-in-Hand RGB-D perception and 3D point localization."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from core.backend import RobotBackend
from core.transforms import (
    HandEyeCalibration,
    camera_point_to_base,
    load_hand_eye_calibration,
    pixel_depth_to_camera_point,
    pose_to_matrix,
)
from core.types import GRIP_DOWN_QUAT_WXYZ, ObjectInfo, Pose, SceneState

from .base import PerceptionProvider

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CALIB_PATH = PROJECT_ROOT / "configs" / "camera_calibration.yaml"
PY312_PYTHON = Path("/opt/conda_envs/py312/bin/python")


def detect_color_objects_rgbd(
    bgr: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    T_base_flange: np.ndarray,
    T_flange_cam: np.ndarray,
    *,
    min_area_px: int = 80,
) -> dict[str, ObjectInfo]:
    """Detect red/green objects in aligned BGR + depth_m (meters) and transform to base frame."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Red wraps around HSV hue 0/180
    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([165, 100, 70]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Green hue range
    mask_green = cv2.inRange(hsv, np.array([35, 80, 60]), np.array([85, 255, 255]))

    specs = [
        ("red_cube", "graspable", "red", mask_red, (0.03, 0.03, 0.03)),
        ("green_target", "target_zone", "green", mask_green, (0.04, 0.04, 0.03)),
    ]

    detected: dict[str, ObjectInfo] = {}
    for obj_id, category, color_name, mask, size_xyz in specs:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < min_area_px:
            continue

        M = cv2.moments(largest)
        if M["m00"] <= 1e-6:
            continue
        u = float(M["m10"] / M["m00"])
        v = float(M["m01"] / M["m00"])

        # Extract robust median depth in a small window around centroid (u, v)
        h, w = depth_m.shape[:2]
        ui, vi = round(u), round(v)
        u0, u1 = max(0, ui - 4), min(w, ui + 5)
        v0, v1 = max(0, vi - 4), min(h, vi + 5)
        patch = depth_m[v0:v1, u0:u1]
        valid = patch[(patch > 0.04) & (patch < 1.50) & np.isfinite(patch)]
        if valid.size == 0:
            continue
        d_m = float(np.median(valid))

        p_cam = pixel_depth_to_camera_point(u, v, d_m, K)
        p_base = camera_point_to_base(p_cam, T_base_flange, T_flange_cam)
        pose = Pose.from_xyz(float(p_base[0]), float(p_base[1]), float(p_base[2]), GRIP_DOWN_QUAT_WXYZ)
        detected[obj_id] = ObjectInfo(
            object_id=obj_id,
            category=category,
            pose=pose,
            size_xyz_m=size_xyz,
            color=color_name,
            confidence=min(1.0, area / 500.0),
        )

    return detected


class OrbbecPerception(PerceptionProvider):
    """Orbbec Gemini 305 RGB-D perception with Eye-in-Hand coordinate transformation."""

    def __init__(
        self,
        backend: RobotBackend,
        calibration_path: str | Path = DEFAULT_CALIB_PATH,
        *,
        serial: str = "",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.backend = backend
        self.calib: HandEyeCalibration = load_hand_eye_calibration(calibration_path)
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

    def capture_aligned_rgbd(self, timeout_ms: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        """Capture one aligned (bgr_uint8, depth_meters_float32) frame pair from Gemini 305."""
        try:
            import pyorbbecsdk as ob  # type: ignore[import-not-found]

            return self._capture_in_process(ob, timeout_ms)
        except ImportError:
            if not PY312_PYTHON.is_file():
                raise RuntimeError("pyorbbecsdk is not installed in current env and py312 bridge not found")
            return self._capture_via_py312_bridge(timeout_ms)

    def _capture_in_process(self, ob, timeout_ms: int) -> tuple[np.ndarray, np.ndarray]:
        ob.Context.set_logger_to_file(ob.OBLogLevel.NONE, "")
        ob.Context.set_logger_to_console(ob.OBLogLevel.ERROR)
        ctx = ob.Context()
        devs = ctx.query_devices()
        if devs.get_count() == 0:
            raise RuntimeError("No Orbbec Gemini 305 device connected")
        dev = devs.get_device_by_index(0)
        pipe = ob.Pipeline(dev)
        cfg = ob.Config()

        c_list = pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        c_prof = c_list.get_video_stream_profile(self.width, self.height, ob.OBFormat.BGR, self.fps)
        cfg.enable_stream(c_prof)

        d_list = pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        d_prof = d_list.get_video_stream_profile(self.width, self.height, ob.OBFormat.Y16, self.fps)
        cfg.enable_stream(d_prof)
        cfg.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        pipe.enable_frame_sync()
        align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)

        pipe.start(cfg)
        try:
            for _ in range(10):
                frames = pipe.wait_for_frames(timeout_ms)
                if frames is None:
                    continue
                aligned = align.process(frames)
                if aligned is None:
                    continue
                fset = aligned.as_frame_set()
                c_frame = fset.get_color_frame() or frames.get_color_frame()
                d_frame = fset.get_depth_frame()
                if c_frame is None or d_frame is None:
                    continue
                bgr = np.frombuffer(c_frame.get_data(), dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
                scale_mm = float(d_frame.get_depth_scale())
                raw_u16 = np.frombuffer(d_frame.get_data(), dtype=np.uint16).reshape((self.height, self.width))
                depth_m = raw_u16.astype(np.float32) * (scale_mm / 1000.0)
                return bgr, depth_m
            raise RuntimeError("Timed out waiting for aligned RGB-D frame from Gemini 305")
        finally:
            pipe.stop()

    def _capture_via_py312_bridge(self, timeout_ms: int) -> tuple[np.ndarray, np.ndarray]:
        with tempfile.TemporaryDirectory(prefix="orbbec_cap_") as tmpdir:
            out_npz = Path(tmpdir) / "frame.npz"
            script = f"""
import numpy as np
import pyorbbecsdk as ob

ob.Context.set_logger_to_file(ob.OBLogLevel.NONE, "")
ob.Context.set_logger_to_console(ob.OBLogLevel.ERROR)
ctx = ob.Context()
devs = ctx.query_devices()
if devs.get_count() == 0:
    raise RuntimeError("No Orbbec device connected")
dev = devs.get_device_by_index(0)
pipe = ob.Pipeline(dev)
cfg = ob.Config()
c_list = pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
c_prof = c_list.get_video_stream_profile({self.width}, {self.height}, ob.OBFormat.BGR, {self.fps})
cfg.enable_stream(c_prof)
d_list = pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
d_prof = d_list.get_video_stream_profile({self.width}, {self.height}, ob.OBFormat.Y16, {self.fps})
cfg.enable_stream(d_prof)
cfg.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
pipe.enable_frame_sync()
align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
pipe.start(cfg)
try:
    for _ in range(12):
        frames = pipe.wait_for_frames({int(timeout_ms)})
        if frames is None:
            continue
        aligned = align.process(frames)
        if aligned is None:
            continue
        fset = aligned.as_frame_set()
        cf = fset.get_color_frame() or frames.get_color_frame()
        df = fset.get_depth_frame()
        if cf is None or df is None:
            continue
        bgr = np.frombuffer(cf.get_data(), dtype=np.uint8).reshape(({self.height}, {self.width}, 3)).copy()
        scale_mm = float(df.get_depth_scale())
        raw_u16 = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(({self.height}, {self.width}))
        depth_m = raw_u16.astype(np.float32) * (scale_mm / 1000.0)
        np.savez({str(out_npz)!r}, bgr=bgr, depth_m=depth_m, sn=dev.get_device_info().get_serial_number())
        break
    else:
        raise RuntimeError("Timed out waiting for Orbbec frame")
finally:
    pipe.stop()
"""
            proc = subprocess.run(
                [str(PY312_PYTHON), "-c", script],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
            if proc.returncode != 0 or not out_npz.is_file():
                raise RuntimeError(f"Orbbec capture bridge failed: {proc.stderr.strip()}")
            data = np.load(out_npz)
            return data["bgr"].copy(), data["depth_m"].copy()

    def observe(self) -> SceneState:
        robot_state = self.backend.get_robot_state()
        bgr, depth_m = self.capture_aligned_rgbd()
        T_base_flange = pose_to_matrix(robot_state.flange_pose)
        objects = detect_color_objects_rgbd(
            bgr,
            depth_m,
            self.calib.K,
            T_base_flange,
            self.calib.T_flange_cam,
        )
        return SceneState(
            robot=robot_state,
            objects=objects,
            timestamp=time.time(),
        )
