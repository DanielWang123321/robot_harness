"""MuJoCo interactive 3D viewer and offscreen video recorder."""

from __future__ import annotations

import importlib
import os
import queue
import sys
import threading
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def configure_glfw_backend() -> None:
    """Select the X11 GLFW backend on Linux desktops before launching MuJoCo viewer."""
    if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
        if not os.environ.get("PYGLFW_LIBRARY"):
            os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
        if os.environ.get("PYGLFW_LIBRARY_VARIANT", "").lower() == "x11" and "glfw" in sys.modules:
            import glfw
            import glfw.library

            if not glfw.platform_supported(glfw.PLATFORM_X11):
                importlib.reload(glfw.library)
                importlib.reload(glfw)
                viewer_mod = sys.modules.get("mujoco.viewer")
                if viewer_mod is not None and getattr(glfw, "_glfw", None) is not None:
                    viewer_mod._simulate.set_glfw_dlhandle(glfw._glfw._handle)


class PassiveViewer:
    """Interactive MuJoCo 3D viewer running on a dedicated render thread."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        configure_glfw_backend()
        import glfw
        import mujoco.viewer

        if (
            sys.platform.startswith("linux")
            and os.environ.get("PYGLFW_LIBRARY_VARIANT") == "x11"
            and glfw.platform_supported(glfw.PLATFORM_X11)
        ):
            glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)

        if not glfw.init():
            _, desc = glfw.get_error()
            detail = desc.decode(errors="replace") if desc else "unknown error"
            raise RuntimeError(f"GLFW initialization failed: {detail}. Ensure DISPLAY is set.")

        results: queue.Queue = queue.Queue()

        def _render() -> None:
            try:
                mujoco.viewer._launch_internal(
                    model,
                    data,
                    run_physics_thread=False,
                    handle_return=results,
                )
            except Exception as exc:  # noqa: BLE001
                results.put(exc)

        self._thread = threading.Thread(target=_render, name="mujoco-viewer", daemon=True)
        self._thread.start()
        result = results.get()
        if isinstance(result, Exception):
            self._thread.join()
            raise RuntimeError(f"MuJoCo viewer failed to start: {result}") from result  # noqa: TRY004
        self._handle = result

    def is_running(self) -> bool:
        return bool(self._handle.is_running())

    def sync(self) -> None:
        if self.is_running():
            self._handle.sync()

    def close(self) -> None:
        try:
            self._handle.close()
        finally:
            if self._thread.is_alive():
                self._thread.join(timeout=2.0)


class VideoRecorder:
    """Offscreen renderer that records simulation frames (front + eye_in_hand) to MP4."""

    def __init__(
        self,
        model: mujoco.MjModel,
        output_path: str | Path,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        camera: str = "front",
    ) -> None:
        self.model = model
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.camera = camera
        self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self._frames: list[np.ndarray] = []

    def capture(self, data: mujoco.MjData) -> None:
        self._renderer.update_scene(data, camera=self.camera)
        rgb = self._renderer.render().copy()
        self._frames.append(rgb)

    def save(self) -> Path:
        if self._frames:
            imageio.mimsave(str(self.output_path), self._frames, fps=self.fps)
        self._renderer.close()
        return self.output_path
