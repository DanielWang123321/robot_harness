"""Persistent Unix Domain Socket session daemon and client for step-by-step Agent control."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

from .runtime import DEFAULT_RUNS_DIR, RobotHarnessRuntime

DEFAULT_SOCKET_PATH = DEFAULT_RUNS_DIR / "harness.sock"


class SessionDaemon:
    """Stateful local server keeping MuJoCo / xArm6 + 3D Viewer alive across CLI commands."""

    def __init__(
        self,
        runtime: RobotHarnessRuntime,
        socket_path: str | Path = DEFAULT_SOCKET_PATH,
    ) -> None:
        self.runtime = runtime
        self.socket_path = Path(socket_path)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._running = False

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a JSON request dict to RobotHarnessRuntime."""
        cmd = str(req.get("cmd", ""))
        args = req.get("args", {})
        if not isinstance(args, dict):
            return {"ok": False, "error": "'args' must be a JSON object"}

        try:
            if cmd == "ping" or cmd == "status":
                obs = self.runtime.observe(save_image=False)
                return {
                    "ok": True,
                    "status": "running",
                    "pid": os.getpid(),
                    "socket": str(self.socket_path),
                    **obs,
                }
            if cmd == "observe":
                return {"ok": True, **self.runtime.observe(**args)}
            if cmd == "xarm":
                method = str(args.get("method", ""))
                kwargs = args.get("kwargs", {})
                save_image = bool(args.get("save_image", False))
                camera = str(args.get("camera", "overview"))
                res = self.runtime.call_xarm(method, save_image=save_image, camera=camera, **kwargs)
                return {"ok": res.get("code", -1) == 0, **res}
            if cmd == "act":
                primitive = str(args.get("primitive", ""))
                params = args.get("params", {})
                save_image = bool(args.get("save_image", True))
                camera = str(args.get("camera", "overview"))
                res = self.runtime.act(primitive, save_image=save_image, camera=camera, **params)
                return {"ok": res.get("code", -1) == 0, **res}
            if cmd == "reset":
                res = self.runtime.reset_scene(**args)
                return {"ok": res.get("code", -1) == 0, **res}
            if cmd == "stop" or cmd == "shutdown":
                self._running = False
                saved_video = self.runtime.close()
                return {"ok": True, "status": "stopped", "saved_video": saved_video}
            return {"ok": False, "error": f"Unknown command: {cmd!r}"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def serve_forever(self) -> None:
        """Run the Unix domain socket event loop while refreshing the 3D viewer."""
        if self.socket_path.exists():
            self.socket_path.unlink()

        self._running = True
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as srv:
            srv.bind(str(self.socket_path))
            srv.listen(8)
            srv.settimeout(0.03)
            try:
                while self._running:
                    self.runtime.tick_viewer()
                    try:
                        conn, _ = srv.accept()
                    except TimeoutError:
                        continue
                    with conn:
                        raw = b""
                        while True:
                            chunk = conn.recv(65536)
                            if not chunk:
                                break
                            raw += chunk
                            if b"\n" in raw:
                                break
                        if not raw.strip():
                            continue
                        try:
                            req = json.loads(raw.decode("utf-8").splitlines()[0])
                            resp = self.handle_request(req)
                        except json.JSONDecodeError as exc:
                            resp = {"ok": False, "error": f"Invalid JSON: {exc}"}
                        conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            finally:
                self.runtime.close()
                if self.socket_path.exists():
                    self.socket_path.unlink()


class SessionClient:
    """Client for sending commands to a running SessionDaemon."""

    def __init__(self, socket_path: str | Path = DEFAULT_SOCKET_PATH, timeout_s: float = 30.0) -> None:
        self.socket_path = Path(socket_path)
        self.timeout_s = float(timeout_s)

    def is_running(self) -> bool:
        if not self.socket_path.exists():
            return False
        try:
            res = self.request("ping")
            return bool(res.get("ok", False))
        except OSError:
            return False

    def request(self, cmd: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.socket_path.exists():
            raise ConnectionError(f"No active session socket at {self.socket_path}")
        payload = json.dumps({"cmd": cmd, "args": args or {}}, ensure_ascii=False) + "\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout_s)
            sock.connect(str(self.socket_path))
            sock.sendall(payload.encode("utf-8"))
            buf = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break
        return json.loads(buf.decode("utf-8").splitlines()[0])
