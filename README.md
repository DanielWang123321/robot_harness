# robot_harness

A **Robot Control Harness** for external AI agents (Antigravity / Claude / Cursor, etc.) targeting **UFACTORY xArm 6 + Gripper G2 + Orbbec Gemini 305**, supporting both **MuJoCo high-fidelity physics simulation (`sim`)** and **real hardware control (`real`)**.

## Key Features

1. **100% API-compatible with `xarm-python-sdk` (`XArmAPI`)**:
   - Simulation backend (`MuJoCoBackend`) and real hardware backend (`XArmBackend`) share identical `XArmAPI` method signatures (`get_position`, `set_position`, `get_servo_angle`, `set_servo_angle`, `get_gripper_position`, `set_gripper_position`, `set_gripper_g2_position`, `set_tcp_offset`, `reset`, etc.) and return code conventions (`0` for success, `31` for workspace violation, `32` for collision).
   - `set_position` natively supports partial updates with default parameters (e.g., passing only `z=100` for vertical movement while maintaining other axes).
2. **Multi-turn closed-loop control optimized for external AI agents (`Observe -> Act -> Feedback`)**:
   - **Persistent session daemon (`SessionDaemon`)**: Maintains MuJoCo 3D interactive window, video recording, and robot state across multiple CLI invocations via local Unix Domain Socket (`runs/harness.sock`).
   - **Vision + physics dual-modal feedback**: Every observation or action returns TCP pose (`mm`/`deg`), joint angles, gripper aperture, **bilateral fingertip contact forces (`contact_forces_n`)**, scene object 3D coordinates, and camera/simulation snapshot PNG (`runs/latest_view.png`).
   - **MCP Server (`stdio`)**: Native support for JSON-RPC 2.0 MCP tool calls (`xarm_observe`, `xarm_set_position`, `xarm_set_gripper_g2_position`, `xarm_pick`, `xarm_place`, etc.).
3. **Built-in multi-layer safety guardrails**:
   - Cartesian workspace hard limits and table collision protection ($Z \ge 8\text{ mm}$).
   - Joint limits and self-collision pre-checks based on inflated capsule models (`CollisionChecker`).
   - Real hardware defaults to `read_only=True` safety lock; motion commands require explicit `--confirm-real` flag.

---

## Quick Start

### 1. Environment Setup

```bash
conda activate py314
```

### 2. Multi-turn Interactive Control via Agent (Recommended)

```bash
# 1) Start persistent background session (launches MuJoCo 3D real-time window)
python main.py session start --backend sim --viewer

# 2) Observe current state and scene (generates runs/latest_view.png snapshot)
python main.py observe

# 3) Call standard XArmAPI atomic commands (supports partial coordinate updates)
python main.py xarm set_gripper_g2_position --pos 84
python main.py xarm set_position --x 300 --y 0 --z 120 --roll 180 --pitch 0 --yaw 0
python main.py xarm set_position --z 20

# 4) Or directly invoke high-level composite skills
python main.py act pick --object cube --grasp-z-mm 13.6
python main.py act place --x-mm 300 --y-mm 300 --z-mm 15

# 5) Stop persistent session
python main.py session stop
```

### 3. Single-Shot Instruction Execution

```bash
python main.py --backend sim --viewer --instruction "Pick up the red cube and place it on the green target."
```

### 4. Start MCP Server (`stdio`)

```bash
python main.py mcp --backend sim
```

### 5. Render Demo Video

```bash
# 1920x1080 / 30fps / H.264, main view + eye-in-hand picture-in-picture
conda run -n py314 python scripts/render_demo_video.py --duration 10 --output runs/robot_harness_demo.mp4
```

---

## Testing & Validation

```bash
# Code style check
conda run -n py314 ruff check src tests main.py

# Run full test suite (unit, simulation, and hardware connectivity tests)
conda run -n py314 pytest -v tests
```

