---
name: robot-harness
description: Control the UFACTORY xArm 6 + Gripper G2 in MuJoCo simulation or on real hardware (192.168.1.60) using XArmAPI-compatible CLI / MCP tools and closed-loop visual/physical feedback.
---

# Robot Control Harness Skill (`XArmAPI` Compatible)

Use this skill whenever the user asks you (the Agent) to inspect, plan, or control the xArm 6 + Gripper G2 robot in simulation or hardware.

## 1. Core Principles
- **Brain-in-the-Loop**: You (the upper-layer Agent) are the planner. Do not rely on one-shot blind scripts when step-by-step verification is appropriate.
- **`XArmAPI` Standard Conventions**:
  - Units: positions in **`mm`**, angles in **`deg`** (`is_radian=False` by default).
  - TCP Offset: initialized to `[0, 0, 173.5, 0, 0, 0]` (Gripper G2 fingertip center). `get_position` and `set_position` directly operate on the fingertip TCP.
  - Top-down orientation: `roll=180, pitch=0, yaw=0`.
  - Safety Guardrails: `WorkspaceValidator` ($Z \ge 8\text{ mm}$ table guard, tilt $\le 10^\circ$) and `CollisionChecker` automatically reject unsafe moves with non-zero error codes (`31` = out of range, `32` = collision, `21` = IK failed).

## 2. Standard Closed-Loop Workflow (CLI Session Mode)

### Step 1: Start a Persistent Session (keeps 3D Viewer & physics state alive)
Run as a background daemon task (`IsDaemon=True`, `WaitMsBeforeAsync=1000`):
```bash
conda run -n py314 python main.py session start --backend sim --viewer --save-video runs/session.mp4
```

### Step 2: Observe Scene & Visually Inspect
```bash
conda run -n py314 python main.py observe --camera overview
```
- Read the returned JSON (`robot_state.tcp_mm_deg`, `scene.red_cube.xyz_mm`, `scene.green_target.xyz_mm`, `snapshot_path`).
- Use `read_file` on `snapshot_path` (`runs/latest_view.png`) to visually inspect the scene.

### Step 3: Execute Actions (`XArmAPI` Atomic Calls or High-Level Skills)

**Option A — Direct `XArmAPI` Calls (`main.py xarm`)**:
```bash
# Move TCP above the cube (mm, deg)
conda run -n py314 python main.py xarm set_position --x 300 --y 0 --z 126.5 --roll 180 --pitch 0 --yaw 0 --speed 150

# Open Gripper G2 (84 mm gap)
conda run -n py314 python main.py xarm set_gripper_g2_position --pos 84

# Descend to grasp height (13.6 mm)
conda run -n py314 python main.py xarm set_position --z 13.6 --speed 100

# Close Gripper G2 (22 mm target gap) and check contact_forces_n in returned JSON
conda run -n py314 python main.py xarm set_gripper_g2_position --pos 22 --save-image

# Return to Home
conda run -n py314 python main.py xarm reset
```

**Option B — High-Level Verified Skills (`main.py act`)**:
```bash
conda run -n py314 python main.py act pick --object-id red_cube
conda run -n py314 python main.py act place --target-id green_target
conda run -n py314 python main.py act move_home
```

### Step 4: Stop Session & Save Video
```bash
conda run -n py314 python main.py session stop
```

