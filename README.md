# robot_harness

面向外部 Agent（Antigravity / Claude Code / Cursor 等）的 **UFACTORY xArm 6 + Gripper G2 + Orbbec Gemini 305** 机器人闭环控制宿主（Robot Control Harness），同时支持 **MuJoCo 高保真物理仿真 (`sim`)** 与 **真机控制 (`real`)**。

## 核心特性

1. **100% 对齐 `xarm-python-sdk` (`XArmAPI`) 签名**：
   - 仿真后端（`MuJoCoBackend`）与真机后端（`XArmBackend`）共享完全一致的 `XArmAPI` 核心方法签名（`get_position`、`set_position`、`get_servo_angle`、`set_servo_angle`、`get_gripper_position`、`set_gripper_position`、`set_gripper_g2_position`、`set_tcp_offset`、`reset` 等）与返回码约定（`0` 为成功，`31` 为越界拦截，`32` 为碰撞拦截）。
   - `set_position` 原生支持缺省参数保持当前值（如仅传 `z=100` 实现垂直升降）。
2. **外部 Agent 优先的多轮闭环操控 (`Observe -> Act -> Feedback`)**：
   - **常驻会话守护进程 (`SessionDaemon`)**：通过本地 Unix Domain Socket (`runs/harness.sock`) 保持 MuJoCo 3D 交互窗口、视频录制和机械臂状态在多次 CLI 调用之间存活。
   - **视觉 + 物理双模态反馈**：每次观测或动作均返回 TCP 位姿 (`mm`/`deg`)、关节角、夹爪开合度、**双指真实接触力 (`contact_forces_n`)**、场景物体 3D 坐标以及相机/仿真快照 PNG (`runs/latest_view.png`)。
   - **MCP Server (`stdio`)**：原生支持 JSON-RPC 2.0 MCP 工具调用（`xarm_observe`、`xarm_set_position`、`xarm_set_gripper_g2_position`、`xarm_pick`、`xarm_place` 等）。
3. **内置多层安全护栏**：
   - 笛卡尔工作空间硬限位与桌面防撞保护（$Z \ge 8\text{ mm}$）。
   - 基于膨胀胶囊模型的关节限位与自碰撞预检（`CollisionChecker`）。
   - 真机默认 `read_only=True` 门锁，需显式传入 `--confirm-real` 方可解锁运动指令。

---

## 快速开始

### 1. 环境准备

```bash
conda activate py314
```

### 2. Agent 多轮交互式控制（推荐）

```bash
# 1) 启动后台常驻会话（自动弹出 MuJoCo 3D 实时窗口）
python main.py session start --backend sim --viewer

# 2) 观测当前状态与场景（生成 runs/latest_view.png 快照）
python main.py observe

# 3) 调用标准 XArmAPI 原子指令（支持部分坐标更新）
python main.py xarm set_gripper_g2_position --pos 84
python main.py xarm set_position --x 300 --y 0 --z 120 --roll 180 --pitch 0 --yaw 0
python main.py xarm set_position --z 20

# 4) 或直接调用高层复合技能
python main.py act pick --object cube --grasp-z-mm 13.6
python main.py act place --x-mm 300 --y-mm 300 --z-mm 15

# 5) 结束常驻会话
python main.py session stop
```

### 3. 单次指令闭环执行

```bash
python main.py --backend sim --viewer --instruction "Pick up the red cube and place it on the green target."
```

### 4. 启动 MCP Server (`stdio`)

```bash
python main.py mcp --backend sim
```

---

## 测试验证

```bash
# 代码规范检查
conda run -n py314 ruff check src tests main.py

# 运行全套单元、仿真与硬件连通性测试
conda run -n py314 pytest -v tests
```
