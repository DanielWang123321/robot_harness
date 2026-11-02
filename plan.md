# 开发-测试计划 (robot_harness: `XArmAPI` 对齐 + 外部 Agent 闭环操控)

## 1. 系统定位与架构原则

本项目定位为**面向外部 Agent（Antigravity / Claude Code / Cursor 等）的机器人控制宿主（`robot_harness`）**，遵循以下三大设计原则：
1. **底层接口 100% 对齐 `xarm-python-sdk` (`XArmAPI`) 签名**：仿真端（[`MuJoCoBackend`](src/execution/mujoco_backend.py)）与真机端（[`XArmBackend`](src/execution/xarm_backend.py)）共享完全一致的 `XArmAPI` 核心方法签名（`get_position`、`set_position`、`get_servo_angle`、`set_servo_angle`、`get_gripper_position`、`set_gripper_position`、`set_gripper_g2_position`、`set_tcp_offset`、`reset` 等），遵循 `(code, data)` / `code` 返回规范与 `mm` / `deg`（可选 `is_radian=True`）参数约定。
2. **消费静态标定配置，剥离低频/单次标定与辨识任务**：
   - 手眼标定（[`configs/camera_calibration.yaml`](configs/camera_calibration.yaml)）与单机运动学标定（[`configs/xarm6_kinematics_L56A01.yaml`](configs/xarm6_kinematics_L56A01.yaml)）作为静态只读配置加载，不在运行时保留标定采集、动力学辨识或离线固件取证代码。
3. **外部 Agent 优先的状态化闭环操控**：
   - 通过本地常驻会话服务（[`SessionDaemon`](src/harness/session.py)）、CLI 子命令（[`main.py`](main.py)）与 MCP Server（[`mcp_server.py`](src/harness/mcp_server.py)），支持外部大模型分步执行 **`Observe`（含视觉快照 PNG） $\rightarrow$ `XArmAPI / Skill Act` $\rightarrow$ `Physical Feedback`（含双指接触力 `N`）** 的多轮闭环操作。

---

## 2. 目录结构

```text
robot_harness/
├── .agent/skills/robot-harness/SKILL.md  # 外部 Agent 操控机械臂的标准工作流 Skill
├── assets/models/xarm6_with_gripper/     # 自包含 xArm 6 + Gripper G2 + Eye-in-Hand 仿真模型
├── configs/
│   ├── default.yaml                      # 工作空间限位、任务参数与硬件默认配置
│   ├── camera_calibration.yaml           # 静态 Eye-in-Hand 手眼标定与相机内参
│   └── xarm6_kinematics_L56A01.yaml      # 静态单机运动学标定 (SN: XI130505L56A01)
├── src/
│   ├── core/                             # 类型定义、SE(3)/TCP 变换、WorkspaceValidator 与 CollisionChecker
│   ├── execution/                        # XArmAPI 对齐双后端 (MuJoCoBackend / XArmBackend)、IK、3D Viewer
│   ├── perception/                       # SimGTPerception (仿真真值) 与 OrbbecPerception (Gemini 305)
│   ├── planning/                         # 基于 XArmAPI 的 SkillLibrary 与 LLMPlanner
│   ├── harness/                          # RobotHarnessRuntime、常驻 SessionDaemon/Client、MCP Server
│   └── orchestrator.py                   # 单次指令闭环编排器
├── main.py                               # 统一 CLI 入口 (session / observe / xarm / act / reset / mcp)
└── tests/
    ├── unit/                             # 坐标变换、安全护栏、XArmAPI 签名、SessionDaemon 与 MCP 测试
    ├── sim/                              # 20 组随机位置 Pick-Place 仿真稳定性测试
    └── hardware/                         # xArm 6 (192.168.1.60) 与 Gemini 305 硬件连通与安全门锁测试
```

---

## 3. 暴露给外部 Agent 的接口规范

### 3.1 `XArmAPI` 对齐控制接口 (`main.py xarm <method>` / MCP)
- **系统与状态**：`motion_enable`、`set_mode`、`set_state`、`clean_error`、`clean_warn`、`emergency_stop`、`get_state`、`get_err_warn_code`、`get_robot_sn`。
- **位姿与关节查询**：`get_position(is_radian=False)`、`get_servo_angle(servo_id=None, is_radian=False)`。
- **笛卡尔与关节运动（内置安全护栏）**：
  - `set_tcp_offset(offset, is_radian=False)`（默认 `[0, 0, 173.5, 0, 0, 0]`）。
  - `set_position(x=None, y=None, z=None, roll=None, pitch=None, yaw=None, speed=None, relative=False, is_radian=False, wait=True)`（支持缺省参数保持当前值；越界或 $Z < 8\text{ mm}$ 拦截并返回 `code=31`）。
  - `set_servo_angle(servo_id=None, angle=None, speed=None, relative=False, is_radian=False, wait=True)`（内置 [`CollisionChecker`](src/core/safety.py) 关节限位与自碰撞拦截）。
  - `reset(speed=None, is_radian=False, wait=True)`。
- **Gripper G2 夹爪控制**：`set_gripper_enable`、`set_gripper_mode`、`set_gripper_speed`、`get_gripper_position`、`set_gripper_position(pos, wait=True)`（`0~840` 脉冲）、`set_gripper_g2_position(pos, speed=100, force=50, wait=True)`（`0~84 mm` 物理开口）。

### 3.2 Harness 感知与高层技能接口
- **`observe` (`xarm_observe`)**：返回 `robot_state`（含 `tcp_mm_deg`、`servo_deg`、`gripper_gap_mm`、`is_grasping`、左右指垫接触力 `contact_forces_n`）、`scene` 物体 3D 坐标（`mm` 与 `m`）及 RGB 快照路径 `snapshot_path`。
- **`act` (`xarm_pick` / `xarm_place` / `move_home`)**：基于 `XArmAPI` 封装的验证级复合动作技能。
- **`session` (`start` / `status` / `stop`)**：管理常驻进程，保持 MuJoCo 3D Viewer 窗口和硬件连接跨调用存活。

---

## 4. 测试与验收矩阵

| 测试阶段 | 测试文件 | 验证内容 | 验收标准 |
| :--- | :--- | :--- | :--- |
| **T1: 单元与 Harness 测试** | `tests/unit/test_transforms_and_safety.py`<br>`tests/unit/test_llm_planner.py`<br>`tests/unit/test_xarm_harness.py` | SE(3)/TCP 转换精度、工作空间与碰撞拦截、`XArmAPI` 签名与缺省参数更新、`SessionDaemon` 跨调用状态持久化、MCP JSON-RPC 协议 | 误差 $< 10^{-6}$；100% 拦截越界指令并返回标准错误码 `31`；常驻会话与 MCP 协议全通过 |
| **T2: 仿真闭环测试** | `tests/sim/test_pick_place_sim.py` | 20 组随机物块/目标位置 Pick-Place 闭环、G2 四连杆映射与双指接触力检测 | 20/20 成功率 $\ge 95\%$，放置误差 $< 50\text{ mm}$ |
| **T3: 硬件连通与门锁测试** | `tests/hardware/test_hardware_connectivity.py` | xArm 6 (`192.168.1.60`) 只读遥测延迟、`read_only=True` 门锁、Gemini 305 对齐 RGB-D 采集 | 遥测延迟 $< 20\text{ ms}$；只读模式 100% 阻断运动指令；RGB-D 帧正常 |
| **T4: 实机闭环测试** | 实机环境 | 释放物理急停后，先以悬空安全偏置验证，再执行低速 (`<= 100 mm/s`) 实机抓取放置 | 连续抓取稳固无碰撞 |
