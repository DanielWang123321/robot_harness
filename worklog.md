# Worklog

## 2026-11-02

- **制定 v0.0.1 (Pick-Place) 开发-测试计划 (`plan.md`)**：
  - 明确硬件与架构方案：xArm 6 (`192.168.1.60`) + Gripper G2 + 奥比中光 Gemini 305（Eye-in-Hand 安装）。
  - 确定感知-规划-执行闭环路线与 M1~M5 开发里程碑、T1~T4 验收测试矩阵。

- **完成 M1~M4 开发、3D 可视化及 T1~T3 自动化测试验证**：
  - 完成自包含仿真资产 (`assets/`)、静态标定配置 (`configs/`)、核心安全与坐标变换 (`src/core/`)、仿真与实机双后端 (`src/execution/`)、真值与 Orbbec 视觉感知 (`src/perception/`) 及轨迹原语 (`src/planning/`)。
  - 新增 MuJoCo 实时 3D 交互窗口 (`--viewer`, `--hold`) 与 MP4 视频导出 (`--save-video`)。

- **升级架构为 `XArmAPI` 对齐的 Robot Control Harness（支持外部 Agent 直接闭环操控）**：
  - **剥离低频标定包并复用 `ufactory_mujoco` 核心模块**：
    - 移除冗余的 `src/calibration/` 目录，手眼标定 (`camera_calibration.yaml`) 与单机运动学标定 (`xarm6_kinematics_L56A01.yaml`) 统一作为静态只读配置在 `src/core/transforms.py` 中加载。
    - 移植 `apply_calibration_to_model`（单机运动学补偿）、`flange_m_quat_to_tcp_mm_rpy` / `tcp_mm_rpy_to_flange_m_quat`（6-DoF `tcp_offset` 变换）、`CollisionChecker`（基于膨胀模型的关节限位与自碰撞检测）以及 `get_gripper_contact_forces_n`（夹爪双指真实接触力 `N` 计算）。
  - **底层接口 100% 对齐 `xarm-python-sdk` (`XArmAPI`) 签名 (`src/core/backend.py`, `src/execution/`)**：
    - `MuJoCoBackend` 与 `XArmBackend` 统一实现 `get_position`、`set_position`（支持缺省参数保持当前值，如仅传 `z=100`）、`get_servo_angle`、`set_servo_angle`、`get_gripper_position`、`set_gripper_position`、`set_gripper_g2_position`、`set_tcp_offset`、`reset`、`get_state`、`get_err_warn_code`、`get_robot_sn` 等标准方法。
    - `SkillLibrary` 内部完全改为调用标准 `XArmAPI` 接口。
  - **新增外部 Agent 接入层 (`src/harness/`, `main.py`, `.agent/skills/robot-harness/SKILL.md`)**：
    - 实现 `RobotHarnessRuntime`：每次动作或观测均返回富反馈 JSON（含 TCP `mm/deg`、关节角、夹爪脉冲/毫米开口、左右指垫接触力 `N`、场景物体 3D 坐标及离屏渲染/相机拍摄的 PNG 快照路径）。
    - 实现 `SessionDaemon` 与 `SessionClient`：基于本地 Unix Domain Socket (`runs/harness.sock`) 保持 MuJoCo 3D Viewer 窗口、视频录制和机械臂状态在多次 CLI 调用之间常驻存活。
    - 实现 `MCPServer` (`src/harness/mcp_server.py`)：支持标准 JSON-RPC 2.0 over `stdio` 的 MCP 工具调用。
    - 扩展 `main.py` 子命令 (`session`、`observe`、`xarm`、`act`、`reset`、`mcp`) 并编写项目级 Skill。
  - **测试验证**：`ruff check` 零告警，`pytest -v tests` 共 17 项测试全部通过（含 `XArmAPI` 签名与部分坐标更新、越界返回错误码 `31`、常驻 `SessionDaemon` 跨调用状态持久化、MCP JSON-RPC 协议、20 组随机仿真 Pick-Place 100% 成功及真机/相机连通性测试）。

- **项目重命名为 `robot_harness` 并发布至 GitHub**：
  - 更新 `pyproject.toml` 项目名为 `robot_harness`，新增 `.gitignore` 与 `README.md`。
  - 初始化 Git 仓库并推送至公开仓库 `DanielWang123321/robot_harness`。
