# Robot Simulation Framework — 實作計畫

> 入口文件：`README.md`（英文）；開發新 cell：`docs/new_cell.md`；名詞與 D/M 編號：`docs/decisions.md`。

狀態：**v0.2，已納入使用者 2026-11-12 的決定**（§9）
日期：2026-11-12

---

## 0. 目標與輸入/輸出

```
3D 檔 (URDF / mesh / USD)          ┐
layout (場景配置 YAML)              ├─► 編譯 ─► 自動點位 ─► 路徑生成 ─► 模擬驗證 ─► 程式/軌跡輸出
workflow (流程 YAML)                ┘          (CPU)        (CPU)                  (實機格式由使用者設計，暫緩)
                                                         ▲
                                     real-time 規劃 (感測修正 / 重規劃 / MPC) ─┘
```

- **路徑規劃兩個目標**
  1. 離線：自動點位（teach points）＋路徑生成，輸出已在模擬中驗證的點位與軌跡。
     **執行環境無 GPU**；生成不必 real-time，新的靜態障礙出現時可隨時自動重新生成。
  2. 線上：real-time 路徑規劃（定位修正、移動物進入工作區、重規劃 / MPC）。
     **執行環境有 GPU（cuRobo）**；**延後實作**，但介面現在就預留。
- **定位**（AprilGrid）目的是補償架點與實機的誤差，兩種模式都要：
  (a) 執行前量測 → 修正點位並重新生成；(b) 執行中量測 → 線上局部修正。
- **虛擬環境可替換**：CPU 環境以 MuJoCo（物理）＋ RViz / web viewer（視覺化）為主，保留 Isaac Sim。
- **後續**：agent 協助架點。
- **第一個測試目標**：原專案 TM5S tray cell（37 slots, strict preset）。

---

## 1. 從原專案（AI-Robotics-V2）學到並保留的東西

| 原專案 | 新框架中的位置 | 備註 |
|---|---|---|
| `cell_core`（assets/config/frames/geometry/sequence/scheduler/scene） | `cells/tm_tray/` plugin ＋ 通用 `rsf.model` | 通用部分抽成框架；TM tray 專屬（rack 產生器、plate-insert 技能）成為 plugin |
| `env_adapter` → Isaac 產生器參數 | `rsf.compile.usd` | 改成「Cell model → 各後端」的編譯器，一個來源 |
| `protocol.py`（ROS2 JSON 軌跡 seam, D9/D30） | `rsf.drivers.ros2_json` | 保留為一種 transport；D30 的 transfer/judge 準則作為驗收參考 |
| `check_cell.py`（M3 mesh 碰撞, margin ladder, 分組） | `rsf.collision` + `rsf.planning.validate` | 改用 cuRobo dev/main 原生 mesh check（§6.1） |
| `paths.py` / `transfer.py` / `planner.py`（M6） | `rsf.planning.path` | EAIK walk 改用 `RobustIK`；評估 cuRobo tool bound 取代 |
| `judge.py`（placement/slip/contacts/tracking/caps） | `rsf.sim.judge` | 後端無關 |
| `reach_sweep.py`, reach_volume | `rsf.planning.teach.reachability` | 擴充為佈局最佳化工具 |
| `locate.py` + AprilgridDetector | `rsf.perception.locator` | |
| `isaacsim/build_env`, `sim_cell.py` | `rsf.sim.isaac` | 移植到 `SimBackend` 介面 |

**保留的工程原則**（D18/D22/triage rule）：
- 幾何只有一個來源，衍生值用算的、不轉抄；跨邊界的測試必須真的跨邊界呼叫。
- 每個檢查輸出 JSON 報告＋單一 verdict；以 TDD 與 negative check / mutant 驗證檢查本身。
- 低必要性＋高成本/高不確定的項目列為 contingent，等症狀出現才做。

---

## 2. 整體架構

### 2.1 分層

```
┌──────────────────────────────────────────────────────────────────────────┐
│ agent (later)   tools API (MCP / function tools) — 只包薄層，呼叫下面純函式 │
├──────────────────────────────────────────────────────────────────────────┤
│ orchestrator    workflow 執行器、scheduler、事件(refill/inspection)、失敗處理 │
├───────────────┬───────────────────────────┬──────────────────────────────┤
│ codegen       │ planning                   │ perception                   │
│ TMscript/ROS2 │  teach  (自動點位)          │  AprilGrid locator           │
│ RobotServer   │  path   (free/line/carry)   │  depth → ESDF (cuRobo)       │
│               │  realtime (MPC/replan)      │                              │
│               │  validate (mesh/dyn/caps)   │                              │
├───────────────┴───────────┬───────────────┴──────────────────────────────┤
│ kinematics backends       │ collision backends                            │
│  RobustIK(EAIK fork)       │  cuRobo (GPU, sphere + exact mesh)            │
│  cuRobo Kinematics (GPU)   │  CPU: MuJoCo / coal (dev, CI, fallback)       │
├───────────────────────────┴──────────────────────────────────────────────┤
│ compile   CellModel → cuRobo SceneCfg / robot yml │ MJCF │ URDF+PlanningScene │ USD │
├──────────────────────────────────────────────────────────────────────────┤
│ model     assets / layout / workflow schema, loaders, audit (pure Python) │
│ core      SE(3), frames, units (numpy only)                                │
└──────────────────────────────────────────────────────────────────────────┘
          sim backends (可替換)                      drivers (實機/模擬同介面)
          Kinematic | MuJoCo | Isaac Sim             SimDriver | TM | ROS2 | RobotServer
          viz: viser/meshcat (web) | RViz (ROS2) | MuJoCo viewer
```

### 2.2 Repo 結構

> 2026-11-12 的草稿規劃。實際結構見 `README.md` 的 Repository map；`skills/`、`drivers/`、`codegen/`、
> `perception/`、`viz/`、`agent/` 尚未建立（技能在各 cell 的 plugin、AprilGrid 在 `rsf.sim.vision`）。

```
robot_simulation_framework/
  pyproject.toml            # extras: [eaik] [curobo] [mujoco] [isaac] [ros2] [viz] [perception] [agent]
  src/rsf/
    core/                   # SE3, quaternion(wxyz), frames, units — numpy only
    model/                  # Asset, Layout, Workflow schema (pydantic), loader, audit, report
    importers/              # urdf, mesh(trimesh), usd(optional pxr), step(optional)
    compile/                # → curobo / mjcf / urdf+planning_scene / usd
    kinematics/             # KinematicsBackend: robust_eaik, curobo
    collision/              # CollisionBackend: curobo_mesh, mujoco, (coal)
    skills/                 # Skill 介面 + 通用技能 (approach/retreat/line/pick/place/sight)
    planning/
      teach/                # 自動點位: 候選 → IK 分支 → 篩選 → 分支鏈選擇
      path/                 # free / cartesian / sweep / jaw, 時間參數化
      validate/             # dense mesh check, caps, tracking
      realtime/             # MPC, replanning, correction
    sim/                    # SimBackend: kinematic, mujoco, isaac ; judge
    viz/                    # viser/meshcat, rviz
    drivers/                # RobotDriver: sim, ros2_json, tm, robotserver
    codegen/                # program export + round-trip parser
    perception/             # aprilgrid locator, depth mapping
    orchestrator/           # workflow runtime, scheduler, events
    agent/                  # (later) tool layer
  cells/tm_tray/            # 測試目標: assets/ layout.yaml workflow.yaml presets/ plugin.py
  tests/                    # CPU 測試；GPU 測試標記 @gpu，於 Kaggle 執行
  tools/kaggle/             # Kaggle GPU runner (cuRobo / benchmarks)
  docs/
```

### 2.3 核心資料模型

**Asset**（可重用的東西）
- `robot`：URDF + cuRobo robot yml（spheres、mesh 設定）＋ IK 設定（EAIK tip_link、idealize tol）＋ 限制（joint/vel/acc/jerk）。
- `tool`：gripper URDF/mesh、TCP frames、可動關節（jaw）、碰撞 proxy、可附著點。
- `object`：fixture/rack/jig/tray mesh 或 primitive，**命名 frame（features）**：grasp、place、label、approach。
- `sensor`：相機內參（含 D10 focal config 與距離查表）、hand-eye 外參。
- `generator`（plugin）：參數化物件（例：rack(layers, n_jigs, gap) → shelves + jigs + slots）。

**Layout**（某一台 cell）
- 場景樹：instance = asset + parent frame + pose；robot mount；presets/overrides（as_built/strict/recommended）。
- 每個 pose 可帶 `confirmed` / 不確定度（D11：日後由 AprilGrid 註冊修正）。

**Workflow**（要做什麼）
- 宣告式 YAML：`tasks`（pick/place/inspect/locate/move）、target selector（例 `incoming/*`）、使用的 skill、scheduler、events/signals、loops。
- 編譯成 **TaskGraph**（狀態機），每個節點 = skill instance(target, tool, params)。

**Skill**（流程原子）
- `skill.expand(world, target, params) -> SkillPlan`：一串 **Cartesian keyframes** ＋ 每段的約束
  （free / line / tool-bound / joint-only sweep）、碰撞群組與 margin、jaw 命令、attach/detach、速度/加速度 caps。
- TM tray 的 `pre_insert → insert → lift → close → withdraw → carry → …` 就是一個 plugin skill（由 `sequence.py` 移植）。

### 2.4 主要介面（草稿）

```python
class KinematicsBackend(Protocol):
    joint_names: list[str]; lower: np.ndarray; upper: np.ndarray
    def fk(self, q) -> SE3: ...
    def ik_all(self, T, *, within_limits=True, expand_2pi=True) -> list[np.ndarray]: ...  # 全部精確分支
    def ik_nearest(self, T, q_ref) -> np.ndarray | None: ...
    def jacobian(self, q) -> np.ndarray: ...

class CollisionBackend(Protocol):
    def set_world(self, world: CompiledWorld) -> None: ...
    def attach(self, link: str, obj: ObjectRef) -> None: ...; def detach(self, link: str) -> None: ...
    def check(self, Q, *, groups: dict[str, float]) -> CollisionReport: ...  # 每群組 margin、命中 link/obstacle、距離

class PathPlanner(Protocol):
    def plan_free(self, q0, q1 | T1, world, opts) -> Trajectory: ...
    def plan_cartesian(self, q0, keyframes, constraints, world, opts) -> Trajectory: ...

class SimBackend(Protocol):
    def load(self, world: CompiledWorld) -> None: ...
    def reset(self, state: CellState) -> None: ...
    def execute(self, traj: Trajectory) -> ExecReport: ...   # 回傳實際 q、接觸、物件 pose
    def render(self, camera: str) -> Image | Depth: ...
    def object_pose(self, name: str) -> SE3: ...

class RobotDriver(Protocol):          # 模擬與實機同介面 (D9)
    def state(self) -> JointState: ...
    def run(self, program: Program) -> ExecReport: ...
    def io(self, name: str, value) -> None: ...
```

---

## 3. 自動點位與路徑生成（離線）

```
workflow ──► TaskGraph ──► skill.expand ──► keyframes(候選, 容許度/對稱)
                                              │
                    IK 全分支 (RobustIK, CPU, 精確) ◄┘
                                              │ 篩選: joint limit / 碰撞(mesh) / 奇異點距離 / manipulability
                                              ▼
                       分支鏈選擇 (DP/graph search，跨同一 skill 的 keyframes：
                       不翻 elbow/wrist、單步 ≤ 門檻、總關節移動最小；泛化 M3 best_chain)
                                              ▼
                       Teach points 表 (joint + TCP pose + 來源/約束/margin)
                                              ▼
            段落規劃: free (cuRobo plan_cspace / plan_pose)
                      line (EAIK walk 或 cuRobo MultiWaypoint + ToolPoseBound，bench 後二選一)
                      carry (tool bound tilt/姿態 + D14 caps)、sweep (單軸)、jaw
                                              ▼
            時間參數化 (cuRobo timeopt / trapezoid caps) → validate (dense mesh、self、carried tray、caps)
                                              ▼
            模擬驗證 (SimBackend + judge) → 報告 → codegen
```

- **佈局評估工具**（reach sweep / reach volume / 基座高度掃描，D19/D23）：輸入 layout 變體，回傳 slot 覆蓋率與最差 margin；這也是 agent 架點的主要工具。
- **CPU / GPU 分工（使用者 2026-11-16 更新）**：點位（P3）與受限動作（shelf 內直線、tilt）在 CPU（EAIK + coal，逐點檢查）；
  **free move（點位之間、含攜帶 tray）由 cuRobo 在 GPU 離線規劃（目前用 Kaggle），上線時匯入路徑**。
  匯入的路徑一律再經 CPU gate（coal exact mesh、0.01 rad 取樣、與點位相同的規則）才成為軌跡。
  沒有 GPU 時（開發、CI）以 CPU RRT-Connect + shortcut + 平滑 作為 fallback（`--paths cpu`）。
- **重新生成**：layout 或 world 改變（新增靜態障礙、定位修正後的 slot pose）→ 只重算受影響的 task/segment（依賴圖快取）。

## 4. Real-time 路徑規劃（線上）

分三級，依偏差大小升級：

1. **定位修正（局部）**：AprilGrid 讀到的 slot pose 與名目差異在容許內 → 以 teach point 為模板重解 IK（同分支）＋重算 Cartesian 段（EAIK，< 10 ms CPU）。
2. **重規劃**：偏差大或 world 改變（depth 掃描到新障礙）→ cuRobo `plan_pose` / `plan_cspace`（warm start，CUDA graph），在 GPU 上目標 < 100 ms。
3. **反應式**：動態障礙 → cuRobo `ModelPredictiveControl` + perception ESDF；安全層做速度縮放/停止。

此部分延後實作；GPU 環境，含「移動物進入工作區」。

## 5. 虛擬環境（可替換）

| 後端 | 用途 | 執行環境 |
|---|---|---|
| `KinematicSim` | 無物理，FK 播放；CI、codegen round-trip | CPU |
| `MujocoSim` | 物理（tray on plate、摩擦、jaw）、接觸、相機 RGB/depth 渲染 | CPU（本環境主力） |
| `IsaacSim` | 高真實渲染（locating）、PhysX；移植原 `build_env` + `sim_cell` | GPU（保留支援） |
| Viz: viser/meshcat | Web 視覺化（雲端/無顯示器可用） | CPU |
| Viz: RViz | ROS2 MarkerArray / PlanningScene / JointState | 需 ROS2 |

- 所有後端由同一份 `CompiledWorld` 產生（MJCF / USD / PlanningScene），並各有一個「量回來比對 config」的測試（原 M1 的教訓：16 mm jig offset、308 mm rack depth 錯誤）。
- Judge（placement/slip/contacts/tracking/caps）只看 `ExecReport`，與後端無關。

## 6. 第三方程式：使用方式與可優化處

### 6.1 cuRobo（dev/main）
使用：robot/scene 設定、`MotionPlanner`（plan_cspace / plan_pose / plan_pose_waypoints）、`MultiWaypointMotionPlanner`、`ToolPoseBound`、`mesh_collision_check` / `mesh_self_collision_check`、`AttachmentManager`、MPC、perception（ESDF）。

可優化/回饋（**需在 Kaggle GPU 驗證**）：
1. **原專案的手刻 mesh 檢查可大幅簡化**：`check_cell.Curobo` 以 `RobotCollisionChecker` + `update_link_mesh` 自建 gripper mesh、拔 jaw link + `lock_joints`、carried tray 用 SAT 自算。dev/main 已有 `extra_collision_meshes`、`get_mesh_collision_links/distance`、`AttachmentManager`（附著物 mesh 寫進保留三角形槽）、`disable_link_collision`。
2. **建議新增 feature：per-obstacle(群組) margin ＋ link×obstacle 允許接觸矩陣（類似 MoveIt ACM）**。原專案為了「target jig/tray margin 0、其他 5 mm、jaw 可碰 target tray」必須建 3 個 checker、每組重新 load world、逐組呼叫，成本 ×(群組數×mesh 種類)。在 kernel 內以 obstacle id → margin / link mask 查表即可一次算完。
3. ~~Cartesian 直線精度~~：使用者確認 `ToolPoseBound` 已有 `center_weight`，在 cost 階段將 motion 拉到區間中央；
   real-time 端以此做 insert/withdraw，benchmark 時開啟。離線端（CPU）仍用 EAIK walk。
4. **CPU 路徑**：已評估，見 `docs/curobo_cpu_port.md`。Python 端的 CUDA kernel 是 53 個 `.cuh` `__global__`（約 13.5k 行，
   經 `curobolib/backends` 代理分派，可並列新增 `warp_backend`）；Warp kernel 在 CPU 上大致可跑（已實測 cuboid 正確，
   mesh 障礙有一個「球心在外、表面穿入回傳 0」的疑點待 GPU 比對）。估計：裝置管線＋FK＋self-collision＋parity 測試 3–4 週
   （得到 CPU 上的 cuRobo 碰撞與 mesh check）；再加 IK/TrajOpt/MotionPlanner 4–6 週。建議先做前者。
5. 原專案 `fix/self_collision_bug`（0 pair 除以零）在 dev/main 已有 guard（`self_collision_params.py:71`），不需再 merge。

### 6.2 EAIK（kthfan fork, robust.1）＋ robot_arm_motion_planner
- 原專案 `paths.py` 自寫 `refine`（有限差分 Jacobian、Python 迴圈、每個 sample 4 次 FK×7）與 `ik_branches`/`_turns`；`robot_arm_motion_planner.RobustIK` 已有解析 Jacobian、`refine_batch`、twin seeds、±2π 展開、`ik_batch`（C++ `IK_batched` 多執行緒）。**改用 RobustIK 可直接消除 findings/07-08 的精度問題並加速 walk / time_line**（後者對每個時間 sample 都做全分支 IK）。
- 建議回饋到 EAIK：
  - C++ 端提供 **`IK_nearest(T, q_seed)` / batched nearest**（含 ±2π 與 limit），避免 Python 端對全分支排序；Cartesian walk 只需要最近分支。
  - **C++ Newton refinement on original model**（目前在 Python `RobustIK`），可與 `IK_batched` 一起多執行緒化。
- 建議回饋到 robot_arm_motion_planner：
  - `KinematicsSolver` 目前綁 pinocchio（calibration 也用 pinocchio）；FK/Jacobian 可只用 EAIK 原模型（`fwdKin`），讓 planner 在無 pinocchio 環境可用。
  - `LINE_T` 的 hold-reconfiguration 可作為本框架的 line segment 後端；需要加入碰撞 callback（目前只看運動學）。
  - TM 專屬（TM pose、PTP_J、speed %）與通用規劃分層，讓 RobotServer 輸出成為本框架 codegen 的一個 target。

### 6.3 AprilgridDetector
- 用於 `Locator`（板子偵測＋pose）。可優化：
  - **ROI / prior pose 引導偵測**：layout 已知預期 board pose，可投影出 ROI 只在該區域偵測，提高速度與小標籤 (<20 px) 偵測率。
  - **多視角融合 API**（D2 提到的槓桿）與 focal config 切換內參的介面。
  - 以實際需求確認後再提出（需要先讀 crate 原始碼）。

## 7. 里程碑

每個里程碑都以 TM tray cell 驗收，並對照原專案已知結果（strict：reach 37/37、M3 collision 37/37、M4 74/74）。

| # | 內容 | 驗收 | 環境 |
|---|---|---|---|
| **P0** | Repo 骨架、pyproject/extras、CI(CPU)、Kaggle GPU runner、介面定義、`rsf.core` | CPU 測試綠燈；Kaggle 跑通 cuRobo smoke | CPU + Kaggle |
| **P1** | Cell model schema + loaders + audit；TM tray 移植成 `cells/tm_tray`（rack generator plugin、3 presets） | 由 YAML 產生的 box world 與原 `cell_core.scene.cell_boxes` 一致到 1e-12；所有 slot frames 一致 | CPU |
| **P2** | Kinematics（RobustIK）＋ collision backends（CPU exact mesh 為主、cuRobo 交叉驗證）；scene compilers（MJCF、cuRobo） | reach sweep strict 37/37（EAIK）；M3 collision 37/37（CPU）；Kaggle 上 cuRobo 結論一致 | CPU + Kaggle |
| **P3** | 自動點位：skill expand、候選、IK 分支、篩選、分支鏈選擇；layout 評估工具 | 37 slots 全部產生 teach points，`chain_q` 與原 M3 相同或更好 | CPU (+Kaggle) |
| **P4** | 路徑生成：free（cuRobo GPU 離線；CPU RRT-Connect fallback）/ line / tilt / jaw、時間參數化（D14/D30 caps）、CPU gate；整個 workflow 輸出一條時間化關節軌跡 | workflow 全程通過 CPU gate + caps | CPU + Kaggle |
| **P5** | Sim：Kinematic、MuJoCo（物理/相機）、Isaac adapter；judge；viz（viser + RViz adapter） | MuJoCo 中 transfer 通過 5 項 judge；Isaac adapter 能載入同一 world | CPU (+Isaac 主機) |
| **P6** | 定位：AprilGrid locator；執行前修正 → 重新生成、執行中局部修正（CPU） | 擾動 slot pose 後仍通過 judge | CPU |
| — | ~~Codegen~~ 暫緩，實機程式格式交給使用者設計；框架只輸出中立的點位表與時間化軌跡（JSON） | | |
| **P7** | Real-time（**延後**）：cuRobo 重規劃、MPC + ESDF、移動物、安全層；延遲 benchmark | 插入障礙物/移動物仍完成流程；延遲達標 | GPU / Kaggle |
| **P8** | Agent：tool API（MCP）、agent 協助架點（提案 → 驗證 → 渲染 → 人確認） | agent 能為新 slot 產生並驗證點位 | — |

P0–P2 先做；P3 起每個階段開始前再細化 todo。

**架子擾動後的定位與修正（strict，2026-11-23）**：layout entity 可 `displace`（剛體移動，slot 以 `slot_at` 由移動後的 jig 位姿重算），
情境 `layouts/perturbed/strict_racks_moved.yaml`：incoming、outgoing 架子各偏 5–7 mm、0.5–0.6°（slot 偏 3.6–10.3 mm）。
`tools/locate_slots.py`：每個 slot 在示教的遠讀位姿讀 AprilGrid → 修正 → 在修正後的近讀位姿再讀 → 每個架子擬合地面剛體移動 → 寫出修正後的 layout。
36/36 slot 讀到；近讀誤差 ≤ 0.05 mm／0.02°；擬合後所有 jig 與真值差 ≤ 0.015 mm。未修正的原計畫在擾動後的 cell 第一次取料即撞 jig（0/24）；
以修正後 layout 重新示教（26/26）、CPU 規劃（0 problems），在擾動後的 cell 模擬全流程 24/24 通過（最差放置 0.39 mm、滑移 0.40 mm）。
目前為理想影像、執行前一次定位、架子水平；見 `docs/p6_locating.md`。

**放寬攜帶限制（2026-11-23）**：攜帶 tilt 15° → 5°（使用者：15° 以下都可接受，暫定 5°）；tilt 動作重新調整為以 flange 為中心的純旋轉
（flange 不位移，原本 52.4 mm，仍全程 ≥ 10 mm）。攜帶規則改為相對水平：板面傾角 ≤ 15°，且重力不得把 tray 往開口側拉
（`motion.carry_tilt_reference: level`；cuRobo ToolPoseBound 取同時滿足兩條規則的最大錐）。手臂組態固定為 J3 +1／J5 −1
（as_built 自動選到 J3 −1 時有一段 free move 規劃不出來）。三個 layout 重新示教 74/74、26/26，CPU 規劃 0 problems，MuJoCo 全流程皆 24/24
（最差放置 0.48 mm、滑移 0.50 mm、追蹤 0.44 mm、攜帶傾角 14.9°）。見 `docs/p5_sim.md`。

**Isaac Sim adapter（2026-11-23）**：World → USD（`rsf.compile.usd`：UsdPhysics articulation 由 URDF 直接寫出、
fixed joint 合併成剛體、凸包碰撞、與 MuJoCo 相同的質量／摩擦／伺服）＋ `rsf.sim.isaac_cell.IsaacCell`（omni.physx 步進、
omni.physics.tensors 讀寫、同一 `ExecReport` 與 judge）；`tools/sim_workflow.py --backend isaac`。編譯器測試 8 項（FK 與 URDF < 1e-6、
質量與 MuJoCo 一致）。Isaac Sim 6.2 在 Kaggle T4 headless 執行（僅物理；無 Vulkan，不能出相機影像）：strict 第一個 cycle 2/2 通過，與 MuJoCo 追蹤／傾角一致到 0.01 mm／0.01°，放置／滑移差 ≤ 0.6 mm。Kaggle 上加裝 NVIDIA 圖形函式庫＋EGL vendor 後 Vulkan 可用，RTX 渲染可行；預建環境 `isaac62`（含 shader cache）讓每次執行由約 1350 s 降到 638 s、第一張影像由 630 s 降到 1.6 s。見 `docs/p5_isaac.md`。

**130 mm tray 與小心攜帶（2026-11-19）**：tray 高度 130 mm；tilt 重新調整（flange 52.4 mm，64 取樣，≥ 10 mm）；三個 layout 37/37、74/74、26/26。
攜帶限制 15°（CPU 受限取樣、cuRobo ToolPoseBound、碰撞檢查）；手臂姿態只在空手時切換（示教變體）。三個 layout 全流程模擬皆 24/24，
模擬中實測攜帶傾角 ≤ 15.0°。

**P6 準備（2026-11-19）**：模擬中的 AprilGrid 讀取：每個 slot 的 label（36h11 4×1）印在 jig 上、EIH 相機（手眼標定＋焦距組態）、
使用者的 AprilgridDetector；strict 74/74 讀取找到正確 board，理想影像下位姿誤差 0.04 mm / 0.07°。見 `docs/p6_locating.md`。

**P5（2026-11-19）**：MuJoCo 重播規劃軌跡（tray 為自由物體，摩擦、重力、接觸；手臂 PID 追蹤；夾爪剛性伺服）＋ D30 五項 judge
（`rsf.sim`、`tools/sim_workflow.py`、`tools/render_sim.py`）。strict 第一個 cycle 2/2 通過（放置 0.25 mm、滑移 0.45 mm）；
推板摩擦降到 0.1 仍通過（tray 靠在推板上，不靠摩擦）。全流程模擬找到 P4 的漏洞：free 攜帶動作未限制工具姿態，8/24 段 tray 傾斜超過 30°，
其中一段 tray 掉落；攜帶速度上限缺角速度。已加入攜帶姿態限制（`carry_max_tilt_deg`）、攜帶物對手臂本身的碰撞檢查、角速度上限。修正後三個 layout 全流程（各 24 個 transfer）
模擬皆 24/24 通過（最差放置 0.51 mm、滑移 0.46 mm、追蹤 0.46 mm）。見 `docs/p5_sim.md`。

**P4（2026-11-16）**：關節運轉餘裕 5°（URDF 極限內縮，`robot.limit_margin_deg`）；workflow → segments（free/line/path/hold）→ 計時 → CPU gate →
`rsf.trajectory/1`（`tools/plan_workflow.py`）。點位改為整個 run 同一手臂組態（J3 −1、J5 +1，自動選擇）並以整圈調整讓 sequence 間移動最少
（原本取料/放料分支不同，攜帶 tray 時 elbow/wrist 翻轉）。free move：cuRobo job（`tools/kaggle/curobo_free_plan.py`）或 CPU fallback。
strict 全流程（12 cycles）CPU 結果見 `docs/p4_paths.md`。

**P3（2026-11-13）**：通用自動點位引擎 `rsf.planning.teach`（skill → waypoint 序列；IK 全分支、直線／路徑逐點連續 IK 與碰撞、
最大最差裕量的分支鏈、點位表 JSON/CSV、`python -m rsf teach`），TM 取放 skill。三個 layout 皆 74/74 序列（37 取＋37 放），
workflow 一次運行 26/26。tilt 改為全程（含終點）離障礙 ≥ 10 mm：支點 (y 120, z 60) mm、後退 35 mm，flange 位移 11.6 mm。
見 `docs/p3_teach.md`。

**P2（2026-11-12）**：EAIK＋RobustIK 運動學後端、可達性掃描、CPU 精確 mesh 碰撞（coal，含允許接觸規則、附著物、自碰撞）、
M3 碰撞判定移植、World→MuJoCo 編譯器。D29 夾爪 strict／as_built 皆 37/37；裝上推板後 D28 insert 深度 0/37（推板插入 tray 9.5 mm），
改為推板取放流程（insert 停在推板前 2 mm → jaw 張開推 tray 至座位後 2 mm → 貼推板攜帶；放料淺 2 mm；tilt 15° 以單一動作往下後方傾斜，flange 位移 8 mm），
取料＋放料三個 layout 皆 37/37。推板實機位置待量測。結果與 cuRobo 交叉驗證見 `docs/p2_results.md`。

**P1 完成（2026-11-12）**：通用 cell model（`rsf.model.world/layout/select/workflow/audit/registry`、`rsf.orchestrator.scheduler`、
`rsf.compile.curobo`、CLI `python -m rsf`），TM tray 以 plugin 形式移植（`cells/tm_tray/plugin/tm_tray`、`layouts/`、`workflows/`）。
與凍結的原 `cell_core`（`tests/reference/ai_robotics_v2`）比對到 1e-12；格式與擴充點見 `docs/cell_model.md`。

**P0 進度（2026-11-12）**：`rsf.core.se3`、`rsf.model.urdf`（解析/FK/編輯/輸出）、TM tray 資產組合
（`tools/assemble_tm_tray_assets.py`，見 `docs/assets.md`）與其測試；Kaggle CLI 認證已確認可用。

## 8. 測試策略

- CPU：純邏輯、EAIK、MuJoCo、codegen 全部在本環境跑（pytest）。
- GPU：`@pytest.mark.gpu` 測試與 benchmark 透過 Kaggle（2×T4）執行，結果（JSON 報告）存回 repo `reports/`。
- 交叉驗證：同一問題用兩個後端（EAIK vs cuRobo IK、MuJoCo vs cuRobo 碰撞）比對；negative check 確認檢查會失敗。

## 9. 待決問題

### 已決定（使用者 2026-11-12）
- 3D 輸入：URDF、mesh、USD。
  → 2026-11-26：零件（治具、物件）可用 USD 組合 mesh 輸入（layout `model:`，`rsf.model.usd_model`）；
  機械手臂仍為 URDF。使用者無法提供 USD 檔，以 jig/tray 轉成的 USD（cm、Y-up、巢狀變換、instancing、
  CollisionAPI）驗證與 URDF 結果一致（`tests/test_usd_model.py`）。
- 實機程式格式：暫緩，由使用者設計。
- 離線（自動點位、路徑生成）無 GPU；real-time 有 GPU，延後實作（含移動物進入工作區）。
  → 2026-11-16 更新：free move 路徑由 cuRobo 在 GPU 離線計算（先用 Kaggle），上線時匯入。
- 實機關節極限與 URDF 相同；需要餘裕，暫定 5°（2026-11-16）。
- Tray 高度更正為 130 mm（2026-11-19）；tray 需小心對待、必須確保：攜帶時姿態維持在取料傾角 15° 內（`carry_max_tilt_deg`），
  CPU 規劃器、cuRobo（ToolPoseBound）與碰撞檢查都強制。
- 後續優先順序（2026-11-19）：1. ToolPoseBound　2. AprilGrid 讀取　3. Isaac Sim adapter　4. 物理參數。
- 新的靜態障礙：離線隨時自動重新生成，不需 real-time。
- 定位：補償架點與實機誤差，執行前與執行中兩種模式都要。
- CPU fallback：依提議（功能正確等級的 CPU 碰撞 + RRT-Connect，cuRobo 交叉驗證）。
- 原專案的限制（air-gapped、numpy 1.26、git 由使用者操作、CrossHost）不再適用。
- D30 作為 M6 的參考決策。
- `ToolPoseBound` 已有 `center_weight`（§6.1 #3）。

- Tray 尺寸：保留原設定（0.1329 × 0.2470，2.5 mm recess）；高度 2026-11-19 更正為 0.130 m（原 0.0851）。
- 實機已裝 air cylinder：預設模型包含；夾爪實際運行不會全關 → `tool.operating_range`（min 0.021）。
- 實作原則：保持彈性與可擴充性（資產組合改由 `cells/<cell>/asset_spec.yaml` 宣告；幾何生成器放在 `rsf.model.primitives`）。

### 待辦（使用者 2026-11-22，尚未實作）
- **AprilGrid 真實影像條件**：模糊、雜訊、曝光、陰影、遮擋、光照（detector 的 `tools/agsim/render.py` degradations）；
  偵測率與位姿誤差對這些條件的關係。
- ~~slot 位置擾動後的定位與修正~~：strict 完成（2026-11-23，見下方進度）；as_built／recommended、執行中逐 slot 讀取尚未做。
- ~~放寬攜帶限制~~：完成（2026-11-23，見下方進度）。
- **實時檢視（P7，2026-11-25）**：模擬端渲染（Isaac RTX）、HTTP API + 瀏覽器檢視器、可在另一台主機看；
  第一版 MJPEG 完成（`docs/live_view.md`）；Isaac Sim 5.0 渲染以 `render_mode=play` 驗證（2026-11-25）。
  已加：暫停/單步/調速/停止、Python debug console（token、預設關閉、不限 localhost、可選 HTTPS）、`--debugpy`；
  可移動視角（共用相機、不需 token；鍵盤 WASD/TG、滑鼠拖曳轉向、雙擊 FPS mouse look、平移、滾輪、Ctrl+滾輪 FOV）；
  錄影（server 端 MP4，依模擬時間、隨暫停而暫停）；Isaac 預設 `render_mode=step`（live_probe 2026-11-25）。
  WebRTC 傳輸（`--live-webrtc`，aiortc，VP8/H.264，UDP；連不上時自動退回 MJPEG；本機延遲 35 ms vs MJPEG 69 ms，
  頻寬約 1/5～1/10，2026-11-26）。之後：多視角同時顯示、WebSocket 等。

### 仍待確認
- cuRobo CPU 移植是否進行、範圍（見 `docs/curobo_cpu_port.md` §4），以及是否把 kthfan/curobo 加入此 session。
- Air cylinder 碰撞 mesh 是否也帶有 tm5s.usd 的安裝偏移（需實機量測）。