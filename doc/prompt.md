# Vibe Coding 自动化开发起始 Prompt

## 1. 任务概述

你将作为 **主 Agent (Master Agent)** 带领你的 **子 Agent (Sub-Agents)** 团队，在没有人工干预的情况下，完全自动化地开发和测试 **NTCyber AI 全栈 Web 应用**。该应用旨在将机器学习模型（垃圾短信/邮件检测、恶意软件检测）集成到一个可交互的 Web 平台中。

### 输入文档
1. **需求规范**: [doc/proposal.md](file:///C:/Users/yongh/Documents/INTI%20Degree/SEM%203/COS30049/Assignment%202/COS30049---Assignment-G1/doc/proposal.md)
2. **详细设计文档**: [doc/detailed-design.md](file:///C:/Users/yongh/Documents/INTI%20Degree/SEM%203/COS30049/Assignment%202/COS30049---Assignment-G1/doc/detailed-design.md)
3. **模块与任务清单**: [doc/tasks/progress.md](file:///C:/Users/yongh/Documents/INTI%20Degree/SEM%203/COS30049/Assignment%202/COS30049---Assignment-G1/doc/tasks/progress.md) 目录下的所有模块任务描述。

---

## 2. 团队角色与分工

### 2.1 主 Agent (Master/Main Agent)
- **核心职责**: 项目经理与技术架构师，跟踪整体开发进度，管控系统级集成质量。
- **具体任务**:
  1. **进度跟踪**: 读取 [doc/tasks/progress.md](file:///C:/Users/yongh/Documents/INTI%20Degree/SEM%203/COS30049/Assignment%202/COS30049---Assignment-G1/doc/tasks/progress.md) 并分析各个模块之间的依赖关系（例如先开发基础工具和数据管道，再开发路由和页面）。
  2. **任务指派**: 创建并调度 **子 Agent (Sub-Agent)** 执行具体模块任务（如前端 API/工具类开发、后端模型加载器开发等）。
  3. **质量管控**: 每次子 Agent 完成模块后，合并代码并运行测试，更新 [doc/tasks/progress.md](file:///C:/Users/yongh/Documents/INTI%20Degree/SEM%203/COS30049/Assignment%202/COS30049---Assignment-G1/doc/tasks/progress.md) 状态。
  4. **全系统集成与测试**: 负责集成所有模块并运行全套测试套件（前端 Vitest + 后端 Pytest）。

### 2.2 子 Agent (Sub-Agent)
- **核心职责**: 单一模块的开发工程师，专注于实现具体任务清单，并编写完善的单元测试。
- **具体任务**:
  1. **需求理解**: 读取分配到的模块任务文件（如 `doc/tasks/home.md` 等），明确该模块的功能、数据库接口、UI 规范和出错边界。
  2. **编码开发**: 严格按照详细设计规范在指定路径编写代码。
  3. **单元测试编写**: 
     - 后端：使用 `pytest` 编写高覆盖率单元测试，采用 mock 机制模拟各种异常（如模型缺失、数据库损坏）。
     - 前端：使用 `vitest` + `React Testing Library` 编写组件与逻辑单元测试，覆盖状态变化、表单验证和 API 调用。
  4. **测试与修复**: 自主在测试隔离区运行测试命令，遇到报错自主修复代码，直到该模块的所有测试 100% 通过。
  5. **反馈交付**: 将模块代码和测试文件交付给主 Agent，并提请更新进度。

---

## 3. 工作流程与规范

### 第一阶段：初始化与环境配置
1. **项目骨架初始化**:
   - 在根目录下创建前端目录 `frontend/`，使用 Vite 创建 React.js + JavaScript 应用项目，并配置 Material UI (MUI v5) 和 Plotly.js (`react-plotly.js`)。
   - 在根目录下创建后端目录 `backend/`，初始化 Python FastAPI 环境，创建好基础的目录结构（`routers/`、`models/`、`preprocessing/`、`database/` 等）。
2. **测试框架配置**:
   - 前端配置 `vitest` + `@testing-library/react`，并加入 npm scripts：`npm run test`。
   - 后端安装 `pytest`、`httpx`、`pytest-asyncio` 等依赖，配置好 `pytest` 运行命令。
3. **静态资源与模型准备**:
   - 确认机器学习模型文件能够正确加载自：`outputs/models/`，包括：
     - `rf_spam.pkl`
     - `nb_spam.pkl`
     - `logistic_regression_spam.pkl`
     - `svm_malware.pkl`
     - `kmeans_malware.pkl`
     - `dbscan_malware.pkl`
   - 确认预处理标定器文件能够正确加载自：`data/processed/malmem_scaler.pkl`。

### 第二阶段：迭代式模块开发与集成
1. 主 Agent 扫描 [doc/tasks/progress.md](file:///C:/Users/yongh/Documents/INTI%20Degree/SEM%203/COS30049/Assignment%202/COS30049---Assignment-G1/doc/tasks/progress.md)，获取下一组待执行的未完成任务。
2. 针对每个任务模块，主 Agent 启动一个子 Agent 并传入该模块的任务描述（例如 `doc/tasks/database.md`）。
3. 子 Agent 按照设计文档及任务描述，进行编码实现及编写单测。
4. 子 Agent 运行测试，遇到编译/运行/测试失败时，通过分析报错信息在工作区内自动进行代码修正，不允许中止或请求人工。
5. 测试通过后，主 Agent 更新对应任务的状态（打勾），并使用 Git 进行局部 Commit 以持久化进度。

### 第三阶段：系统联调与终期验收
1. 主 Agent 将前端与后端进行联调测试，重点测试交互流畅度、API 数据链路完整性、跨域配置（CORS）以及响应式布局。
2. 批量与单样本的恶意软件检测 CSV 导出格式正确无误。
3. 整体测试套件一键运行（Pytest + Vitest），必须确保全通过且覆盖率达到高标准。

---

## 4. 技术标准与实现细节

### 4.1 前端设计要求
- **Aesthetics & UI**: 风格现代高端，应用 MUI 的丰富组件（Grid, DataGrid, Stepper, Accordion 等），并添加细微悬停动画和精致配色。
- **可交互图表 (Plotly.js)**:
  - 必须包含至少 8 种图表类型：仪表盘图 (Gauge)、环形图 (Pie/Donut)、雷达图 (Radar)、水平条形图 (Horizontal Bar)、直方图 (Histogram)、分组条形图 (Grouped Bar)、散点图 (PCA 2D Scatter)、折线图 (ROC Line)。
  - 支持交互放大、过滤和悬停提示，后端数据更新时前端图表实时无缝更新。
- **输入校验**: 前端需要对输入文本字符范围、文件大小和特征字段进行严格的就地验证，防止将错误数据发往后端。

### 4.2 后端设计要求
- **异常捕获与状态码**:
  - 输入参数格式缺失或非法：返回 `422` / `400`。
  - 模型文件未成功载入：返回 `503` (Service Unavailable) 错误信息 `{ "error": "Model not loaded" }`。
  - 路由与服务解耦，异常处理器集中处理捕获的各类异常。
- **数据库持久化**:
  - 使用 SQLite (SQLAlchemy ORM) 来保存和查询历史检测结果。
  - 提供检索过滤（支持日期范围、类型及关键词搜索）以及一键清除数据库的接口。

### 4.3 零人工干预原则 (Zero-Human Interaction)
- 整个生成、修改、调试、构建及测试执行路径必须**完全自闭环**。
- 所有工具链、运行环境及依赖项的声明应尽可能详尽，确保执行时不弹出交互式确认框。

请依据以上规范与步骤开始项目，直到 [doc/tasks/progress.md](file:///C:/Users/yongh/Documents/INTI%20Degree/SEM%203/COS30049/Assignment%202/COS30049---Assignment-G1/doc/tasks/progress.md) 的所有模块打勾完成并全部测试通过！
