# 详细设计文档

## 1. 概述

本设计文档基于 `doc/proposal.md` 的需求说明，详细描述 NTCyber AI Web 应用的架构、模块划分、接口规范、数据流、测试策略和实现细节。目标是让各模块相互独立、可单独测试，并能清晰指导前端与后端开发。

本系统由前端 React + MUI 应用、后端 FastAPI 服务、SQLite 存储、以及已训练的机器学习模型组成。核心功能包括垃圾短信检测、恶意软件检测、数据可视化探索、模型性能仪表盘和历史记录管理。

---

## 2. 总体架构

### 2.1 架构层次

- Frontend: React.js + Vite
- UI 组件: Material UI v5
- 数据可视化: Plotly.js via react-plotly.js
- HTTP 客户端: Axios
- Backend: FastAPI
- 数据库: SQLite via SQLAlchemy
- 机器学习模型: scikit-learn `.pkl`
- 预处理: NLTK、pandas、scikit-learn

### 2.2 模块划分

前端模块:
- `pages/` 页面级模块
- `components/` 可复用组件
- `api/` Axios 请求封装
- `utils/` 通用工具函数

后端模块:
- `routers/` 路由层
- `models/` 模型加载与推理层
- `preprocessing/` 数据预处理层
- `database/` 数据持久化层
- `schemas/` 请求/响应 Pydantic 模型
- `services/` 业务服务层（可选，用于解耦路由与模型逻辑）
- `data/` 静态探索与指标数据

数据库模块:
- `DetectionRecord` ORM 模型，用于历史记录存储
- 基于 SQLite 的持久化和查询接口

### 2.3 关键设计原则

- 模块独立：前端页面、后端路由、模型推理、数据存储、数据可视化各自独立。
- 可测试性：每个模块都有明确输入、输出和错误边界，便于单元测试。
- 异常处理：前端和后端均分别进行输入校验与错误返回。
- 数据复用：前端所有图表数据通过后端预计算接口提供，避免重复计算。
- 响应式设计：使用 MUI Grid 保证不同屏幕尺寸下布局自适应。

---

## 3. 模块设计

### 3.1 前端模块

#### 3.1.1 页面模块

- `Home.jsx`
  - 展示欢迎信息、CTA 按钮、统计卡片
  - 调用 `GET /api/stats`
  - 不包含任何 AI 推理请求

- `SpamDetection.jsx`
  - 文本输入与文件上传双模式表单
  - 模型选择 ToggleButtonGroup
  - 提交按钮触发 `POST /api/spam/predict`
  - 展示多模型结果卡片与图表
  - 支持 CSV 导出结果

- `MalwareDetection.jsx`
  - 手动输入与 CSV 批量上传双模式
  - 调用 `POST /api/malware/predict` 或 `POST /api/malware/predict/batch`
  - 单样本模式显示饼图、仪表盘、雷达图
  - 批量模式显示 DataGrid 与分布图
  - 支持 CSV 导出结果

- `DataExploration.jsx`
  - 数据集选择下拉框
  - 调用 `GET /api/explore/stats`
  - 渲染图表：饼图、直方图、热图、柱状图

- `ModelDashboard.jsx`
  - 调用 `GET /api/dashboard/metrics`
  - 渲染模型性能对比、ROC 曲线、交叉验证误差、混淆矩阵、特征重要性、聚类可视化等

- `History.jsx`
  - 调用 `GET /api/history`，支持过滤参数
  - 显示历史记录 DataGrid
  - 提供清除历史 `DELETE /api/history`
  - 导出历史 CSV

#### 3.1.2 组件模块

- `NavBar.jsx`
  - 顶部导航栏，包含页面链接
  - 负责响应式导航展示

- `SpamResultCard.jsx`
  - 接收单个模型结果并渲染：模型名称、判决、置信度图、Top features 图

- `MalwareResultCard.jsx`
  - 单样本结果视图，负责展示 SVM/DBSCAN/雷达图

- `FileUploadZone.jsx`
  - 支持文件拖拽与选择，前端验证类型和大小
  - 返回文件内容与错误信息

- `ExportButton.jsx`
  - 用于生成 CSV 下载
  - 通用导出组件，支持传入字段和数据

- `FeatureInputAccordion.jsx`
  - 用于展示 39 个恶意软件特征字段
  - 每个折叠面板包含若干数值输入项

- `ModelSelectToggle.jsx`
  - 多选模型切换按钮组
  - 保证至少选择一个模型

#### 3.1.3 API 与工具

- `api/axios.js`
  - 统一 Axios 实例配置 Base URL 和拦截器

- `utils/validation.js`
  - Spam 文本长度校验
  - 文件类型与大小校验
  - 39 个恶意软件特征数值校验
  - CSV 格式检查

- `utils/csv.js`
  - 将前端结果对象转换为 CSV
  - 下载链接生成

- `utils/chartConfig.js`
  - 各类 Plotly 图表的默认配置与布局

### 3.2 后端模块

#### 3.2.1 路由层

- `routers/spam.py`
  - `POST /api/spam/predict`
  - `GET /api/spam/models`

- `routers/malware.py`
  - `POST /api/malware/predict`
  - `POST /api/malware/predict/batch`

- `routers/explore.py`
  - `GET /api/explore/stats`

- `routers/dashboard.py`
  - `GET /api/dashboard/metrics`

- `routers/history.py`
  - `GET /api/history`
  - `DELETE /api/history`

#### 3.2.2 模型加载与推理层

- `models/loader.py`
  - 启动时加载所有 `.pkl` 模型文件
  - 提供 `get_spam_model(name)`、`get_malware_svm()`、`get_malware_dbscan()` 接口
  - 加载失败抛出自定义异常，FastAPI 捕获并返回 503

- `models/spam_predictor.py`
  - 接收原始文本
  - 调用预处理管道得到特征向量
  - 依次运行所选模型
  - 返回分类结果、概率、Top features
  - Top features 从模型 coef / feature importance 与文本 TF-IDF 特征共同计算

- `models/malware_predictor.py`
  - 单样本模式：标准化输入特征后调用 SVM
  - DBSCAN 异常检测：返回 `is_anomaly`、`cluster_id`、`distance_to_centroid`
  - 批量模式：对 CSV 多行输入运行相同逻辑，按行返回结果列表

#### 3.2.3 预处理层

- `preprocessing/pipeline.py`
  - 文本预处理：清洗文本、tokenize、停用词移除、TF-IDF 向量化
  - 训练时的特征顺序一致，用于前端 CSV 校验
  - 39 个恶意软件特征标准化与缺失值处理规则
  - 提供 `transform_spam_text(text)`、`validate_malware_features(features)`、`validate_malware_csv(csv_df)` 等函数

#### 3.2.4 数据持久化层

- `database/db.py`
  - SQLAlchemy engine 与 Session 管理
  - `get_db()` 依赖注入

- `database/models.py`
  - `DetectionRecord` ORM 类，字段包括：`id`, `timestamp`, `type`, `input_preview`, `models`, `verdict`, `confidence`, `raw_data`

- `services/history_service.py`（建议）
  - 提供历史记录创建、查询、删除
  - 支持按检测类型、日期范围、关键字搜索过滤

### 3.3 数据与文件模块

- `data/metrics/` 目录
  - 存放预计算 JSON 或 CSV
  - 包含探索页面与仪表盘页面的数据
  - 例如：`explore_sms_stats.json`, `dashboard_classification_metrics.json`

- `backend/main.py`
  - FastAPI 应用入口
  - 载入模型和数据库
  - 注册路由
  - 添加异常处理器与 CORS 设置

### 3.4 独立测试模块

- 前端测试
  - `SpamDetection.jsx` 表单验证与条件渲染测试
  - `MalwareDetection.jsx` 手动输入与 CSV 上传切换测试
  - `History.jsx` 过滤与导出按钮显示
  - `utils/validation.js` 单元测试

- 后端测试
  - `routers/spam.py` 的预测与错误响应测试
  - `routers/malware.py` 单样本与批量预测测试
  - `routers/history.py` 查询过滤和删除测试
  - `models/loader.py` 模型加载失败测试
  - `preprocessing/pipeline.py` 输入校验测试

---

## 4. 页面详细设计

### 4.1 首页 `Home.jsx`

#### 功能
- 展示应用名称、标语和功能简介
- 显示统计卡片：检测总数、垃圾邮件检测数、恶意软件检测数
- CTA 按钮：跳转到 Spam 检测与 Malware 检测页面

#### 数据来源
- 调用 `GET /api/stats`
- 统计由后端从 SQLite 历史记录计算

#### UI 组件
- MUI `Typography`、`Button`、`Card`
- MUI `Grid` 布局

#### 独立测试点
- 是否正确调用接口并渲染统计值
- CTA 是否能正确导航
- 若接口失败，是否展示警告提示

### 4.2 垃圾邮件检测页面 `SpamDetection.jsx`

#### 输入模式

1. 文本输入 Tab
   - MUI `TextareaAutosize` 或 `TextField` multiline
   - 字符计数器
   - 验证规则：10~10000 字符
2. 文件上传 Tab
   - `FileUploadZone.jsx` 支持 `.txt` / `.eml`
   - 最大 2MB
   - 客户端读取并展示预览

#### 模型选择
- 多选 `ToggleButtonGroup`
- 选项：Random Forest、Naive Bayes、Logistic Regression
- 至少选中一个模型
- 将后端模型标识映射为 `random_forest`, `naive_bayes`, `logistic_regression`

#### 提交逻辑
- 点击 `Analyse Message`
- 若前端校验通过，则发起 `POST /api/spam/predict`
- 请求体：`{ text, models }`
- 请求期间显示 MUI `LinearProgress`
- 防止重复提交

#### 结果展示
- 每个模型使用 `SpamResultCard.jsx`
- 内容包括：模型名称徽章、判决标签、置信度仪表盘、Top features 水平条形图
- 使用 Plotly.js `indicator` 与 `bar` 图表
- 多模型时使用 MUI `Grid` 排列

#### 导出
- `ExportButton.jsx`
- 生成 CSV: `input_text`（截断 200 字符）、`model_name`、`verdict`、`probability`、`timestamp`

#### 交互细节
- 如果后端返回错误，使用 MUI `Alert` 显示错误信息
- 模型列表从 `GET /api/spam/models` 获取（可选），或前端硬编码与后端一致

#### 独立测试点
- 文本/文件输入验证
- 模型选择必选校验
- 请求成功后结果卡片是否展示
- 导出 CSV 是否包含预期字段

### 4.3 恶意软件检测页面 `MalwareDetection.jsx`

#### 输入模式

1. 手动特征表单 Tab
   - 39 个数值字段
   - 字段按逻辑分组显示在 MUI `Accordion` 中
   - 每个字段使用 `TextField type="number"`
   - 按钮：`Fill with Benign Example`, `Fill with Malware Example`

2. CSV 上传 Tab
   - `FileUploadZone.jsx` 支持 `.csv`
   - 验证列数是否为 39 且数值合法
   - 显示 `DataGrid` 预览前五行
   - 批量模式同一次请求全部行

#### 提交逻辑
- 单样本：`POST /api/malware/predict`，请求体含 `features`
- 批量：`POST /api/malware/predict/batch`，请求体含 `rows`
- 显示 `LinearProgress`

#### 单样本结果
- SVM 分类：Plotly 饼图/环形图
- DBSCAN 异常：Plotly 仪表盘图
- 特征雷达：Plotly `scatterpolar` 显示当前样本与类别均值对比

#### 批量结果
- `DataGrid` 表格：行号、SVM 预测、DBSCAN 结果、异常标志
- 批量分布条形图：SVM 预测分布
- 异常计数徽章

#### 导出
- `ExportButton.jsx`
- CSV 包含：`sample_index`、`svm_prediction`、`dbscan_result`、`anomaly_flag`、`timestamp`

#### 独立测试点
- 手动表单与 CSV 上传切换
- CSV 校验与预览
- 单样本与批量接口请求
- 结果展示组件是否正常渲染

### 4.4 数据探索页面 `DataExploration.jsx`

#### 功能
- 允许用户选择数据集：SMS Spam、Enron Email、CIC-MalMem-2022
- 根据选择渲染对应图表
- 前端不做复杂计算，完全依赖后端静态数据

#### 图表
- 类别分布：饼图
- 消息长度分布：直方图
- 关键词频率热图：热图
- 恶意软件类别分布：柱状图
- 特征方差：水平柱状图
- 特征相关性：热图

#### 技术细节
- `GET /api/explore/stats` 返回结构化 JSON
- 前端从返回数据构造 Plotly 图表
- 图表均支持交互：缩放、悬停、图例切换

#### 独立测试点
- 数据集切换是否触发正确接口
- 所有预期图表和图例是否渲染
- 接口错误时是否展示警告

### 4.5 模型仪表盘页面 `ModelDashboard.jsx`

#### 功能
- 展示训练模型性能指标
- 提供比较视图和可视化结果

#### 图表
- 模型性能对比：分组柱状图
- ROC 曲线：折线图
- 交叉验证数据：误差条柱状图
- 混淆矩阵：热图（可选择不同分类器）
- 随机森林特征重要性：水平柱状图
- 逻辑回归概率分布：直方图
- K-Means 聚类可视化：散点图（PCA 2D）
- DBSCAN 聚类可视化：散点图，突出异常点

#### 数据来源
- `GET /api/dashboard/metrics`
- 由后端预计算并存储为 JSON/CSV

#### 独立测试点
- 接口数据是否正确映射到图表
- 混淆矩阵切换是否有效
- 各图表是否响应交互操作

### 4.6 历史记录页面 `History.jsx`

#### 功能
- 显示当前浏览器会话的检测历史
- 可按检测类型、时间范围、关键词筛选
- 支持清除历史与导出 CSV

#### UI 组件
- `DataGrid`
- `Select` 检测类型过滤
- `DatePicker` 日期范围过滤
- 搜索输入框
- `Dialog` 确认清除历史

#### 数据接口
- `GET /api/history?type=&start_date=&end_date=&query=`
- `DELETE /api/history`

#### 导出字段
- `timestamp`, `detection_type`, `input_preview`, `models_used`, `verdict`, `confidence`

#### 独立测试点
- 筛选条件是否工作
- 清除历史后表格为空
- 导出结果是否完整

---

## 5. 后端 API 设计

### 5.1 请求/响应模式

所有 POST 接口接受 `application/json`，所有响应返回 JSON。
错误响应采用 `{ "error": "message" }`。

### 5.2 `/api/spam/predict`

请求体:
```json
{
  "text": "...",
  "models": ["random_forest", "naive_bayes", "logistic_regression"]
}
```

响应体:
```json
{
  "results": [
    {
      "model": "random_forest",
      "verdict": "spam",
      "probability": 0.97,
      "top_features": [
        {"word": "prize", "importance": 0.42},
        {"word": "free", "importance": 0.38}
      ]
    }
  ],
  "detection_id": "abc123",
  "timestamp": "2026-06-10T14:30:00Z"
}
```

### 5.3 `/api/spam/models`

响应体:
```json
{
  "models": [
    {"id": "random_forest", "name": "Random Forest", "description": "Recommended — 98.39% accuracy"},
    {"id": "naive_bayes", "name": "Naive Bayes", "description": "Fastest"},
    {"id": "logistic_regression", "name": "Logistic Regression", "description": "Probability score"}
  ]
}
```

### 5.4 `/api/malware/predict`

请求体:
```json
{
  "features": {
    "nsemaphore": 12.0,
    "ntimer": 3.0,
    "nmutant": 5.0
    // ... 39 features total
  }
}
```

响应体:
```json
{
  "svm": {
    "prediction": "Ransomware",
    "confidence": 0.94
  },
  "dbscan": {
    "is_anomaly": false,
    "cluster_id": 2,
    "distance_to_centroid": 0.34
  },
  "detection_id": "def456",
  "timestamp": "2026-06-10T14:31:00Z"
}
```

### 5.5 `/api/malware/predict/batch`

请求体:
```json
{
  "rows": [
    {"features": { ... 39 features ... }},
    {"features": { ... 39 features ... }}
  ]
}
```

响应体:
```json
{
  "results": [
    {
      "row_index": 1,
      "svm_prediction": "Benign",
      "dbscan": {
        "is_anomaly": false,
        "cluster_id": 0,
        "distance_to_centroid": 0.12
      }
    }
  ],
  "summary": {
    "anomaly_count": 3,
    "prediction_distribution": {
      "Benign": 12,
      "Ransomware": 4,
      "Spyware": 2,
      "Trojan": 1
    }
  },
  "timestamp": "2026-06-10T14:32:00Z"
}
```

### 5.6 `/api/stats`

响应体:
```json
{
  "total_detections": 124,
  "spam_detections": 78,
  "malware_detections": 46
}
```

### 5.7 `/api/explore/stats`

响应体示例:
```json
{
  "sms_spam": {
    "class_distribution": {"spam": 350, "ham": 400},
    "length_histogram": {"bins": [...], "counts": [...]},
    "keyword_heatmap": {
      "keywords": [...],
      "classes": [...],
      "values": [[...]]
    }
  },
  "enron_email": { ... },
  "malware": {
    "category_distribution": {"Trojan": 120, "Ransomware": 80, ...},
    "feature_variance": [...],
    "feature_correlation": {"features": [...], "values": [[...]]}
  }
}
```

### 5.8 `/api/dashboard/metrics`

响应体示例:
```json
{
  "performance_comparison": {
    "models": ["Random Forest", "Naive Bayes", "Logistic Regression", "SVM"],
    "accuracy": [...],
    "precision": [...],
    "recall": [...],
    "f1": [...]
  },
  "roc_curves": [
    {"model": "Random Forest", "fpr": [...], "tpr": [...]},
    ...
  ],
  "cv_results": [
    {"model": "Random Forest", "mean_f1": 0.92, "std_f1": 0.03},
    ...
  ],
  "confusion_matrices": {
    "Random Forest": {"labels": [...], "matrix": [[...]]},
    ...
  },
  "feature_importances": {
    "random_forest": [{"feature": "word_free", "importance": 0.12}, ...]
  },
  "probability_distribution": {
    "logistic_regression": {
      "spam": [...],
      "ham": [...]
    }
  },
  "clustering": {
    "kmeans": {"x": [...], "y": [...], "labels": [...]},
    "dbscan": {"x": [...], "y": [...], "labels": [...], "anomaly": [...]}
  }
}
```

### 5.9 `/api/history`

请求示例:
`GET /api/history?type=spam&start_date=2026-06-01&end_date=2026-06-30&query=free`

响应体:
```json
{
  "history": [
    {
      "id": 1,
      "timestamp": "2026-06-10T14:30:00Z",
      "detection_type": "spam",
      "input_preview": "Congratulations you have won...",
      "models_used": ["random_forest"],
      "verdict": "spam",
      "confidence": 0.97
    }
  ]
}
```

`DELETE /api/history` 响应:
```json
{ "success": true }
```

---

## 6. 数据模型设计

### 6.1 ORM 模型：DetectionRecord

字段设计：
- `id`: Integer, 主键
- `timestamp`: DateTime
- `detection_type`: String, `spam` 或 `malware`
- `input_preview`: String
- `models_used`: String 或 JSON 字符串
- `verdict`: String
- `confidence`: Float
- `raw_data`: JSON 字符串，可存储原始请求数据

### 6.2 前端数据结构

- Spam 结果对象:
  - `model`, `verdict`, `probability`, `top_features`
- Malware 结果对象:
  - `svm`, `dbscan`, `detection_id`, `timestamp`
- 探索数据:
  - 结构化数组 / 对象，适配 Plotly 图表
- 仪表盘数据:
  - 性能指标数组、ROC 曲线数组、混淆矩阵对象、聚类坐标

---

## 7. 数据流与处理流程

### 7.1 垃圾邮件检测流程

1. 用户输入文本或上传文件
2. 前端校验字段合法性
3. 发送 `POST /api/spam/predict`
4. 后端调用 `preprocessing.pipeline.transform_spam_text(text)`
5. 加载所选模型进行预测
6. 计算概率与 Top features
7. 将检测结果写入 SQLite 历史表
8. 返回 JSON
9. 前端渲染 Plotly 图表与结果卡片

### 7.2 恶意软件检测流程

单样本：
1. 用户填写特征或上传 CSV
2. 前端验证数值合法性
3. 发送 `POST /api/malware/predict`
4. 后端校验 39 个特征
5. 标准化后调用 SVM 与 DBSCAN
6. 生成雷达图比较数据
7. 保存历史记录并返回 JSON

批量：
1. 前端上传 CSV，进行列数与数值校验
2. 发送 `POST /api/malware/predict/batch`
3. 后端逐行预测并汇总分布
4. 返回结果列表与摘要

### 7.3 探索与仪表盘流程

- 后端从 `data/metrics/` 读取预计算文件
- `GET /api/explore/stats` 与 `GET /api/dashboard/metrics` 返回静态 JSON
- 前端基于数据渲染 Plotly 可视化

### 7.4 历史记录流程

- 前端调用 `GET /api/history` 带筛选参数
- 后端用 SQLAlchemy 查询 SQLite
- 响应历史列表
- 清除历史调用 `DELETE /api/history`

---

## 8. 输入验证与错误处理

### 8.1 前端验证

- Spam 文本：长度 10~10000
- Spam 文件：仅 `.txt`, `.eml`，最大 2MB
- 模型选择：至少选择一个
- Malware 表单：39 个字段均为有效数值且非空
- Malware CSV：恰好 39 列，且每列均为数值

### 8.2 后端验证

使用 FastAPI 与 Pydantic 自动校验：
- 缺少字段返回 422
- 特征数量不符返回 400
- 模型文件加载失败返回 503
- 内部错误返回 500

### 8.3 错误消息

统一返回结构:
```json
{ "error": "message" }
```

前端展示方式:
- MUI `Alert`
- 过滤提示、内联 helperText、对话框

---

## 9. 部署与运行说明

### 9.1 前端

- 运行 `npm install`
- 运行 `npm run dev`
- 生产构建 `npm run build`

### 9.2 后端

- 创建 Python 环境
- 安装依赖 `pip install -r requirements.txt`
- 运行 `uvicorn backend.main:app --reload`

### 9.3 运行时配置

- 后端需要访问训练好的模型 `.pkl`
- SQLite 文件存储于后端可写目录
- 后端允许跨域访问前端 URL

---

## 10. 测试策略

### 10.1 单元测试

- 前端验证函数
- 后端路由与服务函数
- 数据预处理函数
- 模型加载失败场景

### 10.2 集成测试

- `POST /api/spam/predict` 全链路
- `POST /api/malware/predict` 全链路
- `GET /api/history` 过滤逻辑
- `GET /api/dashboard/metrics` 数据格式

### 10.3 UI 测试

- 页面导航与路由
- 结果卡片渲染
- 图表是否存在
- 导出 CSV 是否生成

---

## 11. 文件结构建议

```
frontend/
  src/
    pages/
      Home.jsx
      SpamDetection.jsx
      MalwareDetection.jsx
      DataExploration.jsx
      ModelDashboard.jsx
      History.jsx
    components/
      NavBar.jsx
      SpamResultCard.jsx
      MalwareResultCard.jsx
      FileUploadZone.jsx
      ExportButton.jsx
      FeatureInputAccordion.jsx
      ModelSelectToggle.jsx
    api/
      axios.js
    utils/
      validation.js
      csv.js
      chartConfig.js
    App.jsx
    main.jsx
  package.json
  vite.config.js

backend/
  main.py
  routers/
    spam.py
    malware.py
    explore.py
    dashboard.py
    history.py
  models/
    loader.py
    spam_predictor.py
    malware_predictor.py
  preprocessing/
    pipeline.py
  database/
    db.py
    models.py
  services/
    history_service.py
  schemas/
    spam.py
    malware.py
    history.py
    explore.py
    dashboard.py
  data/
    metrics/
  requirements.txt

README.md
```

---

## 12. 设计文档总结

本设计文档明确了 NTCyber AI Web 应用的功能边界与模块分工，保证前端与后端对接规范清晰、模块可测试、数据流透明。后续实现时，可据此拆分开发任务，优先完成后端路由与模型加载，再完成页面 UI 和图表渲染。
