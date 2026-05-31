# Module: Backend Main Application

## 目标
实现 FastAPI 应用入口，注册路由、加载模型、配置 CORS 和异常处理。

## 最小可执行任务
- [ ] 创建 `backend/main.py`
- [ ] 初始化 FastAPI 应用并配置 CORS
- [ ] 注册所有子路由：spam、malware、explore、dashboard、history
- [ ] 在启动事件中初始化模型加载器和 SQLite 数据库
- [ ] 添加全局异常处理器，统一返回 JSON 错误
- [ ] 编写集成测试：应用启动、路由注册和全局异常响应
