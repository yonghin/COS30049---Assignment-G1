# Module: Backend Model Loader

## 目标
实现模型加载模块，启动时加载所有训练好的 `.pkl` 文件并提供统一访问接口。

## 最小可执行任务
- [ ] 创建 `models/loader.py`
- [ ] 在应用启动时加载垃圾邮件和恶意软件模型文件
- [ ] 实现 `get_spam_model(name)`、`get_malware_svm()`、`get_malware_dbscan()` 接口
- [ ] 添加模型文件未找到或加载失败的异常处理
- [ ] 与 FastAPI 异常处理器对接，返回 503 状态
- [ ] 编写测试：模型加载成功、文件缺失、异常路径
