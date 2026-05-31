# Module: Backend Dashboard Router

## 目标
实现后端模型仪表盘接口，返回性能指标和聚类可视化数据。

## 最小可执行任务
- [ ] 创建 `routers/dashboard.py`
- [ ] 定义 `GET /api/dashboard/metrics` 接口
- [ ] 读取 `data/metrics/` 中的预计算仪表盘数据
- [ ] 返回性能比较、ROC 曲线、CV 结果、混淆矩阵、特征重要性和聚类数据
- [ ] 添加错误处理和文件读取失败处理
- [ ] 编写后端测试：接口响应格式、异常处理
