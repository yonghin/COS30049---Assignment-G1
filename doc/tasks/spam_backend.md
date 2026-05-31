# Module: Backend Spam Router

## 目标
实现后端垃圾邮件预测路由，暴露模型列表和预测接口。

## 最小可执行任务
- [ ] 创建 `routers/spam.py`
- [ ] 定义 `POST /api/spam/predict` 请求和响应 schema
- [ ] 定义 `GET /api/spam/models` 接口返回模型列表
- [ ] 将路由与预测服务对接，调用 `spam_predictor`
- [ ] 处理请求参数校验和异常返回
- [ ] 编写后端测试：预测成功、模型列表、缺少字段、错误处理
