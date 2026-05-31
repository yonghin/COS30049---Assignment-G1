# Module: Backend History Router

## 目标
实现后端历史记录接口，支持查询过滤和清除历史。

## 最小可执行任务
- [ ] 创建 `routers/history.py`
- [ ] 定义 `GET /api/history` 接口，支持 type、start_date、end_date、query 过滤
- [ ] 定义 `DELETE /api/history` 接口，删除所有历史记录
- [ ] 调用数据库服务或 `history_service` 查询和删除数据
- [ ] 添加异常处理和空结果返回
- [ ] 编写后端测试：过滤查询、清除历史、参数验证
