# Module: Backend Database

## 目标
实现 SQLite 持久化层，保存并查询历史检测记录。

## 最小可执行任务
- [ ] 创建 `database/db.py`，建立 SQLAlchemy engine、SessionFactory 和 `get_db()` 依赖
- [ ] 创建 `database/models.py`，定义 `DetectionRecord` ORM 模型
- [ ] 实现 `models_used` 和 `raw_data` 字段的序列化存储
- [ ] 实现历史记录写入、查询和删除辅助函数
- [ ] 编写测试：ORM 映射、插入查询、删除历史
