# Module: Backend Spam Predictor

## 目标
实现垃圾邮件预测服务，负责文本预处理、模型推理和 Top features 计算。

## 最小可执行任务
- [ ] 创建 `models/spam_predictor.py`
- [ ] 接收原始文本并调用预处理管道生成特征向量
- [ ] 调用选中的 spam 模型进行预测与概率计算
- [ ] 计算每个模型的 Top features（词语与重要性）
- [ ] 返回统一结果结构供路由层调用
- [ ] 添加异常处理和输入校验
- [ ] 编写测试：输入转化、预测结果结构、Top features 计算
