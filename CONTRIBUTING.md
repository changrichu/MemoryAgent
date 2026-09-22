# 贡献指南

欢迎 PR! 🎉

## 开发流程

1. Fork 本仓库
2. 创建分支:`git checkout -b feat/your-feature`
3. 提交代码并写测试
4. 跑 `make test` 全部通过
5. 跑 `make format` 格式化
6. 提 PR

## Commit 规范

```
feat: 新功能
fix:  Bug 修复
docs:  仅文档
style:  代码格式(无逻辑改动)
refactor: 重构
test:  测试
chore: 杂项
```

## 代码风格

- Python 3.10+
- Black 格式化(`line-length=100`)
- PEP 8
- 函数必须有 docstring
- 复杂逻辑必须有 inline 注释

## 测试

```bash
# 跑全部测试
make test

# 单个测试
pytest tests/test_memory.py::TestL1WorkingMemory::test_add_and_get -v

# 覆盖率
pytest --cov=src --cov-report=html
```

## 报告 Bug

用 GitHub Issues,提供:
- 环境(OS / Python 版本 / 包版本)
- 复现步骤
- 期望行为 vs 实际行为
- 错误日志/截图

## 提新功能建议

先开 Issue 讨论设计,达成共识再写代码。
