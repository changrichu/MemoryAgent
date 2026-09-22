# 部署指南

## 一、本地开发(快速验证)

```bash
# 1. 启动依赖服务
docker-compose up -d redis postgres milvus

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API KEY

# 3. 安装依赖
pip install -r requirements.txt

# 4. 初始化数据库
python scripts/init_db.py

# 5. 启动 API
make dev
# 或:uvicorn ragagent.main:app --reload --port 8000

# 6. 测试
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","query":"你好"}'
```

---

## 二、生产部署(Docker Compose)

```bash
# 启动所有服务(包括 app)
docker-compose up -d

# 看日志
docker-compose logs -f memoryagent-app

# 扩缩容
docker-compose up -d --scale memoryagent-app=3
```

---

## 三、K8s 部署(高可用)

详见 [deploy/k8s/](deploy/k8s/) (规划中)

---

## 四、性能调优

### 4.1 LLM 推理优化

| 优化 | 收益 | 实施成本 |
|------|------|---------|
| 语义缓存 | Token -65% | 低 |
| 提示词压缩 | Token -30% | 低 |
| 小模型分级 | 成本 -50% | 中 |
| vLLM 自部署 | 吞吐 × 5 | 高 |
| 流式输出 | UX 显著提升 | 低 |

### 4.2 向量库优化

- 调整 `nprobe`(默认 16): 越大越准越慢
- 调整 `nlist`(默认 128): 越大越准但建索引越慢
- 用 IVF_PQ 压缩存储(亿级规模)

### 4.3 数据库优化

- PostgreSQL: `pgvector` 索引 + `HNSW` 比 `IVFFlat` 快
- Redis: 启用 Pipelining、Lua 脚本

---

## 五、监控告警

### 关键指标

| 指标 | 告警阈值 |
|------|---------|
| P99 延迟 | > 2s |
| 错误率 | > 1% |
| Redis 内存 | > 80% |
| Milvus 慢查询 | > 100ms |
| LLM 失败率 | > 5% |

### 推荐工具

- **Langfuse**(LLM 调用链路)
- **Prometheus + Grafana**(基础设施)
- **Sentry**(异常捕获)
- **ELK**(日志聚合)

---

## 六、备份策略

| 数据 | 备份周期 | 方式 |
|------|---------|------|
| Redis | - | AOF(已配置) |
| PostgreSQL | 每日 | pg_dump |
| Milvus | 每周 | milvus-backup |
| 用户数据 | 实时 | 主从复制 |

---

## 七、安全合规

- **API Key**:不要提交到 git,使用 `.env` 并加入 `.gitignore`
- **数据传输**: HTTPS(部署时由网关负责)
- **敏感字段**: 加密存储(E2E)
- **审计日志**: 所有 LLM 调用记录 30 天
- **GDPR**: 提供 `/memory/clear/{user_id}` 一键清除接口
