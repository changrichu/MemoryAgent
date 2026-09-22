# MemoryAgent 架构详解

> 本文档配合 README.md 阅读,深入讲解系统各层的设计与权衡。

## 一、为什么需要"三层记忆"?

LLM 自身有两大硬伤:

| 痛点 | 表现 | 解决方案 |
|------|------|---------|
| **上下文窗口有限** | 100K token 一满,前面的对话全丢 | 短/中/长期分层 |
| **无跨会话记忆** | 用户下次来,"我又从头开始聊" | L3 长期持久化 |

人类大脑的记忆机制是天然的分层模型:
```
   ┌─ 工作记忆(几秒-几分钟)              ← L1 Redis
   ├─ 短时记忆(几小时-几天)              ← L2 PostgreSQL
   └─ 长时记忆(永久,语义化)              ← L3 Milvus + KV
```

本系统完全借鉴这套模型。

---

## 二、三层记忆职责划分

### L1 - 工作记忆 (Working Memory)

**存储:** Redis List,key=`l1:chat:{user_id}`,value=JSON 消息数组

**用途:**
- 直接拼到 Prompt 里,作为"最近的对话上下文"
- LLM 看到的"立刻、马上"信息

**容量:**
- 默认保留 20 条
- TTL 24 小时,过期自动清理
- 单条 < 1 KB

**为什么这么设计:**
- Redis 读 0.1ms 级别,**毫秒级延迟**
- 频繁读写不会爆 IO
- LRU 滚动自动清理

### L2 - 情景记忆 (Episodic Memory)

**存储:** PostgreSQL + pgvector

**用途:**
- 会话级别的摘要 + 关键事实
- 解决"几十轮对话后,前面对话丢失"问题

**触发:**
- 当 L1 累积 ≥ 10 条 → 调用 LLM 摘要 → 写入 L2
- 保留 30 天,自动过期

**为什么这么设计:**
- PostgreSQL 同时支撑关系查询和向量检索(pgvector)
- 比纯向量库多了事务和索引能力
- 比纯 ES 多了向量能力

### L3 - 长期语义记忆 (Long-term Semantic Memory)

**存储:**
- Milvus 向量库(亿级扩展)
- Redis Hash(用户画像标签)

**用途:**
- 跨会话的事实、偏好、知识库
- 通过重要性评分决定写入

**写入机制:**
- LLM 自动抽取事实 → 入库
- 写入前去重(基于相似度)
- 版本号管理(同一事实可更新)

**为什么这么设计:**
- Milvus 工业级稳定,HNSW/IVF_FLAT 索引快
- KV 标签精确、低延迟(查 profile 秒返)

---

## 三、Agentic RAG 决策流

LangGraph 编排的 6 个节点:

```
                   [query]
                      │
                      ▼
                 ┌─ Router ─┐
                 │ 决策     │
                 └────┬────┘
            ┌─────────┴─────────┐
       ┌────┴────┐               │
   direct answer   retrieve      │
       │          │             │
       │       Rewriter          │
       │          │             │
       │       Retriever         │
       │          │             │
       │       Verifier ←┐      │
       │          │      │      │
       │       Generate │      │
       │          │   retry    │
       │       Memorize          │
       │          │             │
       └──────────┴─────────────┘
                  │
              [answer]
```

### 节点详解

| 节点 | 做什么 | 输入 | 输出 |
|------|--------|------|------|
| **Router** | 决策是否需要 RAG | query + history | decision JSON |
| **Rewriter** | 指代消解、改写 | query + L3 hints | rewritten_query |
| **Retriever** | 三层记忆召回 | rewritten_query | Top K 候选 |
| **Verifier** | Self-RAG 自检 | evidence | accept/retry |
| **Generate** | LLM 生成答案 | 所有上文 | final_answer |
| **Memorize** | 写回 L1/L2/L3 | user_msg + answer | 三层记忆更新 |

---

## 四、混合检索设计

### 召回阶段

```
Query ──┬──→ L1 (Redis 直读)         ← O(1), 无向量
         ├──→ L2 (PG 向量召回 Top 20) ← pgvector
         └──→ L3 (Milvus 向量召回 Top 20) ← COSINE
         └──→ (可选)Neo4j 图遍历       ← 多跳
                          │
                          ↓
                   merge + dedupe
                          │
                          ↓
                  BGE-Reranker-v2-M3   ← Cross-Encoder 精排
                          │
                          ↓
                  Top 5-10 进 Prompt
```

### 为什么用 Cross-Encoder 做精排?

| | Bi-Encoder (Embedding) | Cross-Encoder (Reranker) |
|---|---|---|
| 速度 | 极快(只算一次) | 慢(每对都算) |
| 精度 | 中(只看向量相似度) | 高(细看上下文) |
| 用法 | 粗召回 | 精排 |
| 模型 | BGE-M3 | BGE-Reranker-v2-M3 |

生产做法:**Bi-Encoder 粗召回 50-100 → Reranker 精排到 5-10 → Prompt**。

---

## 五、Self-RAG 自检机制

Verifier 节点判断检索质量,不合格时回到 Rewriter 重检,最多 3 次。

```
Retriever → Verifier
              │
              ├── acceptable → Generate
              └── unacceptable → Rewriter (重写)
                                  │
                                  ↓
                              Retriever 再来一次
```

**避免:**
- 答非所问
- 检索到一堆噪声
- LLM 幻觉

---

## 六、扩展方向

| 方向 | 价值 | 工作量 |
|------|------|--------|
| **GraphRAG** | 多跳推理、全局理解 | 2 周 |
| **多 Agent 协作** | 复杂业务场景 | 3 周 |
| **Web UI** | 用户可观察 | 1 周 |
| **多模态** | 处理 PDF/图片 | 2 周 |
| **联邦部署** | 数据隐私合规 | 1 月 |

详见 [Roadmap](../README.md#-roadmap)。
