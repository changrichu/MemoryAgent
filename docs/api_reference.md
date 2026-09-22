# API 参考

## 1. POST /chat

主对话接口。完成一次完整的 Agentic RAG 流程。

### 请求

```http
POST /chat
Content-Type: application/json

{
  "user_id": "user_123",
  "query": "我之前提到过喜欢什么咖啡?"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | string | 是 | 用户唯一标识 |
| query | string | 是 | 用户提问 |

### 响应

```json
{
  "user_id": "user_123",
  "query": "我之前提到过喜欢什么咖啡?",
  "answer": "你之前提到喜欢美式咖啡,不加糖。",
  "sources": [
    "L2: 用户 2026-09-15 提到的咖啡偏好",
    "L3: 用户偏好 - 美式不加糖"
  ],
  "routing": {
    "decision": "retrieve_long",
    "reason": "含实体'咖啡',需要长期记忆"
  },
  "rewritten_query": "用户曾经喜欢的咖啡风格与口味偏好",
  "logs": [
    "📍 Router: retrieve_long",
    "📍 Rewriter: 用户曾经喜欢的咖啡风格...",
    "📍 Retriever: 召回到 5 条",
    "📍 Verifier: 通过 (faith=0.92)",
    "📍 Generate: 完成生成",
    "📍 Memorize: 已写入三层记忆"
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| answer | string | 最终答案 |
| sources | array | 引用的证据来源 |
| routing | object | 路由决策详情 |
| rewritten_query | string | 改写后的查询 |
| logs | array | 完整执行轨迹(可调试用) |

---

## 2. GET /health

检查服务健康状况。

```http
GET /health
```

```json
{
  "status": "ok",
  "components": {
    "redis": "ok",
    "postgres": "ok",
    "milvus": "ok (128 entities)"
  }
}
```

---

## 3. POST /memory/clear/{user_id}

清空指定用户的 L1 工作记忆。

```http
POST /memory/clear/user_123
```

```json
{ "ok": true, "user_id": "user_123", "cleared": "L1" }
```

---

## 4. GET /memory/profile/{user_id}

获取用户画像。

```http
GET /memory/profile/user_123
```

```json
{
  "user_id": "user_123",
  "profile": {
    "profession": "产品经理",
    "city": "杭州",
    "drink_preference": "美式不加糖"
  }
}
```

---

## 5. Python SDK 调用(推荐用于集成)

```python
from ragagent import AgenticRAGAgent

agent = AgenticRAGAgent()

result = agent.chat(
    user_id="user_123",
    query="推荐几本产品经理必读书",
)

print(result["final_answer"])
print(result["sources"])
```

### Streaming(规划中)

未来会支持 LangChain 的 `agent.stream()` 输出 token-by-token:
```python
for token in agent.stream("user_123", "..."):
    print(token, end="", flush=True)
```

---

## 6. 错误码

| 状态码 | 含义 |
|--------|------|
| 200 | 成功 |
| 422 | 请求参数错误 |
| 500 | 服务异常(详见返回的 detail) |

---

## 7. 限流

目前无强制限流,生产建议在 API 网关层加入:
- 单用户 QPS ≤ 5
- 单 IP QPS ≤ 20

---

## 8. 完整代码示例

参考 [examples/basic_chat.py](../examples/basic_chat.py)。
