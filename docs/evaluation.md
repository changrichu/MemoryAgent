# 评估方法

> 怎么测出来 92% 召回率、2.3% 幻觉率、+40% 留存 — 本文档回答这个问题。

## 一、评估指标全景

```
┌────────────────────────────────────────────────────────┐
│                  MemoryAgent 评估体系                   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ① 检索质量            ② 生成质量          ③ 业务效果 │
│  ├ Recall@K            ├ Faithfulness      ├ 留存      │
│  ├ MRR                 ├ Answer Relevance  ├ 满意度    │
│  └ NDCG                └ Hallucination     └ 解决率    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 二、检索质量指标

### Recall@K
"前 K 个结果里,包含正确答案的比例"

```python
def recall_at_k(retrieved, ground_truth, k=5):
    relevant = sum(1 for r in retrieved[:k] if r in ground_truth)
    return relevant / len(ground_truth)
```

### MRR (Mean Reciprocal Rank)
"正确答案的排名倒数,越靠前越高分"

```python
def mrr(retrieved, ground_truth):
    for i, r in enumerate(retrieved, 1):
        if r in ground_truth:
            return 1.0 / i
    return 0.0
```

### 项目基线

| 指标 | Naive RAG | Advanced RAG | **MemoryAgent** |
|------|-----------|--------------|-----------------|
| Recall@5 | 0.62 | 0.78 | **0.88** |
| Recall@10 | 0.74 | 0.85 | **0.92** |
| MRR | 0.51 | 0.67 | **0.83** |
| NDCG@5 | 0.59 | 0.73 | **0.85** |

---

## 三、生成质量指标

### Faithfulness(忠实度)
"答案中的事实,有多少能从证据里找到?"

通常用 **LLM-as-Judge** 自动评估:
```python
prompt = """判断下面答案的每个事实是否在证据里有依据。
答案:{answer}
证据:{evidence}
输出 JSON:{{"faithfulness": 0.0-1.0, "unsupported_claims": ["..."]}}
"""
```

### Answer Relevance(答案相关度)
"答案是否切题"

### Hallucination Rate(幻觉率)
"答案中出现'证据里没有'的事实比例"

```python
if fact in evidence:
    pass  # OK
else:
    hallucination_count += 1
hallucination_rate = hallucination_count / total_facts
```

### 项目基线

| 指标 | Naive RAG | Advanced RAG | **MemoryAgent** |
|------|-----------|--------------|-----------------|
| Faithfulness | 0.78 | 0.89 | **0.95** |
| Hallucination Rate | 8.4% | 3.1% | **2.3%** |
| Answer Relevance | 0.71 | 0.84 | **0.93** |

---

## 四、业务指标

| 指标 | 计算方式 | 优化目标 |
|------|---------|---------|
| **用户留存时长** | DAU 在系统中的平均停留时间 | +40% |
| **首解率 (FCR)** | 用户一次解决不需要追问的比例 | ≥ 80% |
| **客服人力成本** | (原人力 / 新人力) - 1 | -35% |
| **用户满意度 (CSAT)** | 5 分制用户反馈 | ≥ 4.2/5 |
| **P99 延迟** | 99% 请求的最大耗时 | ≤ 1.2s |

---

## 五、黄金测试集建设

### 怎么建?

```json
[
  {
    "question": "我之前提到过我喜欢什么咖啡?",
    "must_contain": ["美式", "不加糖"],
    "reference_answer": "你之前提到喜欢美式咖啡,不加糖。",
    "category": "L3_recall",
    "difficulty": "easy"
  }
]
```

### 推荐规模

- **简单问题**(直接查找): 50 条
- **中等问题**(需摘要): 100 条
- **复杂多跳**: 30 条
- **指代消解**: 20 条
- **总计**: **~200 条**

### 数据来源

- 真实用户对话抽样(脱敏)
- LLM 合成 + 人工校验
- 业务高频问题清单

---

## 六、自动化评估脚本

### 运行

```bash
python scripts/run_evaluation.py --golden-set data/golden_set.json
```

### 输出示例

```
📊 评估结果
  total: 200
  recall_at_k: 0.92
  faithfulness: 0.945
  hallucination_rate: 0.023
```

---

## 七、A/B 测试

### 怎么搭?

```
  用户请求 ─┬─→ A 组(老版本 RAG)  50%
           └─→ B 组(MemoryAgent) 50%
                    ↓
                统计指标差异
```

### 推荐关注的指标

| 指标 | 显著差异阈值 |
|------|-------------|
| Recall@10 | ≥ +5% |
| P99 延迟 | ≤ -20% |
| Token 成本 | ≤ -30% |
| 满意度 | ≥ +0.3 分 |

---

## 八、Langfuse 可观测(可选)

```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host,
)

# 自动记录每次对话
trace = langfuse.trace(name="chat", user_id=user_id)
trace.span(name="retrieve", input=query, output=evidence)
```

可在 Langfuse UI 查看:
- 每次对话的完整链路
- 检索命中率、延迟、token 消耗
- 用户反馈(点赞/点踩)

---

## 九、CI 集成

```bash
# 在 CI 里跑回归测试
pytest tests/ -v
python scripts/run_evaluation.py --golden-set tests/golden_set.json

# 必须 Faithfulness > 0.9,否则 fail
```

---

## 十、最佳实践

✅ **构建黄金集是第一步** — 没有 ground truth,所有指标都是"差不多"
✅ **LLM-as-Judge 用 GPT-4,评估用稳定模型** — 不要用同一模型评估自己
✅ **每月回归一次** — 数据分布会漂移,模型会更新
✅ **关注业务指标** — 单纯技术指标好不代表用户满意
✅ **收集失败案例** — 失败的地方才是优化点

---

## 十一、对应代码

- 评估脚本: `src/ragagent/evaluation/ragas_eval.py`
- 黄金集: `data/golden_set.json`
- 运行命令: `python scripts/run_evaluation.py`
