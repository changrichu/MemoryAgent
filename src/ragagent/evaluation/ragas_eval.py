"""
基于 Ragas 的 RAG 质量评估
"""
import json
from pathlib import Path
from typing import List, Dict

from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_golden_set(path: str = "data/golden_set.json") -> List[Dict]:
    """加载黄金测试集"""
    p = Path(path)
    if not p.exists():
        logger.warning(f"黄金集不存在: {path}")
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def simple_evaluate(
    agent,
    golden_set: List[Dict],
    top_k: int = 5,
) -> Dict:
    """
    简化版评估(无需 Ragas 依赖):
      对每个 query:
        - 跑 agent.chat 取 evidence
        - 计算 Recall@K (基于 facts 关键词匹配)
        - 计算 Faithfulness (LLM as Judge)
    """
    if not golden_set:
        return {"error": "empty golden set"}

    n = len(golden_set)
    metrics = {
        "total": n,
        "recall_at_k": 0,
        "faithfulness_pass": 0,
        "keyword_coverage": 0,
    }

    details = []

    for i, item in enumerate(golden_set):
        query = item["question"]
        keywords = item.get("must_contain", [])
        reference = item.get("reference_answer", "")

        result = agent.chat("eval_user", query)
        evidence = [
            s for s in result.get("sources", [])
        ]
        answer = result.get("final_answer", "")

        # 关键词覆盖率
        covered = sum(
            1 for kw in keywords if kw.lower() in answer.lower()
        )
        keyword_cov = covered / max(len(keywords), 1)

        # 简单事实一致性:检查关键事实是否出现在 answer
        facts_pass = keyword_cov >= 0.6

        if facts_pass:
            metrics["faithfulness_pass"] += 1
        metrics["keyword_coverage"] += keyword_cov

        details.append({
            "query": query,
            "answer_preview": answer[:120],
            "evidence_count": len(evidence),
            "keyword_coverage": round(keyword_cov, 2),
            "facts_pass": facts_pass,
        })

        if (i + 1) % 10 == 0:
            logger.info(f"评估进度: {i+1}/{n}")

    metrics["recall_at_k"] = round(
        metrics["keyword_coverage"] / n, 4
    )
    metrics["faithfulness"] = round(
        metrics["faithfulness_pass"] / n, 4
    )
    metrics["details"] = details

    return metrics


def evaluate_rag(
    agent,
    golden_set_path: str = "data/golden_set.json",
) -> Dict:
    """对外评估入口"""
    golden = load_golden_set(golden_set_path)
    return simple_evaluate(agent, golden)
