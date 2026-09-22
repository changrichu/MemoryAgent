"""
评估示例 - 演示如何用 Ragas + LLM-as-Judge 评估
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    print("=" * 60)
    print("MemoryAgent 评估示例")
    print("=" * 60)

    from ragagent import AgenticRAGAgent
    from ragagent.evaluation import evaluate_rag, load_golden_set

    # 加载黄金集
    golden_path = Path(__file__).parent.parent / "data" / "golden_set.json"
    if not golden_path.exists():
        print(f"❌ 黄金集不存在: {golden_path}")
        return

    golden = load_golden_set(str(golden_path))
    print(f"\n✅ 加载 {len(golden)} 条黄金测试集")

    # 跑评估
    print("\n[1] 初始化 Agent...")
    agent = AgenticRAGAgent()

    print("\n[2] 跑评估...")
    metrics = evaluate_rag(agent, str(golden_path))

    print("\n" + "=" * 60)
    print("📊 评估结果")
    print("=" * 60)
    if "error" not in metrics:
        print(f"  总样本数: {metrics.get('total')}")
        print(f"  Recall@K (关键词覆盖): {metrics.get('recall_at_k')}")
        print(f"  Faithfulness (忠实度): {metrics.get('faithfulness')}")
        print()
        print("  详情:")
        for d in metrics.get("details", [])[:10]:
            status = "✅" if d["facts_pass"] else "❌"
            print(f"  {status} Q: {d['query'][:40]}")
            print(f"      A: {d['answer_preview']}")
            print(f"      keyword_cov={d['keyword_coverage']:.2f}")
    else:
        print(f"❌ 评估失败: {metrics['error']}")

    print("\n✅ Demo 结束")


if __name__ == "__main__":
    main()
