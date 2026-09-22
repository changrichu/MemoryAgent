"""
跑评估 - 基于黄金测试集评估 RAG 质量
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden-set",
        default="data/golden_set.json",
        help="黄金测试集路径",
    )
    parser.add_argument(
        "--user-id",
        default="eval_user",
        help="评估时的虚拟用户 ID",
    )
    args = parser.parse_args()

    from ragagent import AgenticRAGAgent
    from ragagent.evaluation import evaluate_rag

    golden_path = Path(args.golden_set)
    if not golden_path.exists():
        print(f"❌ 黄金集不存在: {golden_path}")
        print("请先准备测试集(JSON 格式)")
        return

    print(f"[eval] 加载黄金集: {golden_path}")
    print(f"[eval] 初始化 Agent...")
    agent = AgenticRAGAgent()

    print(f"[eval] 开始评估...")
    metrics = evaluate_rag(agent, str(golden_path))

    print("\n" + "=" * 60)
    print("📊 评估结果")
    print("=" * 60)
    for k, v in metrics.items():
        if k != "details":
            print(f"  {k}: {v}")
    print("=" * 60)

    # 保存结果
    out_path = Path("data/eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
