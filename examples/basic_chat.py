"""
基础对话示例 - 演示三层记忆与 Agentic RAG
运行: python examples/basic_chat.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragagent import AgenticRAGAgent


def main():
    print("=" * 60)
    print("MemoryAgent 基础对话 Demo")
    print("=" * 60)

    agent = AgenticRAGAgent()
    user_id = "demo_user_001"

    # 模拟多轮对话,展示记忆能力
    conversations = [
        "你好,我是小明,在做 AI 产品经理",
        "我之前提到过我是做什么工作的?",
        "能推荐几本适合产品经理读的 AI 书吗?",
        "我之前问过你推荐的书,你还记得吗?",
        "再见了",
    ]

    for q in conversations:
        print(f"\n👤 User: {q}")
        result = agent.chat(user_id=user_id, query=q)
        print(f"🤖 Agent: {result['final_answer']}")
        print(f"   [routing={result['routing'].get('decision')}]")
        if result.get("rewritten", {}).get("rewritten_query"):
            rew = result["rewritten"]["rewritten_query"]
            if rew != q:
                print(f"   [rewritten: {rew}]")

    print("\n" + "=" * 60)
    print("✅ Demo 结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
