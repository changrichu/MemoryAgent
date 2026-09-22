"""
带多层记忆的对话 - 直接调用三层记忆模块
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragagent import (
    L1WorkingMemory,
    L2EpisodicMemory,
    L3LongTermMemory,
    get_llm_client,
)


def main():
    print("=" * 60)
    print("三层记忆 Direct Demo")
    print("=" * 60)
    llm = get_llm_client()
    l1 = L1WorkingMemory()
    l2 = L2EpisodicMemory()
    l3 = L3LongTermMemory()

    user_id = "demo_user_002"

    # --- L1 写入 ---
    print("\n[1] 写入 L1 工作记忆...")
    l1.add(user_id, "user", "我喜欢喝美式咖啡,不要糖")
    l1.add(user_id, "assistant", "好的,已记录:美式不加糖")
    msgs = l1.get_recent(user_id)
    print(f"   当前 L1 有 {len(msgs)} 条消息")

    # --- L2 摘要 ---
    print("\n[2] 写入 L2 情景摘要...")
    l2.add_summary(
        user_id=user_id,
        summary="用户表达咖啡偏好,倾向于无糖美式",
        facts=["咖啡偏好: 美式不加糖"],
    )

    # --- L3 长期事实 ---
    print("\n[3] 写入 L3 长期记忆 + 画像...")
    l3.add_fact(
        user_id=user_id,
        content="用户是一名 AI 产品经理,常住在杭州",
        tags=["profile", "profession", "location"],
    )
    l3.add_fact(
        user_id=user_id,
        content="用户偏好简洁直接的沟通风格,反感空话",
        tags=["preference", "communication"],
    )
    l3.update_profile_field(user_id, "profession", "AI 产品经理")
    l3.update_profile_field(user_id, "city", "杭州")
    l3.update_profile_field(user_id, "drink_preference", "美式不加糖")

    # --- 三层记忆联合检索 ---
    print("\n[4] 三层记忆联合检索 query='我喝什么咖啡?'")
    l1_msgs = l1.get_recent(user_id)
    l2_hits = l2.recall(user_id, "咖啡偏好", top_k=3, days=7)
    l3_hits = l3.recall(user_id, "咖啡", top_k=5)
    profile = l3.get_profile(user_id)

    print("   L1:", [m["content"][:50] for m in l1_msgs])
    print("   L2:", [(h["score"], h["summary"][:60]) for h in l2_hits])
    print("   L3:", [(h["score"], h["content"][:60]) for h in l3_hits])
    print("   Profile:", profile)

    print("\n" + "=" * 60)
    print("✅ Demo 结束")


if __name__ == "__main__":
    main()
