"""
数据库初始化脚本
- 创建 L2 (PostgreSQL) 表结构
- 创建 L3 (Milvus) collection
- (可选)导入示例文档与初始数据
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragagent import L2EpisodicMemory, L3LongTermMemory
from ragagent.utils.logger import get_logger

logger = get_logger(__name__)


def init():
    print("[init] 初始化 L2 (PostgreSQL)...")
    l2 = L2EpisodicMemory()
    print("   ✅ L2 表结构就绪")

    print("[init] 初始化 L3 (Milvus)...")
    l3 = L3LongTermMemory()
    print(f"   ✅ L3 collection 就绪,当前实体数: {l3.coll.num_entities}")

    print("\n[init] 初始化示例数据...")
    # 这里可以放系统级的事实(对所有用户都有效)
    sample_facts = [
        "系统版本:MemoryAgent v0.1.0",
        "支持三层记忆架构:L1 Redis / L2 PostgreSQL / L3 Milvus",
        "支持 Re-ranker:BAAI/bge-reranker-v2-m3",
    ]
    print(f"   示例事实 {len(sample_facts)} 条(未写入,需要时调用 l3.add_fact())")

    print("\n✅ 全部初始化完成")


if __name__ == "__main__":
    init()
