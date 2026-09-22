"""
GraphRAG 模块:知识图谱 + 多跳推理
- LLM 抽取实体和关系
- Neo4j 存储图谱
- 支持多跳子图检索
"""
from typing import List, Dict, Optional
import json
import re

from ..config import settings
from ..llm_client import get_llm_client
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GraphStore:
    """基于 Neo4j 的知识图谱存储与检索"""

    def __init__(self, driver=None):
        self.driver = driver
        self.llm = get_llm_client()
        if self.driver is None and settings.neo4j_uri:
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password),
                )
                logger.info("Neo4j 连接成功")
            except Exception as e:
                logger.warning(f"Neo4j 未连接,GraphRAG 不可用: {e}")
                self.driver = None

    # ==========================================
    # 实体抽取(LLM)
    # ==========================================
    def extract_entities_relations(self, text: str) -> Dict:
        """用 LLM 从文本里抽实体和关系"""
        prompt = f"""从以下文本中抽取实体和它们之间的关系。
严格输出 JSON:

{{
  "entities": [
    {{"name": "实体名", "type": "Person/Org/Concept/Location/...", "desc": "简述"}}
  ],
  "relations": [
    {{"source": "实体A", "target": "实体B", "relation": "关系描述"}}
  ]
}}

文本:{text[:2000]}
"""
        result = self.llm.chat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return {
            "entities": result.get("entities", []),
            "relations": result.get("relations", []),
        }

    # ==========================================
    # 写入 Neo4j
    # ==========================================
    def add_document(
        self,
        user_id: str,
        text: str,
        doc_id: Optional[str] = None,
    ) -> Dict:
        """处理一篇文章:抽取 → 入图"""
        if self.driver is None:
            return {"error": "Neo4j not connected"}

        extraction = self.extract_entities_relations(text)
        entities = extraction["entities"]
        relations = extraction["relations"]

        if not entities:
            return {"entities_added": 0, "relations_added": 0}

        with self.driver.session() as session:
            # 创建文档节点
            if doc_id:
                session.run(
                    "MERGE (d:Document {id: $doc_id})",
                    doc_id=doc_id,
                )

            # 创建实体节点
            for ent in entities:
                session.run("""
                    MERGE (e:Entity {name: $name, user_id: $user_id})
                    SET e.type = $type, e.desc = $desc
                    RETURN e
                """,
                    name=ent.get("name", "")[:100],
                    user_id=user_id,
                    type=ent.get("type", "Unknown"),
                    desc=ent.get("desc", "")[:200],
                )

            # 创建关系
            for rel in relations:
                session.run("""
                    MATCH (a:Entity {name: $source, user_id: $user_id})
                    MATCH (b:Entity {name: $target, user_id: $user_id})
                    MERGE (a)-[r:RELATES {type: $relation}]->(b)
                    RETURN r
                """,
                    source=rel.get("source", "")[:100],
                    target=rel.get("target", "")[:100],
                    relation=rel.get("relation", "")[:200],
                    user_id=user_id,
                )

        logger.info(f"Graph 入库: {len(entities)} 实体, {len(relations)} 关系")
        return {
            "entities_added": len(entities),
            "relations_added": len(relations),
        }

    # ==========================================
    # 图遍历检索(多跳)
    # ==========================================
    def traverse_subgraph(
        self,
        user_id: str,
        entity_names: List[str],
        max_hops: int = 2,
    ) -> List[Dict]:
        """从给定实体出发,N 跳扩展子图"""
        if self.driver is None or not entity_names:
            return []

        facts = []
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH path = (start:Entity)-[*1..{max_hops}]-(connected:Entity)
                WHERE start.user_id = $user_id
                  AND start.name IN $names
                UNWIND relationships(path) AS rel
                RETURN DISTINCT
                    startNode(rel).name AS source,
                    rel.type AS relation,
                    endNode(rel).name AS target
                LIMIT 100
            """, user_id=user_id, names=entity_names)

            for record in result:
                facts.append({
                    "content": (
                        f"{record['source']} {record['relation']} {record['target']}"
                    ),
                    "source": "Graph",
                    "score": 0.8,
                    "metadata": {
                        "source_name": record["source"],
                        "relation": record["relation"],
                        "target_name": record["target"],
                    },
                })

        logger.info(f"图遍历:从 {len(entity_names)} 实体召回 {len(facts)} 条事实")
        return facts

    # ==========================================
    # 与混合检索器联动
    # ==========================================
    def retrieve(
        self,
        query: str,
        user_id: str,
        max_hops: int = 2,
        top_entities: int = 5,
    ) -> List[Dict]:
        """
        1. 用 LLM 从 query 里识别可能涉及的实体
        2. 在图中多跳扩展
        3. 返回子图事实
        """
        if self.driver is None:
            return []

        # 1. 抽取 query 中的实体
        prompt = f"""从以下问题里识别可能涉及的关键实体名(人名/公司名/产品名/技术名词等)。
严格输出 JSON:{{"entities": ["实体1", "实体2", ...]}}

问题:{query}
"""
        try:
            result = self.llm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            entities = result.get("entities", [])[:top_entities]
        except Exception as e:
            logger.warning(f"实体抽取失败: {e}")
            entities = []

        if not entities:
            return []

        # 2. 多跳检索
        return self.traverse_subgraph(user_id, entities, max_hops)

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 连接关闭")
