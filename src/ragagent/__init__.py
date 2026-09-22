"""
MemoryAgent: 基于多级记忆机制的 Agentic RAG 对话系统

A production-ready hierarchical memory conversational system.
"""
__version__ = "0.1.0"
__author__ = "MemoryAgent Contributors"

from .config import settings
from .memory import L1WorkingMemory, L2EpisodicMemory, L3LongTermMemory
from .retriever import HybridRetriever, BGEReranker, GraphStore
from .agent import AgenticRAGAgent, Router, Rewriter
from .llm_client import LLMClient
from .fine_tuning import LoRAFineTuner, TrainingDataCollector
from .web_ui import build_app

__all__ = [
    "settings",
    "L1WorkingMemory",
    "L2EpisodicMemory",
    "L3LongTermMemory",
    "HybridRetriever",
    "BGEReranker",
    "GraphStore",
    "AgenticRAGAgent",
    "Router",
    "Rewriter",
    "LLMClient",
    "LoRAFineTuner",
    "TrainingDataCollector",
    "build_app",
]