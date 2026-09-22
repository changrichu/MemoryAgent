"""
LoRA 微调 Hook:让用户数据微调专属模型
- TrainingDataCollector:收集训练数据(对话 + 检索 + 答案)
- LoRAFineTuner:用 LoRA 微调小模型,场景化适配
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from ..config import settings
from ..llm_client import get_llm_client
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingSample:
    """一条训练样本"""
    instruction: str          # 用户问题
    input: str = ""           # 上下文(检索到的记忆)
    output: str               # 标准答案
    score: float = 0.0        # 质量分(用户反馈/自动评估)
    metadata: Optional[Dict] = None
    created_at: str = ""


class TrainingDataCollector:
    """从线上日志收集高质量训练数据"""

    def __init__(self, data_dir: str = "data/fine_tuning"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.samples: List[TrainingSample] = []
        # 缓存数据按天切割
        self._date = datetime.now().strftime("%Y%m%d")

    def add_sample(self, sample: TrainingSample):
        """添加一条样本"""
        sample.created_at = sample.created_at or datetime.now().isoformat()
        self.samples.append(sample)

    def add_from_chat(self, query: str, evidence: List[Dict], answer: str,
                       user_rating: Optional[int] = None,
                       auto_score: float = 0.0):
        """从一次对话中构造样本"""
        # 把检索到的证据拼成 input
        input_text = "\n".join(
            f"[{e.get('source','?')}] {e.get('content','')[:300]}"
            for e in evidence
        )
        # 评分 = 用户评分 优先,否则自动评分
        score = user_rating / 5.0 if user_rating else auto_score

        sample = TrainingSample(
            instruction=query,
            input=input_text,
            output=answer,
            score=score,
            metadata={"evidence_count": len(evidence)},
        )
        self.add_sample(sample)

    def filter_quality(self, min_score: float = 0.7) -> List[TrainingSample]:
        """过滤高质量样本"""
        return [s for s in self.samples if s.score >= min_score]

    def save(self, filename: Optional[str] = None) -> str:
        """保存为 JSONL 格式(可直接用于 SFT)"""
        if filename is None:
            filename = f"train_{self._date}.jsonl"
        out_path = self.data_dir / filename

        # 默认过滤掉低分样本
        samples = self.filter_quality()

        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

        logger.info(f"保存训练数据: {len(samples)} 条 → {out_path}")
        return str(out_path)

    def load(self, filename: str) -> List[TrainingSample]:
        """加载已有数据"""
        path = self.data_dir / filename
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        return [TrainingSample(**d) for d in data]


class LoRAFineTuner:
    """LoRA 微调器(基于 HuggingFace transformers + PEFT)"""

    SUPPORTED_MODELS = {
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-7b": "deepseek-ai/DeepSeek-7B-Chat",
    }

    def __init__(
        self,
        base_model: str = "qwen2.5-7b",
        output_dir: str = "./models/lora",
        lora_rank: int = 8,
        lora_alpha: int = 32,
    ):
        self.base_model_name = self.SUPPORTED_MODELS.get(
            base_model, base_model
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

    def prepare_dataset(self, samples: List[TrainingSample]):
        """把样本转成 HuggingFace Dataset"""
        from datasets import Dataset

        data = []
        for s in samples:
            # Alpaca 风格的 prompt 模板
            prompt = (
                f"### 指令:\n{s.instruction}\n\n"
                f"### 输入:\n{s.input}\n\n"
                f"### 输出:\n{s.output}"
            )
            data.append({"text": prompt})

        logger.info(f"准备训练数据集:{len(data)} 条")
        return Dataset.from_list(data)

    def train(
        self,
        samples: List[TrainingSample],
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_length: int = 2048,
    ) -> str:
        """
        启动 LoRA 微调
        返回:输出目录路径
        """
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling,
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as e:
            logger.error(f"请安装依赖: pip install transformers peft datasets accelerate")
            raise

        # 1. 加载模型
        logger.info(f"加载基模型: {self.base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # 2. 配置 LoRA
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 3. 数据准备
        dataset = self.prepare_dataset(samples)

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize, batched=True)

        # 4. 训练配置
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            gradient_accumulation_steps=2,
            report_to="none",
        )

        # 5. 启动训练
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        logger.info("开始 LoRA 训练...")
        trainer.train()

        # 6. 保存适配器
        save_path = self.output_dir / "final"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"✅ LoRA 训练完成,保存到 {save_path}")
        return str(save_path)

    def merge_and_save(
        self,
        adapter_path: str,
        merged_path: Optional[str] = None,
    ) -> str:
        """把 LoRA 适配器合并回基模型,得到可独立部署的模型"""
        merged_path = merged_path or str(self.output_dir / "merged")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError:
            raise

        logger.info(f"合并 LoRA: {adapter_path} → {merged_path}")
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()

        Path(merged_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(merged_path)
        AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        ).save_pretrained(merged_path)

        logger.info(f"✅ 合并完成: {merged_path}")
        return merged_path


# ===========================================================
# 与主 Agent 的联动:钩子(Hook)
# ===========================================================
class FineTuneHook:
    """在主 Agent 的 memorize 节点处挂载,自动收集训练数据"""

    def __init__(self, collector: Optional[TrainingDataCollector] = None):
        self.collector = collector or TrainingDataCollector()
        self.llm = get_llm_client()

    def collect(
        self,
        query: str,
        evidence: List[Dict],
        answer: str,
        user_rating: Optional[int] = None,
    ):
        """从对话结果构造训练样本"""
        # 如果有用户评分 → 优先用
        if user_rating is not None:
            self.collector.add_from_chat(
                query, evidence, answer, user_rating=user_rating
            )
            return

        # 否则用 LLM 评估
        eval_prompt = f"""评估下面答案的质量(0-1):
问题:{query}
证据:{evidence[:3]}
答案:{answer}

输出 JSON:{{"score": 0.0-1.0, "reason": "..."}}
"""
        try:
            result = self.llm.chat_json(
                [{"role": "user", "content": eval_prompt}],
                temperature=0.0,
            )
            score = float(result.get("score", 0.5))
        except Exception:
            score = 0.5

        if score >= 0.7:  # 只保留高质量
            self.collector.add_from_chat(
                query, evidence, answer, auto_score=score
            )

    def flush(self, min_samples: int = 100):
        """保存累计的训练数据"""
        if len(self.collector.samples) >= min_samples:
            return self.collector.save()
        return None
