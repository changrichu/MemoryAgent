"""
启动 LoRA 微调的脚本
用法: python scripts/finetune_lora.py --base-model qwen2.5-7b --epochs 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragagent.fine_tuning import LoRAFineTuner, TrainingDataCollector
from ragagent.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="qwen2.5-7b")
    parser.add_argument("--data-dir", default="data/fine_tuning")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print(f"🎯 LoRA 微调:{args.base_model}")
    print("=" * 60)

    # 1. 加载训练数据
    collector = TrainingDataCollector(data_dir=args.data_dir)
    data_path = Path(args.data_dir) / f"train_*.jsonl"
    data_files = list(Path(args.data_dir).glob("train_*.jsonl"))

    if not data_files:
        print(f"❌ 未找到训练数据:{data_path}")
        print("请先在 Web UI 上对话 + 反馈,数据会自动收集")
        return

    samples = []
    for f in data_files:
        samples.extend(collector.load(f.name))

    print(f"📊 加载训练样本:{len(samples)} 条")
    if not samples:
        print("❌ 训练数据为空")
        return

    # 2. 训练
    tuner = LoRAFineTuner(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
    )

    adapter_path = tuner.train(
        samples=samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # 3. 合并可部署模型
    merged = tuner.merge_and_save(adapter_path)
    print(f"\n✅ 训练完成!")
    print(f"   适配器: {adapter_path}")
    print(f"   合并模型: {merged}")
    print(f"\n下一步:")
    print(f"   vllm serve {merged} --port 8001")


if __name__ == "__main__":
    main()
