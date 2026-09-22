.PHONY: help install dev test run docker clean format ui finetune eval stop

help:
	@echo "MemoryAgent - 常用命令"
	@echo ""
	@echo "🛠️  开发:"
	@echo "  make install     安装依赖"
	@echo "  make dev         启动 FastAPI 开发服务器"
	@echo "  make ui          启动 Web UI (Streamlit)"
	@echo "  make run         一键启动 API + UI + 依赖"
	@echo "  make stop        停止所有服务"
	@echo ""
	@echo "🧪 测试 & 评估:"
	@echo "  make test        跑单元测试"
	@echo "  make eval        跑 RAG 评估"
	@echo ""
	@echo "🚀 部署:"
	@echo "  make docker      启动所有 Docker 服务"
	@echo "  make docker-down 停止 Docker 服务"
	@echo ""
	@echo "🎯 微调:"
	@echo "  make finetune    启动 LoRA 微调(需 GPU)"
	@echo ""
	@echo "🔧 工具:"
	@echo "  make format      代码格式化"
	@echo "  make clean       清理临时文件"

install:
	pip install -r requirements.txt

dev:
	uvicorn ragagent.main:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run src/ragagent/web_ui/streamlit_app.py --server.port 8501

run:
	bash scripts/run_with_ui.sh

stop:
	bash scripts/stop_ui.sh

test:
	pytest tests/ -v --tb=short

eval:
	python scripts/run_evaluation.py --golden-set data/golden_set.json

finetune:
	python scripts/finetune_lora.py --base-model qwen2.5-7b --epochs 3

docker:
	docker-compose up -d

docker-logs:
	docker-compose logs -f

docker-down:
	docker-compose down

docker-clean:
	docker-compose down -v

format:
	black src/ tests/ examples/ scripts/
	ruff check --fix src/ tests/ examples/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache .ruff_cache

init-db:
	python scripts/init_db.py