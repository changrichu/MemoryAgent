#!/bin/bash
# 一键启动 UI(在后台跑 API + Streamlit)
#
# 用法: bash scripts/run_with_ui.sh
# 端口:
#   8000 - FastAPI
#   8501 - Streamlit

set -e

# 1. 确保依赖服务在跑
echo "[1/4] 检查依赖服务..."
docker-compose up -d redis postgres milvus

# 2. 启动 API
echo "[2/4] 启动 FastAPI (8000)..."
nohup uvicorn ragagent.main:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
echo $! > logs/api.pid

# 3. 启动 Web UI
echo "[3/4] 启动 Streamlit (8501)..."
mkdir -p logs
nohup streamlit run src/ragagent/web_ui/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 > logs/ui.log 2>&1 &
echo $! > logs/ui.pid

# 4. 输出访问地址
echo "[4/4] 启动完成!"
echo ""
echo "==========================================="
echo "  🎉 MemoryAgent 已启动"
echo "==========================================="
echo "  💻 Web UI:    http://localhost:8501"
echo "  🔌 API:       http://localhost:8000"
echo "  📖 API Docs:  http://localhost:8000/docs"
echo "==========================================="
echo ""
echo "查看日志:"
echo "  tail -f logs/api.log"
echo "  tail -f logs/ui.log"
echo ""
echo "停止服务:"
echo "  bash scripts/stop_ui.sh"
