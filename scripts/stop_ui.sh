#!/bin/bash
# 停止 UI 和 API 服务

if [ -f logs/api.pid ]; then
    kill $(cat logs/api.pid) 2>/dev/null && echo "✅ API 已停止"
    rm logs/api.pid
fi

if [ -f logs/ui.pid ]; then
    kill $(cat logs/ui.pid) 2>/dev/null && echo "✅ Web UI 已停止"
    rm logs/ui.pid
fi
