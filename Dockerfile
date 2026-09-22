FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 代码
COPY src/ src/
COPY scripts/ scripts/
COPY examples/ examples/

# 工作目录
WORKDIR /app/src

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "ragagent.main:app", "--host", "0.0.0.0", "--port", "8000"]
