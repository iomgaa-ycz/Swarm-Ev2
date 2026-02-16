#!/bin/bash
cd /home/agent
export PYTHONPATH=$PYTHONPATH:/home/agent
eval "$(conda shell.bash hook)"
conda activate agent
mkdir -p /home/code /home/logs

# 代理状态日志（帮助调试）
if [ -n "$HTTP_PROXY" ]; then
    echo "[Proxy] HTTP_PROXY=$HTTP_PROXY" | tee -a /home/logs/proxy.log
    echo "[Proxy] HTTPS_PROXY=$HTTPS_PROXY" | tee -a /home/logs/proxy.log
    echo "[Proxy] NO_PROXY=$NO_PROXY" | tee -a /home/logs/proxy.log
else
    echo "[Proxy] 未配置（直连模式）" | tee -a /home/logs/proxy.log
fi

python /home/agent/run_mle_adapter.py
