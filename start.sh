#!/bin/bash
cd /home/agent
export PYTHONPATH=$PYTHONPATH:/home/agent
eval "$(conda shell.bash hook)"
conda activate agent
mkdir -p /home/code /home/logs
python /home/agent/run_mle_adapter.py
