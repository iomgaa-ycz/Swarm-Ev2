FROM mlebench-env

ARG SUBMISSION_DIR=/home/submission
ARG LOGS_DIR=/home/logs
ARG CODE_DIR=/home/code
ARG AGENT_DIR=/home/agent

RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent
ARG REQUIREMENTS=${AGENT_DIR}/requirements.txt

COPY requirements_agent.txt ${AGENT_DIR}/requirements.txt

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

ENV HF_ENDPOINT=https://hf-mirror.com
ENV MODEL_SAVE_PATH=${AGENT_DIR}/embedding-models/bge-m3

COPY . ${AGENT_DIR}

# 将大体积模型文件移到 /home 之外，避免 entrypoint.sh 的 find+chmod 触发 overlay2 copy-on-write
RUN mv ${AGENT_DIR}/embedding-models /opt/embedding-models && \
    ln -s /opt/embedding-models ${AGENT_DIR}/embedding-models && \
    chmod -R 555 /opt/embedding-models
