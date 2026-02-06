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
RUN mkdir -p ${MODEL_SAVE_PATH}
COPY scripts/download_model.py ${AGENT_DIR}/scripts/download_model.py
RUN conda run -n ${CONDA_ENV_NAME} python ${AGENT_DIR}/scripts/download_model.py && \
    chmod -R 555 ${MODEL_SAVE_PATH}

COPY . ${AGENT_DIR}
