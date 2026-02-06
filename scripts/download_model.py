"""BGE-M3 Embedding 模型下载脚本。

在 Docker 构建阶段预下载模型到容器内，避免运行时网络依赖。
"""

import os

from huggingface_hub import snapshot_download


def download_model() -> None:
    """从 HuggingFace 镜像站下载 BAAI/bge-m3 模型。"""
    model_name = "BAAI/bge-m3"
    local_dir = os.environ.get("MODEL_SAVE_PATH", "./embedding-models/bge-m3")

    print(f"正在下载模型 {model_name} 到 {local_dir}...")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.DS_Store"],
    )
    print("模型下载完成")


if __name__ == "__main__":
    download_model()
