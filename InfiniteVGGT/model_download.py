from huggingface_hub import snapshot_download
import os

# 配置参数
repo_id = "lch01/StreamVGGT"  # Hugging Face仓库ID
target_dir = "/home/ghr/fs/Junyi/data/proj/model_weights"  # 替换为你要保存的目标路径
os.makedirs(target_dir, exist_ok=True)  # 确保目标路径存在

# 下载整个仓库
try:
    # 核心下载函数：download_all_files=True 下载所有文件
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,  # 不使用符号链接，直接下载文件
        resume_download=True,  # 支持断点续传
        ignore_patterns=None  # 如需忽略某些文件，可填如["*.git/*", "*.md"]
    )
    print(f"仓库已成功下载到：{target_dir}")
except Exception as e:
    print(f"下载失败：{str(e)}")