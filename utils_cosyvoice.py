import os
import torch
import gc
import sys
import folder_paths
from huggingface_hub import snapshot_download as hf_snapshot_download

# --- 尝试导入 ModelScope ---
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    HAS_MODELSCOPE = True
except ImportError:
    HAS_MODELSCOPE = False

# --- 尝试导入 CosyVoice ---
try:
    from cosyvoice.cli.cosyvoice import AutoModel
    HAS_COSYVOICE = True
except ImportError:
    HAS_COSYVOICE = False

# ================= 路径配置 =================

COSY_MODELS_DIR = os.path.join(folder_paths.models_dir, "CosyVoice")
if not os.path.exists(COSY_MODELS_DIR):
    os.makedirs(COSY_MODELS_DIR)

# 全局缓存
LOADED_COSY_MODELS = {}

def _download_model_logic(repo_id, local_dir, source="ModelScope"):
    """下载逻辑"""
    if source == "ModelScope":
        if not HAS_MODELSCOPE:
            raise ImportError("请先安装 modelscope: pip install modelscope")
        print(f"\n[CosyVoice] Downloading from ModelScope: {repo_id} -> {local_dir}")
        ms_snapshot_download(model_id=repo_id, local_dir=local_dir)
    else: # HuggingFace / HF Mirror
        print(f"\n[CosyVoice] Downloading from HF: {repo_id} -> {local_dir}")
        if source == "HF Mirror":
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        elif "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        
        hf_snapshot_download(repo_id=repo_id, local_dir=local_dir, resume_download=True, max_workers=4)

def load_cosyvoice_model(model_name, device, auto_download=False, source="ModelScope"):
    """
    加载 CosyVoice 模型
    """
    if not HAS_COSYVOICE:
        raise ImportError("Critical Dependency Missing: Please run 'pip install cosyvoice hyperpyyaml'")

    # 1. 确定模型路径
    # ModelScope ID: FunAudioLLM/Fun-CosyVoice3-0.5B-2512
    repo_id = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    target_folder_name = "Fun-CosyVoice3-0.5B-2512"
    
    model_path = os.path.join(COSY_MODELS_DIR, target_folder_name)
    
    # 下载模型
    if not os.path.exists(os.path.join(model_path, "cosyvoice.yaml")):
        if auto_download:
            try:
                _download_model_logic(repo_id, model_path, source)
            except Exception as e:
                raise Exception(f"CosyVoice Download failed: {e}")
        else:
            raise FileNotFoundError(f"CosyVoice Model not found at {model_path}")

    # 2. 加载模型
    global LOADED_COSY_MODELS
    if model_path not in LOADED_COSY_MODELS:
        print(f"[CosyVoice] Loading model from {model_path}...")
        torch.cuda.empty_cache()
        
        try:
            # CosyVoice AutoModel 会自动处理 device，但通常需要我们确保环境正确
            # 注意：CosyVoice3 这里的 device 处理可能封装在内部，我们主要负责路径
            model = AutoModel(model_dir=model_path)
            
            LOADED_COSY_MODELS[model_path] = model
            print("[CosyVoice] Model loaded successfully.")
        except Exception as e:
            raise Exception(f"Failed to load CosyVoice model: {str(e)}")

    return LOADED_COSY_MODELS[model_path]

def unload_cosyvoice_model():
    """强制卸载模型"""
    global LOADED_COSY_MODELS
    if LOADED_COSY_MODELS:
        print(f"[CosyVoice] Unloading models...")
        LOADED_COSY_MODELS.clear()
        gc.collect()
        torch.cuda.empty_cache()