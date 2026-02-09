import os
import sys
import torch
import gc
import folder_paths
from huggingface_hub import snapshot_download as hf_snapshot_download

# ================= 关键修改：添加本地路径 =================
# 获取当前文件所在目录
current_node_path = os.path.dirname(os.path.abspath(__file__))

# 将当前插件目录加入系统路径，这样 Python 就能找到本地的 'cosyvoice' 文件夹
if current_node_path not in sys.path:
    sys.path.append(current_node_path)

# 再次确认子目录是否需要加入 (有些环境比较严格)
local_cosy_path = os.path.join(current_node_path, "cosyvoice")
if os.path.exists(local_cosy_path) and current_node_path not in sys.path:
    sys.path.append(current_node_path)
# =======================================================

# --- 尝试导入 ModelScope ---
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    HAS_MODELSCOPE = True
except ImportError:
    HAS_MODELSCOPE = False

# --- 尝试导入 CosyVoice ---
HAS_COSYVOICE = False
try:
    # 优先尝试导入本地或系统安装的 cosyvoice
    # 官方代码结构通常是 cosyvoice.cli.cosyvoice
    from cosyvoice.cli.cosyvoice import CosyVoice
    HAS_COSYVOICE = True
    print(f"[Fun-CosyVoice3] 成功加载 CosyVoice 库")
except ImportError as e:
    print(f"[Fun-CosyVoice3] 导入失败: {e}")
    # 尝试另一种常见的导入路径 (兼容不同版本的源码结构)
    try:
        sys.path.append(os.path.join(current_node_path, "CosyVoice")) # 假如用户放的是大写文件夹
        from cosyvoice.cli.cosyvoice import CosyVoice
        HAS_COSYVOICE = True
        print(f"[Fun-CosyVoice3] 通过备用路径成功加载 CosyVoice")
    except Exception:
        HAS_COSYVOICE = False

# ================= 路径配置 =================

COSY_MODELS_DIR = os.path.join(folder_paths.models_dir, "TTS")
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
        raise ImportError(
            "无法加载 'cosyvoice' 模块。\n"
            "解决方法：\n"
            "1. 请确保您已将 'cosyvoice' 文件夹复制到本插件目录中。\n"
            "2. 或者确保 requirements.txt 中的依赖已安装成功。"
        )

    # 1. 确定模型路径
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
            # 实例化模型
            model = CosyVoice(model_dir=model_path, load_jit=False, load_onnx=False, load_trt=False)
            
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