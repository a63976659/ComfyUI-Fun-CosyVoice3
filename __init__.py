from .fun_node import Fun_CosyVoice3_Node

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "Fun_CosyVoice3": Fun_CosyVoice3_Node
}

# 节点显示名称映射 (这里修改了中文名)
NODE_DISPLAY_NAME_MAPPINGS = {
    "Fun_CosyVoice3": "🎤 Fun CosyVoice3 语音合成"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]