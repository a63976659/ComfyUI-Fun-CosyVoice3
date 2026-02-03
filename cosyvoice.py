import torch
import os
import tempfile
import torchaudio
import numpy as np
import random
from .utils_cosyvoice import load_cosyvoice_model, unload_cosyvoice_model

# 语言列表
LANGUAGES_LIST = ["不指定", "中文", "英语", "日语", "韩语", "德语", "西班牙语", "法语", "意大利语", "俄语"]

# 方言列表
DIALECTS_LIST = ["无", "广东话", "闽南话", "四川话", "东北话", "河南话", "陕西话", "山西话", "上海话", "天津话", "山东话", "宁夏话", "甘肃话"]

class Fun_CosyVoice3_Node:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 核心输入
                "参考音频": ("AUDIO", ),
                "文本内容": ("STRING", {"multiline": True, "default": "你好，欢迎使用 CosyVoice 3.0 语音合成系统。"}),
                
                # 模式选择
                "模式": (["零样本复刻 (Zero-shot)", "指令控制 (Instruct)", "跨语言/精细控制 (Cross-lingual)"], {"default": "零样本复刻 (Zero-shot)"}),
                "参考音频文本": ("STRING", {"multiline": False, "default": "", "placeholder": "【零样本模式必填】输入参考音频里说的话"}),
                
                # --- 细分控制组件 ---
                "语言": (LANGUAGES_LIST, {"default": "中文"}),
                "方言": (DIALECTS_LIST, {"default": "无"}),
                "情感": ("STRING", {"multiline": False, "default": "", "placeholder": "例如：悲伤、激动"}),
                "语速": ("FLOAT", {"default": 0, "min": -30, "max": 30, "step": 0.5, "display": "slider"}),
                "音量": ("FLOAT", {"default": 0, "min": -30, "max": 30, "step": 0.5, "display": "slider"}),
                
                # 系统设定
                "系统提示词": ("STRING", {"multiline": False, "default": "You are a helpful assistant.", "label": "系统设定"}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                
                # 下载设置
                "下载源": (["ModelScope", "HuggingFace", "HF Mirror"], {"default": "ModelScope"}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频输出",)
    FUNCTION = "generate_audio"
    CATEGORY = "💬 AI人工智能"
    DESCRIPTION = (
        "【Fun-CosyVoice 3.0】\n"
        "⚠️ 官方建议：单句文本不要超过 30秒，否则音质和节奏会下降。\n"
        "1. 零样本复刻：必须填写'参考音频文本'。会自动加上语言/情感等指令。\n"
        "2. 指令控制：主要依赖语言、方言、情感等参数控制。\n"
        "3. 机制说明：为防止模型念出提示词，系统会自动构建标准的 <|endofprompt|> 格式。"
    )

    def _save_temp_wav(self, audio_input):
        """保存临时 WAV 用于 CosyVoice 输入"""
        waveform = audio_input['waveform'] 
        sample_rate = audio_input['sample_rate']
        
        if waveform.dim() == 3:
            wav_tensor = waveform[0]
        else:
            wav_tensor = waveform

        if wav_tensor.shape[0] > wav_tensor.shape[1]: 
             wav_tensor = wav_tensor.t()

        # 必须重采样到 16k
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            wav_tensor = resampler(wav_tensor)
            sample_rate = 16000

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        torchaudio.save(temp_file.name, wav_tensor.cpu(), sample_rate)
        return temp_file.name

    def _construct_instruction(self, 语言, 方言, 情感, 语速, 音量):
        """将离散参数组装成自然语言指令"""
        parts = []
        
        if 语言 not in ["不指定", "无"]:
            parts.append(f"请用{语言}。")
            
        if 方言 not in ["不指定", "无"]:
            parts.append(f"使用{方言}。")
            
        if 情感.strip():
            parts.append(f"使用{情感.strip()}的语气。")
        
        if 语速 != 0:
            speed_str = "加快" if 语速 > 0 else "放慢"
            parts.append(f"语速{speed_str}{abs(语速)}。")
            
        if 音量 != 0:
            vol_str = "调大" if 音量 > 0 else "调小"
            parts.append(f"音量{vol_str}{abs(音量)}。")
            
        return "".join(parts)

    def _construct_final_prompt(self, sys_prompt, instruct_str, ref_text=""):
        """
        严格按照官方格式构建 Prompt: 
        System + Instruct + <|endofprompt|> + Reference Text (可选)
        """
        # 官方强调必须有 instruct，如果用户没填任何控制参数，我们给一个默认的无害指令
        # 防止模型因为缺少 instruct 而把 system prompt 当成文本念出来
        final_instruct = instruct_str
        if not final_instruct.strip():
            final_instruct = "请使用该声音合成。" 

        # 拼接 System 和 Instruct
        # 注意：不要自己乱加换行符，除非确定模型需要。CosyVoice通常是紧凑拼接。
        header = f"{sys_prompt} {final_instruct}".strip()
        
        # 加上核心分隔符
        full_prompt = f"{header}<|endofprompt|>"
        
        # 如果有参考文本（Zero-shot模式），追加在后面
        if ref_text.strip():
            full_prompt += ref_text.strip()
            
        return full_prompt

    def generate_audio(self, 参考音频, 文本内容, 模式, 参考音频文本, 语言, 方言, 情感, 语速, 音量, 系统提示词, 随机种子, 下载源, 自动下载模型):
        
        # 长度警告
        if len(文本内容) > 100: # 估算值，中文100字大概30秒左右
            print("[Warning] 待合成文本较长，CosyVoice3 建议单句不超过 30秒，否则可能导致复读或音质下降。")

        # 设置随机种子
        if 随机种子 is not None:
            torch.manual_seed(随机种子)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(随机种子)
        
        temp_wav_path = None
        model_name = "Fun-CosyVoice3-0.5B-2512"

        try:
            # 1. 准备参考音频
            temp_wav_path = self._save_temp_wav(参考音频)
            
            # 2. 加载模型
            model = load_cosyvoice_model(model_name, self.device, auto_download=自动下载模型, source=下载源)
            
            print(f"[CosyVoice3] Generating... Mode: {模式}")
            
            # 3. 构建指令部分
            instruct_str = self._construct_instruction(语言, 方言, 情感, 语速, 音量)
            if instruct_str:
                print(f"[CosyVoice3] Instruct: {instruct_str}")

            generator = None
            
            # 4. 根据模式执行推理
            if 模式 == "零样本复刻 (Zero-shot)":
                if not 参考音频文本.strip():
                    raise ValueError("【零样本模式】必须填写'参考音频文本'！")
                
                # 构建完整的 Prompt (包含 System, Instruct, Separator, RefText)
                final_prompt = self._construct_final_prompt(系统提示词, instruct_str, 参考音频文本)
                print(f"[Debug] Final Prompt: {final_prompt}")

                generator = model.inference_zero_shot(
                    tts_text=文本内容,
                    prompt_text=final_prompt,
                    prompt_speech_16k=temp_wav_path,
                    stream=False
                )

            elif 模式 == "指令控制 (Instruct)":
                # Instruct 模式没有 RefText，但同样必须有 <|endofprompt|>
                final_prompt = self._construct_final_prompt(系统提示词, instruct_str, ref_text="")
                print(f"[Debug] Final Prompt: {final_prompt}")

                generator = model.inference_instruct2(
                    tts_text=文本内容,
                    prompt_text=final_prompt,
                    prompt_speech_16k=temp_wav_path,
                    stream=False
                )

            elif 模式 == "跨语言/精细控制 (Cross-lingual)":
                # Cross-lingual 官方接口可能不使用 text prompt，或者只使用基础 prompt
                # 但为了安全，我们还是构建一个基础的 prompt 对象
                # 注意：inference_cross_lingual 的参数定义可能不包含 prompt_text，视具体版本而定
                # 这里我们假设它主要依赖音频特征。如果支持 prompt_text，逻辑同上。
                
                generator = model.inference_cross_lingual(
                    tts_text=文本内容,
                    prompt_speech_16k=temp_wav_path,
                    stream=False
                )

            # 5. 获取结果
            final_output = None
            for i, result in enumerate(generator):
                final_output = result
            
            if final_output is None:
                raise Exception("生成失败，模型未返回音频数据。")

            out_wav = final_output['tts_speech'] 
            target_sr = model.sample_rate 
            
            if out_wav.dim() == 1:
                out_wav = out_wav.unsqueeze(0).unsqueeze(0)
            elif out_wav.dim() == 2:
                out_wav = out_wav.unsqueeze(0)

            return ({"waveform": out_wav.cpu(), "sample_rate": target_sr},)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"CosyVoice Error: {str(e)}")
        
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            unload_cosyvoice_model()