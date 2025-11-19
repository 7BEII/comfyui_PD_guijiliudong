import json
import os
import time as time_module
from datetime import datetime

# 可选导入，避免导入错误
try:
    import requests
except ImportError:
    requests = None

try:
    import urllib3
except ImportError:
    urllib3 = None

class PD_guijiliudong_chat:
    """
    硅基流动对话模型节点
    用于普通对话、代码生成等任务（不包含推理模型）
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # 对话模型列表（排除推理模型）
        chat_models = [
            # 🔥 推荐模型
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "deepseek-ai/DeepSeek-V2.5",
            "THUDM/glm-4-9b-chat",
            
            # DeepSeek系列（对话）
            "deepseek-ai/DeepSeek-V3.2-Exp",
            "Pro/deepseek-ai/DeepSeek-V3.2-Exp",
            "deepseek-ai/DeepSeek-V3",
            "Pro/deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-Coder-V2-Instruct",
            
            # Qwen系列
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "Qwen/Qwen3-32B",
            "Qwen/Qwen3-14B",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "Qwen/Qwen2.5-72B-Instruct-128K",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen2-7B-Instruct",
            "Pro/Qwen/Qwen2.5-7B-Instruct",
            "Pro/Qwen/Qwen2-7B-Instruct",
            "Tongyi-Zhiwen/QwenLong-L1-32B",
            
            # GLM系列
            "zai-org/GLM-4.6",
            "zai-org/GLM-4.5-Air",
            "zai-org/GLM-4.5",
            "THUDM/GLM-Z1-32B-0414",
            "THUDM/GLM-4-32B-0414",
            "THUDM/GLM-Z1-Rumination-32B-0414",
            "THUDM/GLM-4-9B-0414",
            "Pro/THUDM/glm-4-9b-chat",
            
            # 其他模型
            "inclusionAI/Ling-1T",
            "inclusionAI/Ring-flash-2.0",
            "inclusionAI/Ling-flash-2.0",
            "inclusionAI/Ling-mini-2.0",
            "moonshotai/Kimi-K2-Instruct-0905",
            "ByteDance-Seed/Seed-OSS-36B-Instruct",
            "stepfun-ai/step3",
            "baidu/ERNIE-4.5-300B-A47B",
            "ascend-tribe/pangu-pro-moe",
            "tencent/Hunyuan-A13B-Instruct",
            "MiniMaxAI/MiniMax-M1-80k",
            "internlm/internlm2_5-7b-chat"
        ]
        
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "default": "你是一个有用的AI助手，请根据用户的问题提供准确、有帮助的回答。",
                    "multiline": True
                }),
            },
            "optional": {
                "user_prompt": ("STRING", {
                    "default": "你好，请介绍一下你自己。",
                    "multiline": True
                }),
                "model": (chat_models, {"default": "Qwen/Qwen2.5-32B-Instruct"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_text", "model_name", "info")
    FUNCTION = "generate_chat"
    CATEGORY = "PD_Nodes/Chat"
    
    def load_config(self):
        """从config.json加载API密钥"""
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        
        if not os.path.exists(config_path):
            raise Exception("找不到config.json配置文件")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            api_key = None
            if 'siliconflow_api_key' in config:
                api_key = config['siliconflow_api_key']
            elif 'api_key' in config:
                api_key = config['api_key']
            
            if not api_key or api_key.strip() == "":
                raise Exception("在config.json中找不到有效的API密钥")
                
            return api_key.strip()
            
        except Exception as e:
            raise Exception(f"读取config.json失败: {str(e)}")
    
    def calculate_cost(self, model, prompt_tokens, completion_tokens):
        """计算API调用费用（人民币）"""
        # 硅基流动价格表（元/百万tokens）
        pricing = {
            "Qwen/Qwen2.5-72B-Instruct": {"input": 0.35, "output": 0.35},
            "Qwen/Qwen2.5-32B-Instruct": {"input": 0.14, "output": 0.14},
            "Qwen/Qwen2.5-7B-Instruct": {"input": 0.035, "output": 0.035},
            "deepseek-ai/DeepSeek-V2.5": {"input": 0.14, "output": 0.28},
            "deepseek-ai/DeepSeek-V3": {"input": 0.27, "output": 1.1},
            "THUDM/glm-4-9b-chat": {"input": 0.05, "output": 0.05},
            "deepseek-ai/DeepSeek-Coder-V2-Instruct": {"input": 0.14, "output": 0.28},
        }
        
        # 获取价格，如果没有则使用默认价格
        price = pricing.get(model, {"input": 0.1, "output": 0.1})
        
        # 计算费用（元）
        input_cost = (prompt_tokens / 1000000) * price["input"]
        output_cost = (completion_tokens / 1000000) * price["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
    def call_api(self, api_key, model, messages, temperature, max_tokens, top_p):
        """调用硅基流动API"""
        url = "https://api.siliconflow.cn/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False
        }
        
        if not requests:
            raise Exception("requests库未安装")
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 400:
            try:
                error_detail = response.json().get("error", {}).get("message", "请求参数错误")
            except:
                error_detail = "请求参数错误"
            raise Exception(f"HTTP 400: {error_detail}\n提示：某些参数可能不被该模型支持")
        
        if response.status_code == 403:
            raise Exception("HTTP 403: 权限被拒绝\n可能原因：Pro模型需要付费订阅，或免费额度用完")
        
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}"
            try:
                error_detail = response.json().get("error", {}).get("message", "")
                if error_detail:
                    error_msg += f": {error_detail}"
            except:
                pass
            raise Exception(error_msg)
        
        return response.json()
    
    def generate_chat(self, system_prompt, user_prompt="", model="", api_key="", temperature=0.7, max_tokens=2048, top_p=0.9):
        """生成对话内容"""
        
        start_time = time_module.time()
        
        try:
            # 验证输入
            if not user_prompt or not user_prompt.strip():
                raise Exception("user_prompt不能为空")
            
            # 设置默认模型
            if not model or model.strip() == "":
                model = "Qwen/Qwen2.5-32B-Instruct"
            
            # 获取API密钥
            if not api_key or api_key.strip() == "":
                api_key = self.load_config()
            else:
                api_key = api_key.strip()
            
            # 构建消息
            messages = []
            if system_prompt and system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})
            messages.append({"role": "user", "content": user_prompt.strip()})
            
            # 调用API
            print(f"💬 调用对话模型: {model}")
            response = self.call_api(api_key, model, messages, temperature, max_tokens, top_p)
            
            # 计算耗时
            elapsed_time = time_module.time() - start_time
            
            # 提取结果
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                message = choice["message"]
                
                generated_text = message.get("content", "")
                
                # 构建info信息
                info_lines = []
                info_lines.append(f"{'='*40}")
                info_lines.append(f"💬 对话模型调用信息")
                info_lines.append(f"{'='*40}")
                info_lines.append(f"")
                info_lines.append(f"📋 模型: {model}")
                info_lines.append(f"⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                info_lines.append(f"⏱️  耗时: {elapsed_time:.2f} 秒")
                info_lines.append(f"")
                
                # Token统计和费用
                if "usage" in response:
                    usage = response["usage"]
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    
                    info_lines.append(f"📊 Token使用:")
                    info_lines.append(f"   • 输入: {prompt_tokens:,} tokens")
                    info_lines.append(f"   • 输出: {completion_tokens:,} tokens")
                    info_lines.append(f"   • 总计: {total_tokens:,} tokens")
                    info_lines.append(f"")
                    
                    # 计算费用
                    cost = self.calculate_cost(model, prompt_tokens, completion_tokens)
                    info_lines.append(f"💰 费用估算:")
                    info_lines.append(f"   • 输入: ¥{cost['input_cost']:.6f}")
                    info_lines.append(f"   • 输出: ¥{cost['output_cost']:.6f}")
                    info_lines.append(f"   • 总计: ¥{cost['total_cost']:.6f}")
                    info_lines.append(f"")
                
                # 输出统计
                output_len = len(generated_text)
                info_lines.append(f"📝 输出长度: {output_len:,} 字符")
                info_lines.append(f"✅ 状态: {choice.get('finish_reason', 'unknown')}")
                info_lines.append(f"{'='*40}")
                
                info_text = "\n".join(info_lines)
                
                return (generated_text, model, info_text)
            else:
                error_msg = "API返回格式异常"
                error_info = f"❌ 错误\n📋 模型: {model}\n🚫 {error_msg}"
                return (error_msg, model, error_info)
                
        except Exception as e:
            elapsed_time = time_module.time() - start_time
            error_msg = f"对话过程出现错误: {str(e)}"
            error_info = f"❌ 错误\n📋 模型: {model}\n⏱️  耗时: {elapsed_time:.2f}秒\n🚫 {str(e)}"
            print(f"错误: {error_msg}")
            return (error_msg, model, error_info)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "PD_guijiliudong_chat": PD_guijiliudong_chat
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PD_guijiliudong_chat": "PD 对话模型 (Chat)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

