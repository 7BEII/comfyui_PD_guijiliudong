import base64
import json
import requests
import os
from io import BytesIO
from PIL import Image
import torch
import numpy as np

class PD_guijiliudong_vision_v1:
    """
    硅基流动多模态视觉分析节点
    用于智能对比分析两张图片并生成详细的文字描述
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # VLM模型列表
        vlm_models = [
            "Qwen/Qwen2.5-VL-32B-Instruct", 
            "Qwen/Qwen2.5-VL-72B-Instruct", 
            "Qwen/QVQ-72B-Preview",
            "Qwen/Qwen2-VL-72B-Instruct",
            "Pro/Qwen/Qwen2-VL-7B-Instruct",
            "Pro/Qwen/Qwen2.5-VL-7B-Instruct",
            "deepseek-ai/deepseek-vl2"
        ]
        
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "model": (vlm_models, {"default": "Qwen/Qwen2.5-VL-72B-Instruct"}),
                "prompt": ("STRING", {
                    "default": "请详细对比分析这两张图片的异同点，包括内容、构图、色彩、风格等方面的差异和相似之处。",
                    "multiline": True
                }),
            },
            "optional": {
                "detail": (["high", "low", "auto"], {"default": "high"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 4096}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis_result",)
    FUNCTION = "analyze_images"
    CATEGORY = "PD_Nodes/Vision"
    
    def load_config(self):
        """从config.json加载API密钥"""
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        
        if not os.path.exists(config_path):
            # 如果config.json不存在，检查config.json.example
            example_config_path = os.path.join(os.path.dirname(__file__), "config.json.example")
            if os.path.exists(example_config_path):
                raise Exception(f"请将 {example_config_path} 重命名为 config.json 并填入你的API密钥")
            else:
                raise Exception("找不到config.json配置文件")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 支持多种可能的配置结构
            api_key = None
            if 'siliconflow_api_key' in config:
                api_key = config['siliconflow_api_key']
            elif 'api_key' in config:
                api_key = config['api_key']
            elif 'siliconflow' in config and 'api_key' in config['siliconflow']:
                api_key = config['siliconflow']['api_key']
            
            if not api_key or api_key.strip() == "" or api_key == "your_siliconflow_api_key_here":
                raise Exception("在config.json中找不到有效的硅基流动API密钥，请检查配置")
                
            return api_key.strip()
            
        except json.JSONDecodeError:
            raise Exception("config.json格式错误，请检查JSON语法")
        except Exception as e:
            raise Exception(f"读取config.json失败: {str(e)}")
    
    def tensor_to_base64(self, tensor):
        """将ComfyUI的图像tensor转换为base64编码"""
        try:
            # 确保tensor是正确的形状和数据类型
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)  # 移除batch维度
            
            # 转换为numpy数组并调整数据类型
            image_np = tensor.cpu().numpy()
            if image_np.dtype != np.uint8:
                image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
            
            # 创建PIL图像
            if image_np.shape[2] == 3:  # RGB
                image = Image.fromarray(image_np, 'RGB')
            elif image_np.shape[2] == 4:  # RGBA
                image = Image.fromarray(image_np, 'RGBA')
            elif image_np.shape[2] == 1:  # 灰度图
                image = Image.fromarray(image_np.squeeze(2), 'L')
            else:
                raise ValueError(f"不支持的图像通道数: {image_np.shape[2]}")
            
            # 优化图片大小以减少API调用成本
            max_size = 1024
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # 转换为base64
            buffer = BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{base64_str}"
            
        except Exception as e:
            raise Exception(f"图像转换失败: {str(e)}")
    
    def call_siliconflow_api(self, api_key, model, messages, temperature=0.7, max_tokens=2048):
        """调用硅基流动API"""
        url = "https://api.siliconflow.cn/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ComfyUI-SiliconFlow/1.0"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        import ssl
        import urllib3
        
        # 完全禁用SSL警告
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 尝试多种网络配置
        configs = [
            # 配置1：标准SSL配置
            {"verify": True, "proxies": None},
            # 配置2：禁用SSL验证
            {"verify": False, "proxies": None},
            # 配置3：使用urllib3直接请求
            {"use_urllib3": True}
        ]
        
        for i, config in enumerate(configs):
            try:
                print(f"尝试连接配置 {i+1}/{len(configs)}...")
                
                if config.get("use_urllib3"):
                    # 使用urllib3直接请求
                    import urllib3
                    http = urllib3.PoolManager(cert_reqs='CERT_NONE')
                    
                    response = http.request(
                        'POST',
                        url,
                        body=json.dumps(payload).encode('utf-8'),
                        headers=headers,
                        timeout=120
                    )
                    
                    if response.status == 200:
                        result = json.loads(response.data.decode('utf-8'))
                        if "error" in result:
                            error_msg = result["error"].get("message", "未知API错误")
                            raise Exception(f"API返回错误: {error_msg}")
                        return result
                    else:
                        raise Exception(f"HTTP错误: {response.status}")
                
                else:
                    # 使用requests
                    import requests
                    
                    # 清除所有代理设置
                    import os
                    env_backup = {}
                    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
                    for var in proxy_vars:
                        if var in os.environ:
                            env_backup[var] = os.environ[var]
                            del os.environ[var]
                    
                    try:
                        response = requests.post(
                            url,
                            headers=headers,
                            json=payload,
                            timeout=120,
                            verify=config["verify"],
                            proxies={"http": None, "https": None}
                        )
                        
                        response.raise_for_status()
                        result = response.json()
                        
                        if "error" in result:
                            error_msg = result["error"].get("message", "未知API错误")
                            raise Exception(f"API返回错误: {error_msg}")
                        
                        return result
                        
                    finally:
                        # 恢复环境变量
                        for var, value in env_backup.items():
                            os.environ[var] = value
                            
            except Exception as e:
                print(f"配置 {i+1} 失败: {str(e)}")
                if i == len(configs) - 1:  # 最后一次尝试
                    # 提供详细的解决方案
                    error_solutions = """
网络连接失败，可能的解决方案：
1. 检查网络连接是否正常
2. 尝试使用VPN或更换网络环境
3. 检查防火墙设置
4. 确认API密钥是否正确
5. 稍后重试（可能是服务器临时问题）
6. 联系硅基流动技术支持

当前错误: {error}
                    """.format(error=str(e)).strip()
                    
                    raise Exception(error_solutions)
                
                # 等待后重试
                import time
                time.sleep(2)
                continue
        
        raise Exception("所有网络配置尝试都失败了")
    
    def analyze_images(self, image1, image2, model, prompt, detail="high", temperature=0.7, max_tokens=2048):
        """分析两张图片"""
        
        try:
            # 从配置文件加载API密钥
            api_key = self.load_config()
            
            # 转换图像为base64
            print("正在处理图片...")
            base64_image1 = self.tensor_to_base64(image1)
            base64_image2 = self.tensor_to_base64(image2)
            
            # 构建消息内容
            message_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image1,
                        "detail": detail
                    }
                },
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": base64_image2,
                        "detail": detail
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            
            # 构建API请求消息
            messages = [
                {
                    "role": "user",
                    "content": message_content
                }
            ]
            
            # 调用API
            print("正在调用硅基流动API...")
            response = self.call_siliconflow_api(
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 提取分析结果
            if "choices" in response and len(response["choices"]) > 0:
                analysis_result = response["choices"][0]["message"]["content"]
                
                return (analysis_result,)
            else:
                return (f"API返回格式异常: {json.dumps(response, indent=2, ensure_ascii=False)}",)
                
        except Exception as e:
            error_msg = f"分析过程中出现错误: {str(e)}"
            print(f"错误: {error_msg}")
            return (error_msg,)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "PD_guijiliudong_vision_v1": PD_guijiliudong_vision_v1
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PD_guijiliudong_vision_v1": "PD_guijiliudong_vision_v1"
}

# 导出节点类
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'NODE_DESCRIPTIONS'] 