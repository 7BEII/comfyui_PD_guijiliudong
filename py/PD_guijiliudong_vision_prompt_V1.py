# PD_guijiliudong_visiontext_v1.py
import base64
import json
import requests
import os
from io import BytesIO
from PIL import Image
import torch
import numpy as np

class PD_guijiliudong_vision_prompt_V1:
    """
    硅基流动图片反推提示词节点
    用于输入一张图片，通过VLM模型反推出AI提示词
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # VLM模型列表（支持图片输入的模型）
        vlm_models = [
            # Qwen3 系列（最新，推荐用于提示词反推）
            "Qwen/Qwen3-VL-32B-Instruct",  # 推荐：平衡性能和速度
            "Qwen/Qwen3-VL-8B-Instruct",  # 快速版本
            "Qwen/Qwen3-VL-235B-A22B-Instruct",  # 高性能版本
            "Qwen/Qwen3-VL-30B-A3B-Instruct",  # 中等性能
            "Qwen/Qwen3-VL-8B-Thinking",  # 推理增强
            "Qwen/Qwen3-VL-32B-Thinking",  # 推理增强
            "Qwen/Qwen3-VL-30B-A3B-Thinking",  # 推理增强
            "Qwen/Qwen3-VL-235B-A22B-Thinking",  # 推理增强
            # Qwen2.5 系列（稳定可靠）
            "Qwen/Qwen2.5-VL-32B-Instruct",  # 稳定版本
            "Qwen/Qwen2.5-VL-72B-Instruct",  # 高性能
            "Pro/Qwen/Qwen2.5-VL-7B-Instruct",  # 快速版
            "Qwen/Qwen2.5-VL-7B-Instruct",
            # Qwen2 系列
            "Qwen/Qwen2-VL-72B-Instruct",  # 经典版本
            "Pro/Qwen/Qwen2-VL-7B-Instruct",  # 快速版
            "Qwen/QVQ-72B-Preview",  # 推理增强版
            # GLM 系列（智谱清言）
            "zai-org/GLM-4.5V",  # 推荐：GLM视觉模型
            "Pro/THUDM/GLM-4.5V-8B-Vision-Thinking",  # 推理增强
            "THUDM/GLM-4.1V-9B-Thinking",  # 推理增强
            # DeepSeek 系列
            "deepseek-ai/deepseek-vl2",  # 识别准确
        ]
        
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图片，格式为 B H W C
                "model": (vlm_models, {"default": "Qwen/Qwen3-VL-32B-Instruct"}),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "留空则从config.json读取"
                }),
                "system_prompt": ("STRING", {
                    "default": "You are an AI prompt expert who can analyze images. Please look closely at the image and provide a detailed and accurate description as required.",
                    "multiline": True
                }),
                "user_prompt": ("STRING", {
                    "default": "Help me describe, in a paragraph of AI prompts, This style of clothing is uniformly called boshifu, Don't describe clothing details and colors, in no more than 128 words, in English. such as: a girl wearing a boshifu stood beside the road sign on Tsinghua Road, smiling and holding sunflowers. The background was a green forest, and the sunlight shone on her. The scene was filled with a warm atmosphere, showcasing the joy of graduation season and the vitality of youth.",
                    "multiline": True
                }),
                "base_url": ("STRING", {
                    "default": "https://api.siliconflow.cn/v1",
                    "multiline": False,
                    "placeholder": "API基础地址"
                }),
                "detail": (["high", "low", "auto"], {"default": "auto"}),
                "temperature": ("FLOAT", {"default": 0.71, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "image_to_prompt"
    CATEGORY = "PD_Nodes/VisionPrompt"
    
    def load_config(self):
        """从config.json加载API密钥和base_url"""
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        
        if not os.path.exists(config_path):
            # 如果config.json不存在，返回空配置
            return {"api_key": None, "base_url": "https://api.siliconflow.cn/v1"}
        
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
            
            # 获取base_url
            base_url = config.get('base_url', 'https://api.siliconflow.cn/v1')
            
            return {
                "api_key": api_key.strip() if api_key else None,
                "base_url": base_url
            }
            
        except json.JSONDecodeError:
            print("警告: config.json格式错误，使用默认配置")
            return {"api_key": None, "base_url": "https://api.siliconflow.cn/v1"}
        except Exception as e:
            print(f"警告: 读取config.json失败: {str(e)}")
            return {"api_key": None, "base_url": "https://api.siliconflow.cn/v1"}
    
    def tensor_to_base64(self, tensor):
        """将ComfyUI的图像tensor转换为base64编码"""
        try:
            # 确保tensor是正确的形状 (B, H, W, C)
            if tensor.dim() == 4:
                # 如果batch维度大于1，只取第一张图片
                if tensor.shape[0] > 1:
                    tensor = tensor[0]
                else:
                    tensor = tensor.squeeze(0)  # 移除batch维度，得到 (H, W, C)
            
            # 转换为numpy数组并调整数据类型
            image_np = tensor.cpu().numpy()
            if image_np.dtype != np.uint8:
                # 如果数据在[0,1]范围内，转换为[0,255]
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
                else:
                    image_np = image_np.clip(0, 255).astype(np.uint8)
            
            # 创建PIL图像
            if len(image_np.shape) == 3:
                if image_np.shape[2] == 3:  # RGB
                    image = Image.fromarray(image_np, 'RGB')
                elif image_np.shape[2] == 4:  # RGBA
                    image = Image.fromarray(image_np, 'RGBA')
                    # 转换为RGB（去除alpha通道）
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
                    image = rgb_image
                else:
                    raise ValueError(f"不支持的图像通道数: {image_np.shape[2]}")
            elif len(image_np.shape) == 2:  # 灰度图
                image = Image.fromarray(image_np, 'L')
                # 转换为RGB
                image = image.convert('RGB')
            else:
                raise ValueError(f"不支持的图像维度: {image_np.shape}")
            
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
    
    def call_siliconflow_api(self, api_key, model, messages, base_url="https://api.siliconflow.cn/v1", temperature=0.71, max_tokens=512):
        """调用硅基流动API"""
        # 确保base_url以/v1结尾
        if not base_url.endswith('/v1'):
            if base_url.endswith('/'):
                base_url = base_url + 'v1'
            else:
                base_url = base_url + '/v1'
        
        url = f"{base_url}/chat/completions"
        print(f"API URL: {url}")
        print(f"使用模型: {model}")
        
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
        
        print(f"=== API 请求信息 ===")
        print(f"URL: {url}")
        print(f"Model: {model}")
        print(f"Temperature: {temperature}")
        print(f"Max Tokens: {max_tokens}")
        print(f"===================")
        
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
                            error_info = result["error"]
                            error_msg = error_info.get("message", "未知API错误")
                            error_type = error_info.get("type", "未知类型")
                            error_code = error_info.get("code", "未知代码")
                            
                            # 针对不同错误类型提供解决方案
                            if "not found" in error_msg.lower() or "model" in error_msg.lower():
                                raise Exception(f"模型错误: {error_msg}\n\n可能的原因:\n1. 该模型在当前API服务中不可用\n2. 模型名称拼写错误\n3. 您的账户没有访问该模型的权限\n\n建议:\n- 尝试其他模型(如 Qwen/Qwen2.5-VL-32B-Instruct)\n- 检查您的API服务商支持的模型列表")
                            elif "auth" in error_msg.lower() or "key" in error_msg.lower():
                                raise Exception(f"认证错误: {error_msg}\n\n请检查:\n1. API Key是否正确\n2. API Key是否已过期\n3. 账户余额是否充足")
                            else:
                                raise Exception(f"API错误 [{error_type} - {error_code}]: {error_msg}")
                        
                        return result
                        
                    finally:
                        # 恢复环境变量
                        for var, value in env_backup.items():
                            os.environ[var] = value
                            
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if hasattr(e, 'response') else 'unknown'
                print(f"配置 {i+1} 失败: HTTP {status_code} - {str(e)}")
                if i == len(configs) - 1:  # 最后一次尝试
                    error_solutions = f"""
API请求失败 (HTTP {status_code})

请求信息:
- URL: {url}
- Model: {model}

错误详情: {str(e)}

可能的原因和解决方案:
1. 【模型不可用】: 该模型可能在您的API服务中不支持
   - 建议: 尝试其他模型如 Qwen/Qwen2.5-VL-32B-Instruct
   
2. 【API Key问题】: 密钥错误或无权限
   - 检查: 确认API Key是否正确
   - 检查: 账户是否有余额
   
3. 【Base URL错误】: 当前使用的Base URL可能不正确
   - 当前URL: {base_url}
   - 标准URL: https://api.siliconflow.cn/v1
   - 建议: 在节点参数中修改base_url

4. 【网络连接】: 网络无法访问API服务
   - 尝试使用VPN或更换网络环境
                    """.strip()
                    raise Exception(error_solutions)
            except Exception as e:
                print(f"配置 {i+1} 失败: {str(e)}")
                if i == len(configs) - 1:  # 最后一次尝试
                    error_solutions = f"""
网络连接失败

请求信息:
- URL: {url}
- Model: {model}

错误详情: {str(e)}

可能的解决方案:
1. 检查网络连接是否正常
2. 确认Base URL是否正确 (当前: {base_url})
3. 检查防火墙是否阻止了连接
4. 确认API密钥是否正确
5. 尝试使用VPN或更换网络环境
6. 稍后重试（可能是服务器临时问题）
                    """.strip()
                    raise Exception(error_solutions)
                
                # 等待后重试
                import time
                time.sleep(2)
                continue
        
        raise Exception("所有网络配置尝试都失败了")
    
    def image_to_prompt(self, image, model, api_key="", system_prompt="", user_prompt="", base_url="https://api.siliconflow.cn/v1", detail="auto", temperature=0.71, max_tokens=512):
        """将图片反推为提示词"""
        
        try:
            # 验证图片张量格式
            if image.dim() == 4:
                batch, height, width, channels = image.shape
                print(f"接收到图片: Batch={batch}, Height={height}, Width={width}, Channels={channels}")
            else:
                print(f"图片张量维度: {image.shape}")
            
            # 优先使用节点参数中的api_key和base_url，如果为空则从配置文件读取
            config = self.load_config()
            
            # 如果节点参数中没有提供api_key，从配置文件读取
            if not api_key or api_key.strip() == "":
                api_key = config["api_key"]
                if not api_key:
                    raise Exception("未提供API密钥。请在节点参数中输入API密钥，或在config.json中配置")
            else:
                api_key = api_key.strip()
            
            # 如果节点参数中没有提供base_url或使用默认值，尝试从配置文件读取
            if not base_url or base_url == "https://api.siliconflow.cn/v1":
                if config["base_url"] and config["base_url"] != "https://api.siliconflow.cn/v1":
                    base_url = config["base_url"]
            
            print(f"使用API URL: {base_url}")
            print(f"使用模型: {model}")
            
            # 转换图像为base64
            print("正在处理图片...")
            base64_image = self.tensor_to_base64(image)
            
            # 构建消息内容
            message_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image,
                        "detail": detail
                    }
                },
                {
                    "type": "text",
                    "text": user_prompt if user_prompt else "Please describe this image in detail as an AI prompt."
                }
            ]
            
            # 构建API请求消息
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": message_content
            })
            
            # 调用API
            print("正在调用硅基流动API...")
            response = self.call_siliconflow_api(
                api_key=api_key,
                model=model,
                messages=messages,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 提取分析结果
            if "choices" in response and len(response["choices"]) > 0:
                prompt_result = response["choices"][0]["message"]["content"]
                print(f"成功生成提示词: {prompt_result[:100]}...")
                return (prompt_result,)
            else:
                return (f"API返回格式异常: {json.dumps(response, indent=2, ensure_ascii=False)}",)
                
        except Exception as e:
            error_msg = f"图片反推提示词过程中出现错误: {str(e)}"
            print(f"错误: {error_msg}")
            return (error_msg,)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "PD_guijiliudong_vision_prompt_V1": PD_guijiliudong_vision_prompt_V1
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PD_guijiliudong_vision_prompt_V1": "PD guijiliudong vision prompt V1"
}

# 节点描述
NODE_DESCRIPTIONS = {
    "PD_guijiliudong_vision_prompt_V1": "使用硅基流动API将图片反推为AI提示词，支持自定义系统提示词和用户提示词"
}

# 导出节点类
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'NODE_DESCRIPTIONS']
