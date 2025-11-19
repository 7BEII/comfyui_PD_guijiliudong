import base64
import json
import requests
import os
from io import BytesIO
from PIL import Image
import torch
import numpy as np
import time as time_module
from datetime import datetime
import threading
import signal


class TimeoutException(Exception):
    """超时异常"""
    pass


def timeout_handler(func, timeout_seconds):
    """
    带超时的函数执行器
    使用线程来确保在指定时间后强制返回
    """
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # 线程还在运行，说明超时了
        raise TimeoutException(f"操作超时（{timeout_seconds}秒）- 已自动中断")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]


class PD_guijiliudong_vision_v1:
    """
    硅基流动多模态视觉分析节点 - 双图对比专用版
    用于智能对比分析两张图片并生成详细的文字描述
    
    ⚠️ 重要说明：
    - 本节点专门用于双图对比，仅包含明确支持多图输入的VLM模型
    - 不支持多图的模型已被移除，避免长时间等待后失败
    - 推荐使用 Qwen/Qwen2.5-VL-32B-Instruct（平衡速度和质量）
    - 如需单图分析，请使用 "PD guijiliudong VLM" 节点
    """
    
    # 模型定价表（元/百万tokens）- 仅包含支持双图的模型
    # 数据来源：硅基流动官网 https://siliconflow.cn/pricing
    MODEL_PRICING = {
        # ⭐⭐⭐ Qwen2.5-VL 系列（强烈推荐，快速稳定）
        "Qwen/Qwen2.5-VL-72B-Instruct": {"input": 0.8, "output": 0.8},
        "Qwen/Qwen2.5-VL-32B-Instruct": {"input": 0.5, "output": 0.5},
        "Qwen/Qwen2.5-VL-7B-Instruct": {"input": 0.2, "output": 0.2},
        
        # ⭐⭐ Qwen2-VL 系列（经典版本，稳定支持双图）
        "Qwen/Qwen2-VL-72B-Instruct": {"input": 0.8, "output": 0.8},
        "Pro/Qwen/Qwen2-VL-7B-Instruct": {"input": 0.35, "output": 0.35},
        
        # ⭐ Qwen3-VL 系列（最新版本，处理时间较长）
        "Qwen/Qwen3-VL-8B-Instruct": {"input": 0.2, "output": 0.2},
        "Qwen/Qwen3-VL-32B-Instruct": {"input": 0.5, "output": 0.5},
        
        # 默认定价（如果模型不在列表中）
        "_default": {"input": 0.5, "output": 0.5},
    }
    
    def __init__(self):
        pass
    
    def calculate_cost(self, model, prompt_tokens, completion_tokens):
        """
        计算API调用成本
        
        Args:
            model: 模型名称
            prompt_tokens: 输入token数
            completion_tokens: 输出token数
            
        Returns:
            tuple: (input_cost, output_cost, total_cost) 单位：元
        """
        # 获取模型定价，如果不存在则使用默认定价
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["_default"])
        
        # 计算成本（价格是每百万tokens）
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return input_cost, output_cost, total_cost
    
    def format_cost(self, cost):
        """
        格式化价格显示
        
        Args:
            cost: 价格（元）
            
        Returns:
            str: 格式化的价格字符串
        """
        if cost >= 0.01:
            # 大于等于1分，显示元
            return f"¥{cost:.4f}"
        elif cost >= 0.001:
            # 1-10毫，显示分
            return f"¥{cost:.4f} ({cost*100:.2f}分)"
        elif cost > 0:
            # 小于1毫，显示厘
            return f"¥{cost:.6f} ({cost*1000:.3f}厘)"
        else:
            return "¥0.0000"
    
    @classmethod
    def INPUT_TYPES(cls):
        # ⚠️ 重要说明：本节点专门用于双图对比分析，仅包含明确支持多图输入的VLM模型
        # 某些VLM模型仅支持单图输入，使用它们会导致长时间等待后失败
        # 参考：https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions
        
        vlm_models = [
            # ⭐⭐⭐ Qwen2.5-VL 系列（推荐，快速稳定）
            "Qwen/Qwen2.5-VL-32B-Instruct",  # 最推荐：平衡速度和质量
            "Qwen/Qwen2.5-VL-7B-Instruct",   # 快速版本，性价比高
            "Qwen/Qwen2.5-VL-72B-Instruct",  # 高精度版本，专业分析
            
            # ⭐⭐ Qwen2-VL 系列（经典版本，稳定）
            "Qwen/Qwen2-VL-72B-Instruct",    # 经典高性能版
            "Pro/Qwen/Qwen2-VL-7B-Instruct", # 经典Pro快速版
            
            # ⭐ Qwen3-VL 系列（最新版本，处理时间较长）
            "Qwen/Qwen3-VL-8B-Instruct",     # 支持双图，快速版本
            "Qwen/Qwen3-VL-32B-Instruct",    # 支持双图，但需要更长时间（90-120s）
            
            # ⚠️ 以下模型经测试不支持双图，已移除：
            # ❌ Qwen3-VL-235B - 不支持双图
            # ❌ GLM系列（4.5/4.6/4.5V）- 不支持双图输入
            # ❌ DeepSeek-vl2 - 不支持双图输入
            # ❌ DeepSeek-V3/R1系列 - 不支持双图
            # ❌ GLM-4.5-Air - 不支持双图
            
            # 💡 速度对比：
            # - Qwen2.5-VL 和 Qwen2-VL：速度快，适合批量处理（推荐）
            # - Qwen3-VL-8B：中等速度，建议timeout≥60秒
            # - Qwen3-VL-32B：速度较慢，建议timeout≥90秒
            # - 单图分析请使用 "PD guijiliudong VLM" 节点
        ]
        
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "model": (vlm_models, {"default": "Qwen/Qwen2.5-VL-32B-Instruct"}),
                "prompt": ("STRING", {
                    "default": "请详细对比分析这两张图片的异同点，包括内容、构图、色彩、风格等方面的差异和相似之处。",
                    "multiline": True
                }),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "detail": (["high", "low", "auto"], {"default": "high"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 4096}),
                "timeout": ("INT", {"default": 40, "min": 10, "max": 300, "step": 5}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_text", "model_name", "info")
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
    
    def call_siliconflow_api(self, api_key, model, messages, temperature=0.7, max_tokens=2048, max_retries=3, timeout=60):
        """调用硅基流动API
        
        Args:
            api_key: API密钥
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            max_retries: 最大重试次数（默认3次）
            timeout: 超时时间（秒，默认60秒）
        """
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
        from urllib3.exceptions import TimeoutError as Urllib3TimeoutError
        
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
        
        last_error = None
        
        # 总重试次数限制
        for retry_count in range(max_retries):
            for i, config in enumerate(configs):
                try:
                    print(f"尝试连接配置 {i+1}/{len(configs)} (总重试 {retry_count+1}/{max_retries})...")
                    
                    if config.get("use_urllib3"):
                        # 使用urllib3直接请求
                        http = urllib3.PoolManager(
                            cert_reqs='CERT_NONE',
                            timeout=urllib3.util.Timeout(connect=10, read=timeout)
                        )
                        
                        response = http.request(
                            'POST',
                            url,
                            body=json.dumps(payload).encode('utf-8'),
                            headers=headers
                        )
                        
                        if response.status == 200:
                            result = json.loads(response.data.decode('utf-8'))
                            if "error" in result:
                                error_msg = result["error"].get("message", "未知API错误")
                                error_type = result["error"].get("type", "unknown")
                                error_code = result["error"].get("code", "unknown")
                                
                                # 特殊处理模型不存在的错误
                                if "model" in error_msg.lower() or "not found" in error_msg.lower():
                                    raise Exception(f"❌ 模型错误: {error_msg}\n\n💡 解决方案:\n1. 该模型可能不支持双图输入或暂时不可用\n2. 本节点支持的双图对比模型（共7个）:\n   ⭐⭐⭐ Qwen/Qwen2.5-VL-32B-Instruct（最推荐，快速）\n   ⭐⭐ Qwen/Qwen2.5-VL-7B-Instruct（最快最便宜）\n   ⭐⭐ Qwen/Qwen2.5-VL-72B-Instruct（高精度）\n   ⭐ Qwen/Qwen2-VL-72B-Instruct（经典版）\n   ⭐ Pro/Qwen/Qwen2-VL-7B-Instruct（经典快速）\n   ⭐ Qwen/Qwen3-VL-8B-Instruct（最新快速版）\n   ⭐ Qwen/Qwen3-VL-32B-Instruct（最新版，需90-120s）\n3. 注意：Qwen3-VL 系列处理时间较长，建议timeout≥90秒\n4. 单图分析请使用 \"PD guijiliudong VLM\" 节点\n\n错误详情: type={error_type}, code={error_code}")
                                else:
                                    raise Exception(f"API返回错误: {error_msg} (type={error_type}, code={error_code})")
                            return result
                        else:
                            raise Exception(f"HTTP错误: {response.status}")
                    
                    else:
                        # 使用requests，设置连接超时和读取超时
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
                            # 设置连接超时10秒，读取超时timeout秒（强制）
                            print(f"⏱️  发送API请求... (连接超时:10s, 读取超时:{timeout}s)")
                            req_start = time_module.time()
                            
                            response = requests.post(
                                url,
                                headers=headers,
                                json=payload,
                                timeout=(10, timeout),  # (connect_timeout, read_timeout) - 强制超时
                                verify=config["verify"],
                                proxies={"http": None, "https": None}
                            )
                            
                            req_time = time_module.time() - req_start
                            print(f"✅ 收到API响应，耗时: {req_time:.2f}秒")
                            
                            response.raise_for_status()
                            result = response.json()
                            
                            if "error" in result:
                                error_msg = result["error"].get("message", "未知API错误")
                                error_type = result["error"].get("type", "unknown")
                                error_code = result["error"].get("code", "unknown")
                                
                                # 特殊处理模型不存在的错误
                                if "model" in error_msg.lower() or "not found" in error_msg.lower():
                                    raise Exception(f"❌ 模型错误: {error_msg}\n\n💡 解决方案:\n1. 该模型可能不支持双图输入或暂时不可用\n2. 本节点支持的双图对比模型（共7个）:\n   ⭐⭐⭐ Qwen/Qwen2.5-VL-32B-Instruct（最推荐，快速）\n   ⭐⭐ Qwen/Qwen2.5-VL-7B-Instruct（最快最便宜）\n   ⭐⭐ Qwen/Qwen2.5-VL-72B-Instruct（高精度）\n   ⭐ Qwen/Qwen2-VL-72B-Instruct（经典版）\n   ⭐ Pro/Qwen/Qwen2-VL-7B-Instruct（经典快速）\n   ⭐ Qwen/Qwen3-VL-8B-Instruct（最新快速版）\n   ⭐ Qwen/Qwen3-VL-32B-Instruct（最新版，需90-120s）\n3. 注意：Qwen3-VL 系列处理时间较长，建议timeout≥90秒\n4. 单图分析请使用 \"PD guijiliudong VLM\" 节点\n\n错误详情: type={error_type}, code={error_code}")
                                else:
                                    raise Exception(f"API返回错误: {error_msg} (type={error_type}, code={error_code})")
                            
                            print(f"✅ API调用成功完成")
                            return result
                            
                        finally:
                            # 恢复环境变量
                            for var, value in env_backup.items():
                                os.environ[var] = value
                                
                except (requests.exceptions.Timeout, Urllib3TimeoutError) as e:
                    last_error = f"⏱️ 请求超时（{timeout}秒）: {str(e)}"
                    print(f"⏱️ 配置 {i+1} 超时: {last_error}")
                    # 超时立即抛出，不再重试
                    raise TimeoutException(f"API请求超时（{timeout}秒）- 已自动中断")
                        
                except requests.exceptions.RequestException as e:
                    last_error = f"网络请求错误: {str(e)}"
                    print(f"配置 {i+1} 失败: {last_error}")
                    # 如果还有更多配置可以尝试，继续下一个配置
                    if i < len(configs) - 1:
                        continue
                    # 如果所有配置都尝试过了，且还有重试次数，进入下一轮重试
                    elif retry_count < max_retries - 1:
                        import time
                        time.sleep(1)
                        break  # 跳出内层循环，进入下一轮重试
                    else:
                        # 最后一次重试失败，跳出内层循环
                        break
                        
                except Exception as e:
                    last_error = f"未知错误: {str(e)}"
                    print(f"配置 {i+1} 失败: {last_error}")
                    # 如果还有更多配置可以尝试，继续下一个配置
                    if i < len(configs) - 1:
                        continue
                    # 如果所有配置都尝试过了，且还有重试次数，进入下一轮重试
                    elif retry_count < max_retries - 1:
                        import time
                        time.sleep(1)
                        break  # 跳出内层循环，进入下一轮重试
                    else:
                        # 最后一次重试失败，跳出内层循环
                        break
        
        # 所有重试都失败
        error_solutions = f"""
网络连接失败（已重试{max_retries}次），可能的解决方案：
1. 检查网络连接是否正常
2. 尝试使用VPN或更换网络环境
3. 检查防火墙设置
4. 确认API密钥是否正确
5. 稍后重试（可能是服务器临时问题）
6. 联系硅基流动技术支持

最后错误: {last_error if last_error else '未知错误'}
        """.strip()
        
        raise Exception(error_solutions)
    
    def analyze_images(self, image1, image2, model, prompt, api_key="", detail="high", temperature=0.7, max_tokens=2048, timeout=40):
        """分析两张图片"""
        
        start_time = time_module.time()
        
        # 检测是否为批量输入
        batch_size1 = image1.shape[0] if image1.dim() == 4 else 1
        batch_size2 = image2.shape[0] if image2.dim() == 4 else 1
        is_batch = batch_size1 > 1 or batch_size2 > 1
        
        # 动态调整超时时间
        max_total_time = timeout
        
        # 对于特定模型和批量情况，自动增加超时时间
        if "Qwen3-VL" in model:
            # Qwen3-VL 处理时间较长
            if "32B" in model:
                # 32B 版本更慢
                if is_batch:
                    max_total_time = max(timeout, 120)  # 批量至少120秒
                    print(f"⚠️  检测到批量输入，Qwen3-VL-32B 模型自动延长超时至 {max_total_time}秒")
                else:
                    max_total_time = max(timeout, 90)  # 单次至少90秒
                    if timeout < 90:
                        print(f"⚠️  Qwen3-VL-32B 模型处理时间较长，建议设置timeout≥90秒")
            elif "8B" in model:
                # 8B 版本相对快一些
                if is_batch:
                    max_total_time = max(timeout, 90)  # 批量至少90秒
                    print(f"ℹ️  检测到批量输入，Qwen3-VL-8B 模型自动延长超时至 {max_total_time}秒")
                else:
                    max_total_time = max(timeout, 60)  # 单次至少60秒
                    if timeout < 60:
                        print(f"ℹ️  Qwen3-VL-8B 模型建议设置timeout≥60秒")
        elif is_batch:
            # 其他模型的批量处理也适当延长
            max_total_time = max(timeout, 60)
            print(f"ℹ️  检测到批量输入 (batch_size={max(batch_size1, batch_size2)})，自动延长超时至 {max_total_time}秒")
        
        print("="*60)
        print(f"⏱️  [超时设置] 最大等待时间: {max_total_time}秒")
        print(f"📋 [视觉模型] {model}")
        print(f"🖼️  [功能] 双图对比分析")
        if is_batch:
            print(f"📦 [批量模式] Batch Size: {max(batch_size1, batch_size2)}")
        print("="*60)
        
        try:
            # 优先使用传入的API密钥，如果没有则从配置文件加载
            if not api_key or api_key.strip() == "":
                api_key = self.load_config()
            else:
                api_key = api_key.strip()
            
            # 转换图像为base64 - 使用超时保护
            print("🖼️  正在处理图片...")
            
            def process_images():
                return (self.tensor_to_base64(image1), self.tensor_to_base64(image2))
            
            try:
                base64_image1, base64_image2 = timeout_handler(process_images, min(15, max_total_time))
                print(f"✅ 图片处理完成")
            except TimeoutException as e:
                raise Exception(f"图片处理超时（超过15秒）- {str(e)}")
            
            # 检查是否已经超时
            elapsed = time_module.time() - start_time
            if elapsed > max_total_time:
                raise TimeoutException(f"⏱️ 总时间超过{max_total_time}秒限制 - 已中断")
            
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
            
            # 调用API - 使用严格超时保护
            print("🌐 正在调用硅基流动API...")
            remaining_time = max_total_time - (time_module.time() - start_time)
            if remaining_time <= 5:
                raise TimeoutException(f"⏱️ 剩余时间不足（{remaining_time:.1f}秒）- 已中断")
            
            # API调用的超时时间
            api_timeout = max(int(remaining_time) - 2, 10)  # 留2秒余量
            print(f"⏱️  API超时设置: {api_timeout}秒")
            
            def call_api():
                return self.call_siliconflow_api(
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=1,  # 减少重试次数，避免超时
                    timeout=api_timeout
                )
            
            try:
                response = timeout_handler(call_api, remaining_time)
                print(f"✅ API调用成功")
                
                # 显示Token使用和价格信息
                if "usage" in response:
                    usage = response["usage"]
                    p_tokens = usage.get('prompt_tokens', 0)
                    c_tokens = usage.get('completion_tokens', 0)
                    t_tokens = usage.get('total_tokens', 0)
                    
                    # 计算成本
                    i_cost, o_cost, total_cost = self.calculate_cost(model, p_tokens, c_tokens)
                    
                    print(f"📊 Token使用: 输入 {p_tokens:,} + 输出 {c_tokens:,} = 总计 {t_tokens:,}")
                    print(f"💰 费用估算: {self.format_cost(total_cost)} 元 (输入 {self.format_cost(i_cost)} + 输出 {self.format_cost(o_cost)})")
                
            except TimeoutException as e:
                raise Exception(f"⏱️ API调用超时（{remaining_time:.0f}秒）- 已自动中断，请尝试：\n1. 增加超时时间\n2. 使用更快的模型（如8B版本）\n3. 检查网络连接")
            
            # 计算耗时
            elapsed_time = time_module.time() - start_time
            
            # 提取分析结果
            if "choices" in response and len(response["choices"]) > 0:
                analysis_result = response["choices"][0]["message"]["content"]
                
                # 构建info信息
                info_lines = []
                info_lines.append(f"{'='*40}")
                info_lines.append(f"🖼️  双图对比分析信息")
                info_lines.append(f"{'='*40}")
                info_lines.append(f"")
                info_lines.append(f"📋 模型: {model}")
                info_lines.append(f"⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                info_lines.append(f"⏱️  耗时: {elapsed_time:.2f} 秒")
                info_lines.append(f"")
                
                # Token统计和价格
                if "usage" in response:
                    usage = response["usage"]
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    
                    # 计算成本
                    input_cost, output_cost, total_cost = self.calculate_cost(
                        model, prompt_tokens, completion_tokens
                    )
                    
                    info_lines.append(f"📊 Token使用:")
                    info_lines.append(f"   • 输入: {prompt_tokens:,} tokens")
                    info_lines.append(f"   • 输出: {completion_tokens:,} tokens")
                    info_lines.append(f"   • 总计: {total_tokens:,} tokens")
                    info_lines.append(f"")
                    
                    # 突出显示总价格
                    info_lines.append(f"💰 本次调用费用: {self.format_cost(total_cost)} 元")
                    info_lines.append(f"   (输入: {self.format_cost(input_cost)} + 输出: {self.format_cost(output_cost)})")
                    
                    # 添加定价信息
                    pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["_default"])
                    is_default = model not in self.MODEL_PRICING
                    info_lines.append(f"   模型定价: 输入¥{pricing['input']}/M tokens, 输出¥{pricing['output']}/M tokens{'  (估算)' if is_default else ''}")
                    info_lines.append(f"")
                
                # 输出统计
                output_len = len(analysis_result)
                info_lines.append(f"📝 输出长度: {output_len:,} 字符")
                info_lines.append(f"✅ 状态: {response['choices'][0].get('finish_reason', 'unknown')}")
                info_lines.append(f"{'='*40}")
                
                info_text = "\n".join(info_lines)
                
                return (analysis_result, model, info_text)
            else:
                error_msg = f"API返回格式异常: {json.dumps(response, indent=2, ensure_ascii=False)}"
                error_info = f"❌ 错误\n📋 模型: {model}\n⏱️  耗时: {elapsed_time:.2f}秒\n🚫 {error_msg}"
                return (error_msg, model, error_info)
                
        except TimeoutException as e:
            elapsed_time = time_module.time() - start_time if 'start_time' in locals() else 0
            error_msg = f"⏱️ 操作超时: {str(e)}"
            error_info = f"⏱️ 超时中断\n📋 模型: {model}\n⏱️  已等待: {elapsed_time:.2f}秒\n🚫 {str(e)}\n\n💡 建议：增加timeout参数值或使用更快的模型"
            print(f"⏱️ 超时: {error_msg}")
            return (error_msg, model, error_info)
            
        except Exception as e:
            elapsed_time = time_module.time() - start_time if 'start_time' in locals() else 0
            error_msg = f"❌ 分析过程中出现错误: {str(e)}"
            error_info = f"❌ 错误\n📋 模型: {model}\n⏱️  耗时: {elapsed_time:.2f}秒\n🚫 {str(e)}"
            print(f"❌ 错误: {error_msg}")
            return (error_msg, model, error_info)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "PD_guijiliudong_vision_v1": PD_guijiliudong_vision_v1
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PD_guijiliudong_vision_v1": "PD_guijiliudong VLM 2image"
}

# 节点描述
NODE_DESCRIPTIONS = {
    "PD_guijiliudong_vision_v1": "使用硅基流动API智能对比分析两张图片并生成详细的文字描述"
}

# 导出节点类
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'NODE_DESCRIPTIONS'] 