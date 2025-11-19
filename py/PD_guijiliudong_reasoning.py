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

class PD_guijiliudong_reasoning:
    """
    硅基流动推理模型节点
    专门用于支持深度思维链的推理模型（DeepSeek-R1、QwQ等）
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # 推理模型列表
        reasoning_models = [
            # DeepSeek-R1系列（最强推理）✅ 推荐
            "deepseek-ai/DeepSeek-R1",  # ✅ 稳定可用
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # ✅ 稳定可用
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  # ✅ 稳定可用
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  # ✅ 稳定可用，速度快
            
            # Qwen推理系列
            "Qwen/QwQ-32B",  # ✅ 稳定可用
        ]
        
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "default": "你是一个强大的AI推理助手，擅长深度思考和逻辑推理。请一步步分析问题，展示你的推理过程。",
                    "multiline": True
                }),
            },
            "optional": {
                "user_prompt": ("STRING", {
                    "default": "请解释一下量子纠缠的原理。",
                    "multiline": True
                }),
                "model": (reasoning_models, {"default": "deepseek-ai/DeepSeek-R1"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "retry_count": ("INT", {"default": 3, "min": 1, "max": 5}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("reasoning", "output_text", "model_name", "info")
    FUNCTION = "generate_reasoning"
    CATEGORY = "PD_Nodes/Reasoning"
    
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
            "deepseek-ai/DeepSeek-R1": {"input": 0.55, "output": 2.19},
            "Pro/deepseek-ai/DeepSeek-R1": {"input": 1.0, "output": 4.0},
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {"input": 0.14, "output": 0.28},
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {"input": 0.07, "output": 0.14},
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {"input": 0.035, "output": 0.07},
            "Qwen/QwQ-32B": {"input": 0.135, "output": 0.135},
            "Qwen/Qwen3-30B-A3B-Thinking-2507": {"input": 0.1, "output": 0.1},
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
    
    def call_api(self, api_key, model, messages, temperature, max_tokens, retry_count=3):
        """调用硅基流动API（支持重试）"""
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
            "stream": False
        }
        
        if not requests:
            raise Exception("requests库未安装")
        
        # 根据模型类型设置超时时间
        timeout = 300  # 默认5分钟
        if "235B" in model or "80B" in model:
            timeout = 600  # 大模型10分钟
        
        # 重试机制
        last_error = None
        for attempt in range(retry_count):
            try:
                if attempt > 0:
                    wait_time = attempt * 2  # 递增等待时间
                    print(f"⏳ 第 {attempt + 1} 次重试，等待 {wait_time} 秒...")
                    time_module.sleep(wait_time)
                
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                
                if response.status_code == 200:
                    return response.json()
                
                # 处理非200状态码
                error_msg = f"HTTP {response.status_code}"
                error_detail = ""
                try:
                    error_json = response.json()
                    error_detail = error_json.get("error", {}).get("message", "")
                    if not error_detail:
                        error_detail = str(error_json.get("error", ""))
                    if error_detail:
                        error_msg += f": {error_detail}"
                except:
                    error_msg += f": {response.text[:200]}"
                
                # 某些错误不需要重试
                if response.status_code in [400, 401, 403, 404]:
                    raise Exception(f"{error_msg}\n💡 提示: 此模型可能不可用或需要特殊权限")
                
                last_error = Exception(error_msg)
                
            except requests.exceptions.Timeout as e:
                last_error = Exception(f"请求超时（>{timeout}秒）\n💡 提示: 模型响应较慢，建议稍后重试或选择其他模型")
            except requests.exceptions.ConnectionError as e:
                last_error = Exception(f"网络连接失败: {str(e)}\n💡 提示: 请检查网络连接或代理设置")
            except requests.exceptions.RequestException as e:
                last_error = Exception(f"请求异常: {str(e)}")
            except Exception as e:
                if "timeout" in str(e).lower():
                    last_error = Exception(f"请求超时: {str(e)}\n💡 提示: 网络不稳定或模型负载过高")
                else:
                    last_error = e
        
        # 所有重试都失败
        if last_error:
            raise last_error
        
        raise Exception("API调用失败，未知错误")
    
    def generate_reasoning(self, system_prompt, user_prompt="", model="", api_key="", temperature=0.6, max_tokens=4096, retry_count=3, debug_mode=False):
        """生成推理内容"""
        
        start_time = time_module.time()
        
        try:
            # 验证输入
            if not user_prompt or not user_prompt.strip():
                raise Exception("user_prompt不能为空")
            
            # 设置默认模型
            if not model or model.strip() == "":
                model = "deepseek-ai/DeepSeek-R1"
            
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
            
            # 模型提示
            model_tips = {
                "Pro/": "⚠️  此模型需要Pro权限",
                "80B": "⚠️  大模型，响应较慢（超时10分钟）",
                "235B": "⚠️  超大模型，响应很慢（超时10分钟）",
            }
            for key, tip in model_tips.items():
                if key in model:
                    print(tip)
                    break
            
            # 调用API
            print(f"🧠 调用推理模型: {model}")
            print(f"🔄 重试次数: 最多 {retry_count} 次")
            print(f"⏳ 等待响应中...")
            response = self.call_api(api_key, model, messages, temperature, max_tokens, retry_count)
            
            # 计算耗时
            elapsed_time = time_module.time() - start_time
            print(f"✅ API调用成功，耗时: {elapsed_time:.2f}秒")
            
            # 打印原始响应结构用于调试
            if debug_mode:
                print(f"\n{'='*50}")
                print(f"🔍 调试模式 - API完整响应:")
                print(f"{'='*50}")
                print(json.dumps(response, indent=2, ensure_ascii=False))
                print(f"{'='*50}\n")
            
            print(f"📦 API响应结构: {list(response.keys())}")
            
            # 提取结果
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                message = choice["message"]
                
                print(f"📝 Message字段: {list(message.keys())}")
                
                # 尝试多种可能的思维链字段名
                reasoning_chain = ""
                possible_fields = ["reasoning_content", "reasoning", "thinking", "thought_process"]
                for field in possible_fields:
                    if field in message and message[field]:
                        reasoning_chain = message[field]
                        print(f"✅ 找到思维链字段: {field}")
                        break
                
                generated_text = message.get("content", "")
                
                # 打印思维链信息
                if reasoning_chain:
                    reasoning_len = len(reasoning_chain)
                    print(f"🧠 思维链长度: {reasoning_len:,} 字符")
                    print(f"🧠 思维链前200字符预览:\n{reasoning_chain[:200]}...")
                else:
                    print(f"⚠️  未找到思维链内容，可能的原因：")
                    print(f"   1. 模型不支持思维链输出")
                    print(f"   2. API字段名不匹配")
                    print(f"   3. 思维链内容为空")
                
                # 构建info信息
                info_lines = []
                info_lines.append(f"{'='*40}")
                info_lines.append(f"🤖 推理模型调用信息")
                info_lines.append(f"{'='*40}")
                info_lines.append(f"")
                info_lines.append(f"📋 模型: {model}")
                info_lines.append(f"⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                info_lines.append(f"⏱️  耗时: {elapsed_time:.2f} 秒")
                info_lines.append(f"")
                
                # 思维链状态（详细信息）
                if reasoning_chain:
                    reasoning_len = len(reasoning_chain)
                    reasoning_lines = reasoning_chain.count('\n') + 1
                    info_lines.append(f"🧠 思维链: ✅ 已生成")
                    info_lines.append(f"   • 字符数: {reasoning_len:,}")
                    info_lines.append(f"   • 行数: {reasoning_lines:,}")
                    # 查找思维链字段名
                    found_field = ""
                    for field in ["reasoning_content", "reasoning", "thinking", "thought_process"]:
                        if field in message and message[field]:
                            found_field = field
                            break
                    if found_field:
                        info_lines.append(f"   • 字段名: {found_field}")
                else:
                    info_lines.append(f"🧠 思维链: ❌ 未生成")
                    info_lines.append(f"   • 可能原因: API不返回该字段")
                    info_lines.append(f"   • 建议: 开启debug_mode查看完整响应")
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
                info_lines.append(f"")
                
                # 添加使用提示
                if reasoning_chain:
                    info_lines.append(f"💡 提示:")
                    info_lines.append(f"   • reasoning 输出包含完整的思维链")
                    info_lines.append(f"   • 可连接到文本显示节点查看详细推理过程")
                    info_lines.append(f"   • output_text 包含最终答案")
                else:
                    info_lines.append(f"💡 提示:")
                    info_lines.append(f"   • 开启 debug_mode 可查看完整API响应")
                    info_lines.append(f"   • 某些模型可能不支持单独的思维链输出")
                    info_lines.append(f"   • 思维链可能包含在 output_text 中")
                
                info_lines.append(f"{'='*40}")
                
                info_text = "\n".join(info_lines)
                
                # 最终打印总结
                print(f"\n🎉 推理完成!")
                print(f"   • 思维链: {'✅ ' + str(len(reasoning_chain)) + ' 字符' if reasoning_chain else '❌ 未生成'}")
                print(f"   • 输出文本: {len(generated_text)} 字符")
                print(f"   • 总耗时: {elapsed_time:.2f}秒\n")
                
                return (reasoning_chain, generated_text, model, info_text)
            else:
                error_msg = "API返回格式异常"
                error_info = f"❌ 错误\n📋 模型: {model}\n🚫 {error_msg}"
                return ("", error_msg, model, error_info)
                
        except Exception as e:
            elapsed_time = time_module.time() - start_time
            error_msg = f"推理过程出现错误: {str(e)}"
            
            # 构建详细的错误信息
            error_info_lines = []
            error_info_lines.append(f"{'='*40}")
            error_info_lines.append(f"❌ 调用失败")
            error_info_lines.append(f"{'='*40}")
            error_info_lines.append(f"")
            error_info_lines.append(f"📋 模型: {model}")
            error_info_lines.append(f"⏱️  耗时: {elapsed_time:.2f}秒")
            error_info_lines.append(f"")
            error_info_lines.append(f"🚫 错误信息:")
            error_info_lines.append(f"{str(e)}")
            error_info_lines.append(f"")
            
            # 根据错误类型提供建议
            error_str = str(e).lower()
            if "timeout" in error_str or "超时" in error_str:
                error_info_lines.append(f"💡 解决建议:")
                error_info_lines.append(f"   1. 增加 retry_count（当前: {retry_count}）")
                error_info_lines.append(f"   2. 选择更小更快的模型:")
                error_info_lines.append(f"      • deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
                error_info_lines.append(f"      • Qwen/QwQ-32B")
                error_info_lines.append(f"   3. 检查网络连接是否稳定")
                error_info_lines.append(f"   4. 稍后重试（可能是服务器负载过高）")
            elif "403" in error_str or "401" in error_str or "权限" in error_str:
                error_info_lines.append(f"💡 解决建议:")
                error_info_lines.append(f"   1. 检查API密钥是否正确")
                error_info_lines.append(f"   2. 此模型可能需要Pro权限")
                error_info_lines.append(f"   3. 尝试使用标准模型:")
                error_info_lines.append(f"      • deepseek-ai/DeepSeek-R1")
                error_info_lines.append(f"      • deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
            elif "404" in error_str or "不可用" in error_str:
                error_info_lines.append(f"💡 解决建议:")
                error_info_lines.append(f"   1. 此模型可能已下线或不存在")
                error_info_lines.append(f"   2. 推荐使用以下稳定模型:")
                error_info_lines.append(f"      ✅ deepseek-ai/DeepSeek-R1")
                error_info_lines.append(f"      ✅ deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
                error_info_lines.append(f"      ✅ deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
                error_info_lines.append(f"      ✅ Qwen/QwQ-32B")
                error_info_lines.append(f"   3. 开启 debug_mode 查看详细信息")
            elif "network" in error_str or "网络" in error_str or "connection" in error_str:
                error_info_lines.append(f"💡 解决建议:")
                error_info_lines.append(f"   1. 检查网络连接")
                error_info_lines.append(f"   2. 检查代理设置")
                error_info_lines.append(f"   3. 尝试使用其他网络环境")
                error_info_lines.append(f"   4. 增加 retry_count 参数")
            else:
                error_info_lines.append(f"💡 解决建议:")
                error_info_lines.append(f"   1. 开启 debug_mode 查看详细信息")
                error_info_lines.append(f"   2. 检查API密钥配置")
                error_info_lines.append(f"   3. 尝试使用推荐的稳定模型")
                error_info_lines.append(f"   4. 查看控制台完整错误信息")
            
            error_info_lines.append(f"")
            error_info_lines.append(f"{'='*40}")
            
            error_info = "\n".join(error_info_lines)
            
            print(f"\n❌ 错误: {error_msg}")
            print(error_info)
            
            return ("", error_msg, model, error_info)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "PD_guijiliudong_reasoning": PD_guijiliudong_reasoning
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PD_guijiliudong_reasoning": "PD 推理模型 (Reasoning)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

