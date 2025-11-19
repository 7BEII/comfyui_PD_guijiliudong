# ComfyUI PD Guijiliudong Nodes (硅基流动 VLM 节点)

这是一个为 ComfyUI 开发的自定义节点套件，专门用于集成硅基流动 (SiliconFlow) 的多模态大模型 (VLM)、对话模型和推理模型。

通过本插件，您可以轻松在 ComfyUI 中使用 Qwen2.5-VL, GLM-4.5V, DeepSeek-R1 等强大的云端模型进行图像分析、多图对比、智能对话和深度推理。

## ✨ 主要功能

*   **单图分析 (VLM)**: 支持对单张图片进行详细描述、反推提示词、内容分析。
*   **双图对比 (VLM 2image)**: 专门优化的双图输入节点，支持对比两张图片的差异、相似度或进行联合分析。
*   **智能对话 (Chat)**: 支持多种大语言模型进行通用对话。
*   **深度推理 (Reasoning)**: 集成 DeepSeek-R1 等推理模型，支持思维链 (Chain of Thought) 展示。

## 🛠️ 安装说明

1.  进入您的 ComfyUI `custom_nodes` 目录：
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  克隆本项目：
    ```bash
    git clone https://github.com/YOUR_USERNAME/comfyui_PD_guijiliudong.git
    ```
3.  安装依赖：
    ```bash
    cd comfyui_PD_guijiliudong
    pip install -r requirements.txt
    ```
4.  **配置 API 密钥**：
    *   在 `py` 目录下找到 `config.json.example` 文件。
    *   将其重命名为 `config.json`。
    *   用文本编辑器打开 `config.json`，填入您的硅基流动 API Key：
        ```json
        {
          "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
          "base_url": "https://api.siliconflow.cn/v1"
        }
        ```
    *   如果没有 API Key，请前往 [硅基流动官网](https://siliconflow.cn) 注册申请。
  
 -  邀请码：https://cloud.siliconflow.cn/i/8C2qIGwR 
 -  登录时候输入：8C2qIGwR  

    ---
      git remote add origin https://github.com/7BEII/comfyui_PD_guijiliudong.git 

## 📖 节点介绍与使用

### 1. PD guijiliudong VLM 2image (双图对比)
**推荐用于两张图片的对比分析任务。**

*   **支持模型**:
    *   `Qwen/Qwen2.5-VL-32B-Instruct` (⭐⭐⭐ 强烈推荐，平衡速度与质量)
    *   `Qwen/Qwen2.5-VL-7B-Instruct` (⭐⭐ 快速版)
    *   `Qwen/Qwen2.5-VL-72B-Instruct` (⭐⭐ 专业版，高精度)
    *   `zai-org/GLM-4.5V` (GLM 视觉版)
    *   `deepseek-ai/deepseek-vl2`
*   **使用建议**:
    *   **日常对比**: 推荐使用 Qwen2.5-VL-32B，Timeout 设置 40-60秒。
    *   **批量处理**: 推荐 Qwen2.5-VL-7B，Timeout 设置 30-40秒。
    *   **专业分析**: 推荐 Qwen2.5-VL-72B，Timeout 设置 90-120秒。
*   **注意**: 部分旧模型（如 Qwen3-VL系列）因多图支持不稳定已从该节点移除，请使用单图节点。

### 2. PD guijiliudong VLM (单图分析)
**适用于单张图片的描述和分析，支持最全的模型列表。**

*   支持所有上述模型以及 Qwen3-VL 系列、GLM-4.5-Air 等。

### 3. PD guijiliudong Reasoning (推理)
**专注于逻辑推理和复杂问题求解。**

*   支持 `DeepSeek-R1`, `QwQ-32B` 等推理模型。
*   输出包含思维链过程，适合需要展示推理步骤的场景。

### 4. PD guijiliudong Chat (对话)
**通用的 LLM 对话节点。**

*   支持 Qwen2.5 全系列, DeepSeek-V3, GLM-4 等主流文本模型。

## 📅 更新日志 (2025-11-18)

**VLM 2image 节点重要更新：**
1.  **模型优化**: 移除了不支持双图输入或不稳定的模型（如 Qwen3-VL, DeepSeek-V3 文本版等），解决了调用超时和报错问题。
2.  **新增推荐**: 明确标记了推荐模型（⭐⭐⭐），帮助用户快速选择。
3.  **错误提示**: 优化了错误处理，当模型不可用时提供更清晰的指引。

## 💰 成本参考 (硅基流动)

*   **Qwen2.5-VL-7B**: ¥0.2 / M tokens (最便宜)
*   **Qwen2.5-VL-32B**: ¥0.5 / M tokens (推荐)
*   **Qwen2.5-VL-72B**: ¥0.8 / M tokens
*   *(价格仅供参考，请以官网实时定价为准)*

## 🔗 参考链接

*   [硅基流动官网](https://siliconflow.cn)
*   [API 文档](https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions)
