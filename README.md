# Medical VLM Fine-tuning & Inference Assistant

这是一个基于 [Unsloth](https://github.com/unslothai/unsloth) 和 [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) 的全流程医疗视觉大模型项目，涵盖了 **SFT (监督微调)** 和 **GRPO (强化学习)** 两个核心阶段。
本项目演示了如何通过 LoRA 高效微调多模态大模型，使其具备专业的医疗影像诊断能力，并引入了类似 DeepSeek-R1 的强化学习技术来进一步提升模型的推理逻辑（Chain-of-Thought），最终提供了一个支持双模型切换的 Streamlit 可视化对话界面。

## 🚀 项目亮点

*   **⚡ 高效微调 (Unsloth)**：利用 Unsloth 加速 Qwen3-VL-8B 的 LoRA 微调，相比传统方法训练速度提升 2x，显存占用降低 60%。
*   **🏥 SFT 监督微调**：针对医疗影像（如 X 光、CT 等）进行指令微调，使模型学会放射科医生的专业术语和诊断格式。
*   **🧠 GRPO 强化学习**：引入 **Group Relative Policy Optimization (GRPO)** 算法，通过奖励函数（格式奖励、推理步骤奖励、准确率奖励）引导模型生成更详细、更有条理的思维链（Reasoning）。
*   **🖥️ 交互式 Web UI**：基于 Streamlit 构建的现代化界面，支持 **实时切换 SFT 模型与 RL 模型**，直观对比强化学习带来的能力提升。

## 📂 项目结构

```
.
├── train.py            # 微调脚本 (SFT)
├── train_grpo.py       # 强化学习脚本 (GRPO)
├── app.py              # Streamlit 可视化部署应用
├── requirements.txt    # 项目依赖文件
├── README.md           # 项目说明文档
├── data/               # 训练数据集目录
└── lora_model/         # (自动生成) 微调后的 LoRA 权重
```

## 🛠️ 准备工作

### 1. 环境安装
建议使用 Conda 创建虚拟环境：
```bash
conda create -n vlm python=3.10
conda activate vlm
pip install -r requirements.txt
```
*注意：Unsloth 的安装可能需要特定的 CUDA 版本，请参考 [Unsloth 官方文档](https://github.com/unslothai/unsloth) 进行适配。*

### 2. 下载基础模型
本项目使用 [Qwen3-VL-8B-Instruct-bnb-4bit](https://www.modelscope.cn/models/unsloth/Qwen3-VL-8B-Instruct-bnb-4bit) 作为基座模型。你可以使用 modelscope CLI 进行下载：

```bash
# 安装 modelscope
pip install modelscope

# 下载模型到本地 models 目录
modelscope download --model unsloth/Qwen3-VL-8B-Instruct-bnb-4bit --local_dir models/unsloth/Qwen3-VL-8B-Instruct-bnb-4bit
```

### 3. 准备数据集
本项目使用 [Radiology-mini](https://huggingface.co/datasets/open-data/Radiology-mini) 数据集进行演示。请前往 Hugging Face 下载并将数据解压到 `data/` 目录中。

## 🏃‍♂️ 快速开始

### 1. 第一阶段：监督微调 (SFT)
首先运行 `train.py` 进行基础能力的对齐，让模型学会医疗诊断的格式与术语：

```bash
python train.py
```
*   **输入**：`data/` 目录下的图文对。
*   **输出**：SFT 后的 LoRA 权重，自动保存至 `lora_model/`。

### 2. 第二阶段：强化学习 (GRPO)
在 SFT 的基础上，使用 `train_grpo.py` 引入强化学习，进一步提升模型的推理能力（Reasoning）：

```bash
python train_grpo.py
```
*   **前提**：必须先完成第一阶段 SFT 训练（脚本会自动加载 `lora_model/`）。
*   **核心机制**：通过格式奖励、步骤奖励和准确率奖励，引导模型生成思维链（Chain-of-Thought）。
*   **输出**：RL 优化后的权重，自动保存至 `grpo_model/`。

### 3. 启动 Web 应用
训练完成后，使用 Streamlit 启动可视化界面：
```bash
streamlit run app.py
```
访问终端显示的 URL即可使用。

## 🧪 效果展示
![alt text](docs/image.png)
![alt text](docs/image-4.png)


