# train_grpo.py - Improved Medical VLM Reinforcement Learning (GRPO) Script
# 
# 🏥 医疗视觉大模型强化学习脚本 (GRPO) 
#
#
# 功能：
# 1. 加载 SFT 后的模型或基座模型
# 2. 定义多维度奖励函数：格式、长度、步骤、准确率
# 3. 执行 GRPO 训练
# 4. 保存最终模型

import sys
import os
import re
import torch
import shutil
from unsloth import FastVisionModel, is_bf16_supported
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from transformers import AutoTokenizer

# =================================================================
# 配置区域
# =================================================================
# 尝试查找的模型路径列表 (按优先级)
MODEL_CANDIDATES = [
    "lora_model",  # 优先加载当前目录下的 SFT 模型
    "/root/autodl-tmp/models/unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    "/root/autodl-tmp/models/unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit", # 自动下载
]

DATASET_PATH = "./data"
OUTPUT_DIR = "outputs_grpo" # 恢复为原来的 outputs_grpo
MAX_PROMPT_LENGTH = 1024
MAX_COMPLETION_LENGTH = 1024  # 从 512 增加到 1024，防止 CoT 被截断

def get_model_path():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path) or path.startswith("unsloth/"):
            return path
    return MODEL_CANDIDATES[-1] # Fallback to download

def main():
    print("Starting Improved Medical VLM GRPO Training...")
    
    # 1. 模型加载 - 尝试多个路径
    model = None
    tokenizer = None
    
    for model_name in MODEL_CANDIDATES:
        # 如果是本地路径且不存在，跳过
        if not model_name.startswith("unsloth/") and not os.path.exists(model_name):
            continue
            
        print(f"📦 Attempting to load model from: {model_name}")
        print("⏳ Loading model weights... (This may take 1-2 minutes)")
        try:
            # 检查是否为本地路径，如果是则强制 local_files_only 以避免网络卡顿
            is_local = os.path.exists(model_name)
            
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=model_name,
                load_in_4bit=True,
                device_map="auto",
                use_gradient_checkpointing="unsloth",
                local_files_only=is_local, # 恢复此参数以加快本地加载
            )
            print(f"✅ Successfully loaded: {model_name}")
            break
        except Exception as e:
            print(f"⚠️ Failed to load {model_name}: {e}")
            continue
    
    if model is None:
        print("❌ All model candidates failed to load. Exiting.")
        return

    # 2. LoRA 配置
    if hasattr(model, "peft_config") and len(model.peft_config) > 0:
        print(" Model already has LoRA adapters. Enabling training mode...")
        FastVisionModel.for_training(model)
    else:
        print("🆕 Adding new LoRA adapters...")
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False, # 通常锁住 Vision Tower
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_rslora=False,
        )

    # 3. 数据准备
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path '{DATASET_PATH}' not found!")
        return
        
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset(DATASET_PATH, split="train")
    
    # 系统提示词：强调 CoT (Chain of Thought)
    SYSTEM_PROMPT = """You are a professional radiologist. Analyze the given medical image.
Strictly follow this format for your output:

<reasoning>
Write your detailed observation, reasoning logic, and analysis process here.
</reasoning>
<answer>
Write your final diagnostic conclusion here.
</answer>
"""

    def format_data(sample):
        # 确保图片存在
        if 'image' not in sample:
            return None
            
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample['image']},
                    {"type": "text", "text": "Analyze this image."}
                ]
            }
        ]
        return {
            "prompt": messages,
            "ground_truth": sample['caption']
        }

    # 处理数据集
    original_len = len(dataset)
    # 1. 先过滤掉没有图片的样本
    dataset = dataset.filter(lambda x: x.get('image') is not None)
    
    # 2. 格式化
    dataset = dataset.map(format_data, remove_columns=dataset.column_names, num_proc=4)
    print(f"✅ Dataset loaded. Samples: {len(dataset)} (Original: {original_len})")

    # 4. 奖励函数定义
    
    # (A) 格式奖励：严格检查 XML 标签
    def xml_format_reward(completions, **kwargs):
        rewards = []
        pattern = r"<reasoning>[\s\S]*?</reasoning>\s*<answer>[\s\S]*?</answer>"
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            match = re.search(pattern, text)
            rewards.append(1.0 if match else 0.0)
        return rewards

    # (B) 长度奖励：鼓励详细推理
    def length_reward(completions, **kwargs):
        rewards = []
        target_len = 200 # 期望的推理长度字符数
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            reasoning = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
            if reasoning:
                content = reasoning.group(1).strip()
                # 使用高斯函数形式的软奖励，在 target_len 附近最高
                # 或者简单的非线性奖励
                l = len(content)
                if l < 50: rewards.append(-0.5) # 太短
                elif l > 500: rewards.append(-0.2) # 太长可能啰嗦
                else: rewards.append(0.5)
            else:
                rewards.append(0.0)
        return rewards

    # (C) 准确率奖励：基于关键词覆盖
    def accuracy_reward(completions, ground_truth, **kwargs):
        rewards = []
        for completion, truth in zip(completions, ground_truth):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            
            # 提取预测答案
            ans_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            if ans_match:
                pred = ans_match.group(1).strip().lower()
            else:
                # 降级策略：如果没有标签，取最后一部分
                pred = text.split("\n")[-1].strip().lower()
            
            truth = truth.lower()
            
            # 简单的词袋重叠计算
            def get_tokens(s):
                return set(re.findall(r"\w+", s)) - {"the", "a", "an", "is", "of", "in", "and", "to"}
                
            pred_tokens = get_tokens(pred)
            truth_tokens = get_tokens(truth)
            
            if not truth_tokens:
                rewards.append(0.5) # 防止除零
                continue
                
            overlap = len(pred_tokens & truth_tokens)
            recall = overlap / len(truth_tokens)
            
            # 阶梯式奖励
            if recall > 0.8: rewards.append(2.0)
            elif recall > 0.5: rewards.append(1.0)
            elif recall > 0.2: rewards.append(0.5)
            else: rewards.append(0.0)
            
        return rewards

    # 5. 训练参数
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="grpo_medical_vlm",
        learning_rate=2e-6,           # 保守的学习率
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=1,
        per_device_train_batch_size=1, # 显存受限时设为 1
        gradient_accumulation_steps=8, # 增加累积步数以模拟大 batch
        num_generations=4,            # Group Size (G)
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=10,                 # 训练步数
        save_steps=25,
        save_total_limit=2,
        report_to="none",             # 关闭 wandb 除非配置了
        use_vllm=False,               # 设为 False 保证兼容性
        bf16=is_bf16_supported(),
        beta=0.01,                    # KL 惩罚系数
    )

    # 6. 初始化 Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[xml_format_reward, length_reward, accuracy_reward],
        args=training_args,
        train_dataset=dataset,
    )

    # 7. 开始训练
    print(" Starting training...")
    trainer.train()
    
    # 8. 保存结果
    final_output_dir = "grpo_model"
    print(f"Saving final model to {final_output_dir}...")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
