import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, concatenate_datasets
import os

# ========== 1. 目录配置 ==========
model_dir = "../FlagAlphaLlama3-Chinese-8B-Instruct"  # 预训练模型路径
data_path1 = "../data/train.json"  # 本地 JSON 数据集
data_path_eval = "../data/eval.json"  # 本地评估 JSON 数据集
output_dir = "../models/lora_finetuned"  # 保存微调后的模型

# ========== 2. 加载 Tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token  # 适配 LLaMA，避免 Padding 报错

# ========== 3. 读取数据 ==========
# 加载本地 JSON 训练数据集
dataset1 = load_dataset("json", data_files={"train": data_path1})

# 加载 Hugging Face 上的 CSV 数据集
dataset2 = load_dataset("soniawmeyer/travel-conversations-finetuning")  # 直接加载数据集

# 合并两个训练数据集
train_dataset = concatenate_datasets([dataset1["train"], dataset2["train"]])

# 加载本地评估数据集
eval_dataset = load_dataset("json", data_files={"eval": data_path_eval})["eval"]

# ⚠️ 确保你的 JSON 和 CSV 文件的列名是 `question` 和 `answer`
def preprocess_function(examples):
    # 读取问题和答案
    questions = examples["question"]
    answers = examples["answer"]

    # 确保问题和答案是字符串类型，并进行处理
    questions = [q if isinstance(q, str) else (q[0] if q else "") for q in questions]
    answers = [a if isinstance(a, str) else (a[0] if a else "") for a in answers]

    # 拼接问题和答案
    texts = [f"{q} {a}".strip() for q, a in zip(questions, answers)]
    # 使用 tokenizer 对拼接的文本进行编码，设置最大长度并进行填充
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()  # 训练需要 `labels`
    return tokenized

# 处理训练数据集
tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)

# 处理评估数据集
tokenized_eval_datasets = eval_dataset.map(preprocess_function, batched=True)

# ========== 4. 加载基础模型 ==========
model = AutoModelForCausalLM.from_pretrained(
    model_dir, device_map="cuda", torch_dtype=torch.float16
)


print("当前设备:", torch.cuda.current_device())  # 打印当前的 GPU 设备
print("是否使用 GPU:", torch.cuda.is_available())  # 检查 GPU 是否可用
print("GPU 名称:", torch.cuda.get_device_name(torch.cuda.current_device()))  # 打印 GPU 的名称




# ========== 5. 配置 LoRA ==========
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 因果语言模型任务
    r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # 适配 LLaMA 的注意力层
)

# 添加 LoRA 适配层
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 显示可训练参数

# ========== 6. 训练参数 ==========
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # 累积梯度，减少显存压力
    num_train_epochs=3,
    save_total_limit=2,
    save_steps=500,
    eval_strategy="steps",  # 修改为新的参数名
    eval_steps=500,
    logging_dir="../logs",
    logging_steps=10,
    report_to="none"
)

# ========== 7. 启动训练 ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_eval_datasets,  # 提供评估数据集
    tokenizer=tokenizer
)

trainer.train(resume_from_checkpoint="../models/lora_finetuned/checkpoint-15500")


# ========== 8. 保存微调后的模型 ==========
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

#cd scripts
#python train.py