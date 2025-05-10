import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# ===== 模型路径和设备设置 =====
model_dir = "../models/merged_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 量化配置 =====
if device == "cuda":
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# ===== 加载数据函数（兼容多格式）=====
def load_eval_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    eval_data = []
    skipped = 0
    for item in raw_data:
        # 支持两种字段格式
        instruction = item.get("instruction") or item.get("question")
        output = item.get("output") or item.get("answer")
        input_text = item.get("input", "")  # 没有 input 就默认空字符串

        if not instruction or not output:
            skipped += 1
            continue  # 忽略不完整数据

        eval_data.append({
            "instruction": instruction,
            "input": input_text,
            "reference_output": output
        })

    print(f"加载数据完成，有效样本数: {len(eval_data)}，跳过无效样本数: {skipped}")
    return eval_data

# ===== 推理函数 =====
def generate_answer(instruction, input_text=""):
    prompt = instruction if input_text == "" else f"{instruction}\n{input_text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===== 主流程 =====
eval_data_path = "../data/eval.json"
results_path = "../data/eval_results.json"

try:
    eval_data = load_eval_data(eval_data_path)

    print("开始评估……")
    results = []
    for sample in tqdm(eval_data):
        instruction = sample["instruction"]
        input_text = sample["input"]
        reference_output = sample["reference_output"]

        generated_output = generate_answer(instruction, input_text)

        results.append({
            "instruction": instruction,
            "input": input_text,
            "reference_output": reference_output,
            "generated_output": generated_output
        })

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"评估完成，结果已保存至 {results_path}")

except Exception as e:
    print(f"主流程失败: {e}")
