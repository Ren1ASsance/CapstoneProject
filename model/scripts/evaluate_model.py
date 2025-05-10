import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate
import numpy as np

# 读取评估结果文件
with open('../data/eval_results.json', 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

# 准备 BLEU 分数的平滑函数
smooth = SmoothingFunction().method4

# 加载 ROUGE 和 METEOR 指标
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

# 评估函数：计算 BLEU 分数
def calculate_bleu(reference, generated):
    reference_tokens = reference.split()
    generated_tokens = generated.split()
    bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smooth)
    return bleu_score

# 存储评估结果
bleu_scores = []
rouge_scores = []
meteor_scores = []

for entry in eval_data:
    instruction = entry.get('instruction', '')
    input_text = entry.get('input', '')  # 目前是空字符串
    reference_output = entry.get('reference_output', '')
    generated_output = entry.get('generated_output', '')

    # 计算 BLEU 分数
    bleu_score = calculate_bleu(reference_output, generated_output)
    bleu_scores.append(bleu_score)

    # 计算 ROUGE 分数
    rouge_score = rouge.compute(predictions=[generated_output], references=[reference_output])
    
    # 输出 rouge_score 的内容以调试
    print("ROUGE score:", rouge_score)  # 打印 ROUGE 分数以查看结构
    
    # 假设返回的是一个字典，包含多种 ROUGE 指标，我们尝试提取 'rouge-1' 的 F1 分数
    if 'rouge-1' in rouge_score:
        rouge_scores.append(rouge_score['rouge-1']['f'])
    else:
        rouge_scores.append(0.0)  # 如果没有 'rouge-1'，则返回0.0作为默认值

    # 计算 METEOR 分数
    meteor_score = meteor.compute(predictions=[generated_output], references=[reference_output])
    
    # 提取 METEOR 分数
    meteor_scores.append(meteor_score['meteor'])  # 假设 'meteor' 键包含数值分数

# 计算平均 BLEU 分数
average_bleu_score = np.mean(bleu_scores) if bleu_scores else 0
# 计算 BLEU 分数的中位数
median_bleu_score = np.median(bleu_scores) if bleu_scores else 0
# 计算 BLEU 分数的标准差
std_bleu_score = np.std(bleu_scores) if bleu_scores else 0

# 计算 ROUGE 分数的平均数和中位数
average_rouge_score = np.mean(rouge_scores) if rouge_scores else 0
median_rouge_score = np.median(rouge_scores) if rouge_scores else 0
std_rouge_score = np.std(rouge_scores) if rouge_scores else 0

# 计算 METEOR 分数的平均数和中位数
average_meteor_score = np.mean(meteor_scores) if meteor_scores else 0
median_meteor_score = np.median(meteor_scores) if meteor_scores else 0
std_meteor_score = np.std(meteor_scores) if meteor_scores else 0

# 输出评估结果统计
print(f"Average BLEU Score: {average_bleu_score}")
print(f"Median BLEU Score: {median_bleu_score}")
print(f"Standard Deviation of BLEU Score: {std_bleu_score}")

print(f"Average ROUGE-1 Score: {average_rouge_score}")
print(f"Median ROUGE-1 Score: {median_rouge_score}")
print(f"Standard Deviation of ROUGE-1 Score: {std_rouge_score}")

print(f"Average METEOR Score: {average_meteor_score}")
print(f"Median METEOR Score: {median_meteor_score}")
print(f"Standard Deviation of METEOR Score: {std_meteor_score}")

# 综合判断模型是否达标
# 设置评估指标的阈值
BLEU_THRESHOLD = 0.3
ROUGE_THRESHOLD = 0.4
METEOR_THRESHOLD = 0.5

# 综合评分
overall_score = (average_bleu_score + average_rouge_score + average_meteor_score) / 3

if average_bleu_score >= BLEU_THRESHOLD and average_rouge_score >= ROUGE_THRESHOLD and average_meteor_score >= METEOR_THRESHOLD:
    print("模型训练达标！")
else:
    print("模型训练未达标！")

# 保存评估结果到一个 JSON 文件
eval_results = {
    "average_bleu_score": average_bleu_score,
    "median_bleu_score": median_bleu_score,
    "std_bleu_score": std_bleu_score,
    "average_rouge_score": average_rouge_score,
    "median_rouge_score": median_rouge_score,
    "std_rouge_score": std_rouge_score,
    "average_meteor_score": average_meteor_score,
    "median_meteor_score": median_meteor_score,
    "std_meteor_score": std_meteor_score,
    "overall_score": overall_score
}

with open("../data/evaluation_analysis.json", "w", encoding="utf-8") as f:
    json.dump(eval_results, f, indent=2, ensure_ascii=False)

print("评估结果已保存至 evaluation_analysis.json")
