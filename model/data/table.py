import json
import matplotlib.pyplot as plt

# 读取你的 JSON 文件
with open('evaluation_analysis.json', 'r') as f:
    data = json.load(f)

# 提取 BLEU 分数
average_bleu = data['average_bleu_score']

# 提取 ROUGE 分数
rouge_results = data['rouge_results']
rouge1 = [item['rouge1'] for item in rouge_results]
rouge2 = [item['rouge2'] for item in rouge_results]
rougeL = [item['rougeL'] for item in rouge_results]

# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(rouge1, label='ROUGE-1', marker='o')
plt.plot(rouge2, label='ROUGE-2', marker='x')
plt.plot(rougeL, label='ROUGE-L', marker='s')
plt.axhline(y=average_bleu, color='r', linestyle='--', label=f'Average BLEU: {average_bleu:.2f}')
plt.title('Evaluation Metrics Over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()
