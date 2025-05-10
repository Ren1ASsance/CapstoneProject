import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# ==== 配置路径（请确保路径正确） ====
base_model_path = "../FlagAlphaLlama3-Chinese-8B-Instruct"  # 基础模型路径
adapter_path = "../models/lora_finetuned"                  # LoRA适配器路径
output_path = "../models/merged_model"                     # 合并后输出路径

try:
    # ==== 加载基础模型 ====
    print(f"🔍 正在加载基础模型: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,        # 明确指定精度
        device_map="auto",
        trust_remote_code=True            # 关键参数：允许自定义模型代码
    )
    
    # ==== 加载tokenizer ====
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    # ==== 加载LoRA适配器 ====
    print(f"🔄 正在加载LoRA适配器: {adapter_path}")
    
    # 先检查适配器配置
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    print(f"适配器配置: {adapter_config}")
    
    # 加载适配器（添加is_trainable=False避免意外训练）
    model = PeftModel.from_pretrained(
        model, 
        adapter_path,
        is_trainable=False
    )

    # ==== 合并模型 ====
    print("⚡ 正在合并LoRA适配器...")
    merged_model = model.merge_and_unload()  # 注意使用返回值
    
    # ==== 保存合并后的模型 ====
    print(f"💾 正在保存模型到: {output_path}")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True  # 安全序列化
    )
    tokenizer.save_pretrained(output_path)
    
    print("✅ 合并完成！合并后的模型已保存至:", output_path)

except Exception as e:
    print(f"❌ 发生错误: {str(e)}")
    # 打印完整错误信息（调试用）
    import traceback
    traceback.print_exc()