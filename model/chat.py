import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class EnhancedLlamaChat:
    def __init__(self):
        self.MODEL_PATH = "models/merged_model"
        self._init_device()
        self._load_model()
        self._warmup()

    def _init_device(self):
        """初始化设备配置"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🛠️ 运行设备: {self.device.upper()}")
        if self.device == "cuda":
            print(f"  显卡型号: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA内存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    def _load_model(self):
        """加载模型和tokenizer"""
        print("🔌 正在加载模型...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_PATH,
                device_map="auto",  # 自动分配GPU/CPU
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_PATH,
                trust_remote_code=True
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"ℹ️ 设置pad_token: {self.tokenizer.pad_token}")
            
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"✅ 模型加载完成 | 参数量: {total_params/1e9:.2f}B")

        except Exception as e:
            print(f"❌ 加载失败: {str(e)}")
            raise

    def _warmup(self):
        """预热模型"""
        print("🔥 模型预热中...")
        test_prompt = "模型预热"
        inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            _ = self.model.generate(**inputs, max_new_tokens=10)
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def generate(self, prompt, **kwargs):
        """生成文本"""
        inputs = self.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
        ).to(self.device)

        default_generate_args = {
        "max_new_tokens": 256,
        "do_sample": True,              # 默认启用采样模式
        "top_k": 50,
        "top_p": 0.9,
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        }

        # 合并默认参数与外部传入的 kwargs（外部优先）
        generate_args = {**default_generate_args, **kwargs}

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **generate_args
            )

            return self.tokenizer.decode(output[0], skip_special_tokens=True)   


    def interactive_chat(self):
        """交互式对话"""
        print("\n💬 进入对话模式 (输入 'quit' 退出)")
        while True:
            try:
                user_input = input("\n👤 您: ")
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                response = self.generate(
                    f"<|User|>{user_input}<|Bot|>",
                    temperature=0.8
                ).split("<|Bot|>")[-1].strip()
                
                print(f"\n🤖 AI: {response}")
                
            except KeyboardInterrupt:
                print("\n🛑 对话终止")
                break

if __name__ == "__main__":
    bot = EnhancedLlamaChat()
    
    # 快速测试
    print("\n🚀 测试生成:")
    test_q = "Where is the captical of American？"
    print(f"Q: {test_q}")
    print(f"A: {bot.generate(test_q)}")
    
    # 开启对话
    bot.interactive_chat()
