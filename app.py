from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# CORS 设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "model/models/merged_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",         
    load_in_8bit=True,         
    trust_remote_code=True
)

print("Model loaded successfully with 8-bit quantization on Windows!")

class UserInput(BaseModel):
    prompt: str

async def generate_response(user_input: UserInput):
    try:
        # 生成响应
        inputs = tokenizer(user_input.prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

    #uvicorn backend.app:app --reload
