"""
Test your fine-tuned Qwen2.5-Coder model with 4-bit quantization + offload.
Run this AFTER finetune_qwen.py has completed.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ── CONFIG 
BASE_MODEL_ID  = "Qwen/Qwen2.5-Coder-3B-Instruct"
FINETUNED_DIR  = "./qwen-finetuned"
OFFLOAD_DIR    = "./offload"           # โฟลเดอร์สำหรับ offload layers


# ── สร้างโฟลเดอร์ offload อัตโนมัติ
os.makedirs(OFFLOAD_DIR, exist_ok=True)


# ── 4-BIT CONFIG 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ── LOAD TOKENIZER 
print(" Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    FINETUNED_DIR,
    trust_remote_code=True,
)
print(" Tokenizer loaded")


# ── LOAD BASE MODEL 
print("\n Loading base model in 4-bit...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
    trust_remote_code=True,
)
print(" Base model loaded")


# ── LOAD LoRA WEIGHTS 
print("\n Loading LoRA fine-tuned weights...")
model = PeftModel.from_pretrained(
    base_model,
    FINETUNED_DIR,
    offload_dir=OFFLOAD_DIR,          
    offload_buffers=True,
)
model.eval()
print(" Model ready!\n")


# ── GENERATE FUNCTION 
def generate(instruction: str, max_new_tokens: int = 256) -> str:
    prompt = f"""<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── TEST PROMPTS 
if __name__ == "__main__":
    test_prompts = [
        "Write a Python function to  find maximun number in list  and explain how function work"
    ]

    for prompt in test_prompts:
        print(f" Instruction : {prompt}")
        print(f" Response    :\n{generate(prompt)}")
        print("-" * 60)
