import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
# ── Config 
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Using device : {DEVICE}")
print(f"Using dtype  : {DTYPE}")


# ── Load tokenizer
print(f"\nLoading tokenizer from '{MODEL_ID}' ...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)
print("Tokenizer loaded ✓")


# ── Load model 
print(f"\nLoading model from '{MODEL_ID}' ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",          # automatically places layers on available hardware
    trust_remote_code=True,
)
model.eval()
print("Model loaded ✓")
print(f"Model parameters : {sum(p.numel() for p in model.parameters()):,}")


# ── Quick inference test
def generate(prompt: str, max_new_tokens: int = 256) -> str:
    """Run a simple chat-style generation."""
    messages = [{"role": "user", "content": prompt}]

    # Apply the chat template (adds <|im_start|> / <|im_end|> tokens)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    #Convert text to numbers
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    #Generate a response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic output
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Test prompt 
if __name__ == "__main__":
    test_prompt = "Write a Python function that reverses a string."
    print(f"\nPrompt : {test_prompt}\n")
    print("Response:")
    print(generate(test_prompt))