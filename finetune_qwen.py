import json
import torch
from datasets import Dataset
from transformers import (
  AutoModelForCausalLM, #automatically loads the right model
  AutoTokenizer,
  BitsAndBytesConfig,
  TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

MODEL_ID        = "Qwen/Qwen2.5-Coder-3B-Instruct"
DATASET_FILE    = "dataset_fixed.jsonl"      # your dataset file
OUTPUT_DIR      = "./qwen-finetuned"   # where the fine-tuned model is saved
MAX_SEQ_LENGTH  = 512                  # max tokens per sample
NUM_EPOCHS      = 5                    # how many times to train on full dataset
BATCH_SIZE      = 2                    # samples per step (lower = less VRAM)
LEARNING_RATE   = 2e-4                 # how fast the model learns



with open(DATASET_FILE, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f if line.strip()]
    
# Format each sample into a chat-style prompt
def format_sample(sample):
    output = sample['output']
    
    if 'explanation' in sample:
        output += f"\n\n# Explanation:\n# {sample['explanation']}"
    
    if 'examples' in sample:
        output += f"\n\n# Examples:\n# {sample['examples']}"
    
    return {
        "text": f"""<|im_start|>user
        {sample['instruction']}<|im_end|>
        <|im_start|>assistant
        {output}<|im_end|>"""
    }
formatted_data = [format_sample(s) for s in raw_data]
dataset = Dataset.from_list(formatted_data)

# Split into train (90%) and validation (10%)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset  = dataset["test"]

#Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token   # use EOS as padding token
tokenizer.padding_side = "right"            # pad on the right side

#Load model in 4- bits

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # compress model to 4-bit
    bnb_4bit_quant_type="nf4",                  # NF4 = best quality 4-bit format
    bnb_4bit_compute_dtype=torch.float16,       # compute in float16
    bnb_4bit_use_double_quant=True,             # extra compression
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False   # disable cache during training

# APPLY LoRA 
# LoRA adds small trainable layers instead of training all 3 billion parameters.

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,               # LoRA rank — higher = more capacity but more memory (8-64)
    lora_alpha=32,      # scaling factor, usually 2x the rank
    lora_dropout=0.05,  # dropout for regularization (prevents overfitting)
    target_modules=[    # which layers to apply LoRA to
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)

trainable, total = 0, 0
for p in model.parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()

print(f"   LoRA applied")
print(f"   Trainable params : {trainable:,}  ({100 * trainable / total:.2f}% of total)")
print(f"   Total params     : {total:,}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,     # accumulate gradients to simulate larger batch
    learning_rate=LEARNING_RATE,
    fp16=True,                          # use float16 to save memory
    logging_steps=10,                   # print loss every 10 steps
    evaluation_strategy="epoch",        # evaluate after each epoch
    save_strategy="epoch",              # save checkpoint after each epoch
    load_best_model_at_end=True,        # keep the best checkpoint
    warmup_ratio=0.03,                  # gradual warmup at start of training
    lr_scheduler_type="cosine",         # gradually reduce learning rate
    report_to="none",                   # disable wandb/tensorboard logging
)

print("\n Setting up trainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",      # the field in dataset that has the text
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,                  # don't pack multiple samples into one
)

#TRAIN
print("\n Starting training...")
print(f"   Epochs        : {NUM_EPOCHS}")
print(f"   Batch size    : {BATCH_SIZE}")
print(f"   Learning rate : {LEARNING_RATE}")
print(f"   Output dir    : {OUTPUT_DIR}\n")

trainer.train()

print("\n Training complete!")


# ── 9. SAVE THE FINE-TUNED MODEL
print(f"\n Saving model to '{OUTPUT_DIR}'...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(" Model saved!")
print(f"\n Fine-tuning complete! Your model is saved in '{OUTPUT_DIR}'")
