import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import GRPOConfig, GRPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from bitsandbytes.optim import AdamW8bit


reward_path = Path("/home/jinduog/EN601.687/Qwen2.5-0.5B-Reward/checkpoint-87324") 
assert reward_path.is_dir(), "Reward model folder not found!"

dataset   = load_dataset("trl-lib/tldr", split="train")


rm_tokenizer = AutoTokenizer.from_pretrained(
    reward_path, local_files_only=True, padding_side="right")
rm_tokenizer.pad_token = rm_tokenizer.eos_token

reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_path,
    num_labels=1,
    device_map="auto",
    torch_dtype="bfloat16",
    local_files_only=True,
).eval()

def reward_fn(completions, **_):
    with torch.no_grad():
        toks = rm_tokenizer(completions, padding=True, truncation=True,
                            return_tensors="pt").to(reward_model.device)
        return reward_model(**toks).logits.squeeze(-1).cpu().tolist()

cfg = GRPOConfig(
    output_dir="grpo_qwen_r1",        
    per_device_train_batch_size=8,    
    gradient_accumulation_steps=4, 
    max_prompt_length=512,          
    max_completion_length=128,
    num_iterations=4,                  
    beta=0.1,                         
    learning_rate=1e-5,             
    max_steps=10_000,
    logging_steps=50,
    save_steps=1_000,
)

trainer = GRPOTrainer(model="Qwen/Qwen2-0.5B-Instruct", reward_funcs=reward_fn,
                      train_dataset=dataset, args=cfg,)
trainer.train()
