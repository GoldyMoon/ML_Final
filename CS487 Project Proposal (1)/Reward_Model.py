from trl import RewardConfig, RewardTrainer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, LlamaTokenizer,LlamaForSequenceClassification, AutoModelForCausalLM
import torch.nn as nn
from bitsandbytes.optim import AdamW8bit

print("Prepare tokenizer")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token        # reuse <eos> as <pad>
    tokenizer.padding_side = "right" 


print("Prepare model")
# model = AutoModelForSequenceClassification.from_pretrained(
#     "meta-llama/Llama-2-7b-chat-hf", num_labels=1, ignore_mismatched_sizes=True    
# )


model = LlamaForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    num_labels=1,
    ignore_mismatched_sizes=True  # Needed because we're adding a new head
)

model.config.pad_token_id = tokenizer.pad_token_id
model.gradient_checkpointing_enable()

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = RewardConfig(output_dir="LLaMA2-7B-Reward", 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # optional but helps with large models
    optim="adamw_bnb_8bit",
    fp16 = True,
    gradient_checkpointing=True,
    learning_rate=5e-6,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch")
trainer = RewardTrainer(
    args=training_args,
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
)
print("Prepare training")
trainer.train()
