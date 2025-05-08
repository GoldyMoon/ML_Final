import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset
from trl import RewardTrainer, RewardConfig
import numpy as np

def compute_metrics(eval_preds):
    """
    Pair‑wise accuracy for a reward model.

    Args
    ----
    eval_preds : transformers.trainer_utils.EvalPrediction
        • predictions – ndarray shaped (N, 2) → (reward_chosen, reward_rejected)  
        • label_ids   – unused by the metric

    Returns
    -------
    dict
        {"pairwise_accuracy": float}  # fraction where reward_chosen > reward_rejected
    """

    rewards, _ = eval_preds           
    rewards = np.asarray(rewards)
    if rewards.ndim == 1:              
        rewards = rewards.reshape(-1, 2)

    chosen, rejected = rewards[:, 0], rewards[:, 1]
    accuracy = (chosen > rejected).mean().item()

    return {"pairwise_accuracy": accuracy}


reward_model_name = "./models/Qwen2.5-1.5B-Instruct"
ref_model_name    = "./grpo_qwen_r1_1.5B_helpful-base_new/checkpoint-500"  
tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_name, num_labels=1
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
reward_model.config.pad_token_id = tokenizer.pad_token_id


policy_model = AutoModelForCausalLM.from_pretrained(reward_model_name).eval()
ref_model    = AutoModelForCausalLM.from_pretrained(ref_model_name).eval()
for p in policy_model.parameters():
    p.requires_grad = False
for p in ref_model.parameters():
    p.requires_grad = False

beta = 0.1 


class KLWeightedRewardTrainer(RewardTrainer):
    def __init__(self, policy_model, ref_model, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = next(self.model.parameters()).device
        self.policy_model = policy_model.to(device).eval()
        self.ref_model    = ref_model.to(device).eval()
        self.beta         = beta

    def compute_kl_weight(self, input_ids, attention_mask):
        """
        Compute S(x) = ∑ₜ [ log π(xₜ|…) − log π₀(xₜ|…) ] 
        and return exp(β·S(x)) for *each* example in the batch.
        """

        device = input_ids.device
        self.policy_model = self.policy_model.to(device)
        self.ref_model    = self.ref_model.to(device)

        with torch.no_grad():
            logits_policy = self.policy_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            logits_ref    = self.ref_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

            logp_policy = F.log_softmax(logits_policy[:, :-1, :], dim=-1)
            logp_ref    = F.log_softmax(logits_ref   [:, :-1, :], dim=-1)

            next_tokens = input_ids[:, 1:].unsqueeze(-1)  # (B, L−1, 1)
            lp_pol = logp_policy.gather(-1, next_tokens).squeeze(-1)  # (B, L−1)
            lp_ref = logp_ref   .gather(-1, next_tokens).squeeze(-1)  # (B, L−1)

            mask = attention_mask[:, 1:]

            S = ((lp_pol - lp_ref) * mask).sum(dim=-1)  # (B,)
            return torch.exp(self.beta * S).clamp(max=5)  

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        chosen_outputs = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
        )
        rejected_outputs = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
        )
        r_chosen = chosen_outputs["logits"]   # (B,)
        r_reject = rejected_outputs["logits"] # (B,)


        margin = r_chosen - r_reject
        per_example_loss = - F.logsigmoid(margin)  # (B,)


        weights = self.compute_kl_weight(
            inputs["input_ids_chosen"].to(model.device),
            inputs["attention_mask_chosen"].to(model.device),
        )

        loss = (weights * per_example_loss).mean()
        print(weights)
        return (loss, {
                "rewards_chosen": r_chosen,
                "rewards_rejected": r_reject,
            }) if return_outputs else loss



import re
multi_turn_pat = re.compile(
    r"Human:\s*(.*?)\nAssistant:\s*(.*?)(?=(?:\nHuman:|\Z))", re.DOTALL
)

def preprocess(batch):
    """
    batch: dict of lists, because batched=True
    returns: dict with tokenised chosen/rejected pairs
    """
    chosen_texts = []
    rejected_texts = []

    for c, r in zip(batch["chosen"], batch["rejected"]):
        chosen_turns = multi_turn_pat.findall(c)
        rejected_turns = multi_turn_pat.findall(r)


        chosen_text = "\n".join(
            f"User: {u.strip()}\nAssistant: {a.strip()}" for u, a in chosen_turns
        )
        rejected_text = "\n".join(
            f"User: {u.strip()}\nAssistant: {a.strip()}" for u, a in rejected_turns
        )
        chosen_texts.append(chosen_text)
        rejected_texts.append(rejected_text)

    chosen_enc = tokenizer(
        chosen_texts, truncation=True, padding="max_length", max_length=2048
    )
    rejected_enc = tokenizer(
        rejected_texts, truncation=True, padding="max_length", max_length=2048
    )

    return {
        "input_ids_chosen": chosen_enc["input_ids"],
        "attention_mask_chosen": chosen_enc["attention_mask"],
        "input_ids_rejected": rejected_enc["input_ids"],
        "attention_mask_rejected": rejected_enc["attention_mask"],
    }
dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

train_ds = train_dataset.map(preprocess, batched=True, num_proc=32).remove_columns(["chosen", "rejected"])
eval_ds = eval_dataset.map(preprocess,batched=True, num_proc=32).remove_columns(["chosen", "rejected"])




reward_config = RewardConfig(
    max_length=2048,
    output_dir="./reward_model_1.5B-helpful-base_2",
    per_device_train_batch_size=1,
    per_gpu_eval_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    save_strategy="steps",
    save_steps=500,
    learning_rate=1e-5,
    logging_dir="./logs/reward_model_1.5B-helpful-base_2",
    logging_steps=1,
    load_best_model_at_end=True,
    eval_strategy="steps",
    eval_steps=500,
    metric_for_best_model="pairwise_accuracy",
    greater_is_better=True,
    save_total_limit=3,
    bf16=True,
    report_to="tensorboard",
)



trainer = KLWeightedRewardTrainer(
    model=reward_model,
    policy_model=policy_model,
    ref_model=ref_model,
    beta=beta,
    args=reward_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer, 
    compute_metrics=compute_metrics,
)

trainer.train()