# at the top of the script, after imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, functools

@functools.lru_cache()                       # load once per model name
def load_local_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,          # or float16 if GPUs < Ampere
        device_map="auto",                   # spread across all GPUs
        trust_remote_code=True,
    )
    return tok, mdl
