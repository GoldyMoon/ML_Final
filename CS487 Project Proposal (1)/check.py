from pathlib import Path
p = Path("/home/jinduog/EN601.687/Qwen2.5-0.5B-Reward").resolve()
print("Exists:", p.exists())
print("Is dir:", p.is_dir())
print("Listing:", list(p.iterdir())[:5])   # show first 5 entries
