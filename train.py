from lora.lora import run_lora

model_ref = "CyberStew/qwen"

run_lora(model_ref=model_ref, data_path="./data", adapter_file="./checkpoints/cj.npz", tokenizer_config={"trust_remote_code": True})