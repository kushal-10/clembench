import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2-Llama3-76B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()


"""
timm==1.0.8
"""