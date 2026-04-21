import torch
from vggt.models.vggt import VGGT

custom_cache_dir = "/home/ghr/fs/Junyi/data/proj/model_weights"

model = VGGT.from_pretrained(
    "facebook/VGGT-1B", 
    cache_dir=custom_cache_dir
)