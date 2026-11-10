#!/usr/bin/env python3
# ПРИКИДКА→РЕШЕНИЕ: фильтр патч-сета E-77 на РЕАЛЬНЫХ именах state_dict (meta-device,
# 0 байт весов): dim==2-фильтр теряет A_log/dt_bias/conv1d (механизм delta-нетта) ⇒
# решение: патчить всё слойное, кроме норм; массы per-tensor. Воспроизводимо (0 GPU/RAM).
import os; os.environ["OMP_NUM_THREADS"] = "2"
import torch
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained("/data/Qwen/Qwen3.5-0.8B")
with torch.device("meta"):
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
ks = list(m.state_dict().keys())
old = {n for n in ks if ".layers." in n and n.endswith(".weight") and "norm" not in n}  # ~dim2 proxy
new = {n for n in ks if ".layers." in n and not n.endswith("layernorm.weight") and "attn.norm.weight" not in n}
lost = sorted(n.split(".", 3)[3] for n in (new - old) if ".layers.0." in n)
print("старый фильтр терял (слой 0):", lost)
print("размер патч-сета: новый", len(new), "старый~", len(old))
