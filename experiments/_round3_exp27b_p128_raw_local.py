"""Local (CPU) recompute of exp27b RAW p128 — no Kaggle needed.

exp27b_dense.py's J-portion (24,576 J, ~12h T4) is a Kaggle artifact; the
RAW p128 [S, 128] per sample needs only ONE forward per sample (32 forwards,
minutes on CPU). This script reuses exp27b_dense's own functions
(_load_pool, _forward_dense) with the same seed/defaults, so the pool,
U128 selection and forward convention are identical to the Kaggle run.

CANONICAL path (future Kaggle runs): raw p128 is stored IN the J-ckpt as
`P128_RAW` per (dom, si, t) — see exp27b_dense.py. This standalone
`results/exp27b_p128_raw.pt` is a LOCAL stand-in (we can't run the 12h J-ckpt
here); it carries the same fingerprints (U128/pool_fp/S) so it can be
validated against the Kaggle J-cache later.

NOTE: pool is drawn from a streaming dataset (C4 et al.); identical seeds
do NOT guarantee identical streams. Fingerprint match with the Kaggle run
is checked downstream (exp34/35), not asserted here.
"""

import importlib.util
import os
import sys
import time

import torch

from transformers import AutoModelForCausalLM

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_spec = importlib.util.spec_from_file_location(
    "exp27b_dense", os.path.join(_ROOT, "experiments", "exp27b_dense.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LAYERS_DEFAULT = _mod.LAYERS_DEFAULT
S_DENSE = _mod.S_DENSE
_forward_dense = _mod._forward_dense
_load_pool = _mod._load_pool

MODEL = "Qwen/Qwen3.5-0.8B"
DOMAINS = ("wiki", "math", "code", "chat")
N_SAMPLES = 8
SEED = 42


def main() -> None:
    S = S_DENSE
    layers = list(LAYERS_DEFAULT)
    torch.manual_seed(SEED)
    t0 = time.time()

    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    )
    m.eval()

    pool = _load_pool(MODEL, list(DOMAINS), N_SAMPLES, 256)
    for dom in DOMAINS:
        assert len(pool[dom]) == N_SAMPLES
        print(f"[pool] {dom}: {len(pool[dom])} samples @256", flush=True)

    # U_128: top-128 by final-logit frequency at S=256 (identical to exp27 v11).
    counts: dict[int, int] = {}
    for dom in DOMAINS:
        for s in pool[dom]:
            with torch.no_grad():
                out = m(input_ids=s["input_ids"], attention_mask=s["attention_mask"])
            p = torch.softmax(out.logits[0, -1].float(), dim=-1)
            for t in p.topk(128).indices.tolist():
                counts[t] = counts.get(t, 0) + 1
    U128 = sorted([t for t, _ in sorted(counts.items(), key=lambda x: -x[1])[:128]])
    print(f"U_128 built ({len(U128)} tokens)")

    fwd: dict[tuple, dict] = {}
    for dom in DOMAINS:
        for si in range(N_SAMPLES):
            fwd[(dom, si)] = _forward_dense(m, pool[dom][si], S, layers, U128)
            print(f"  {dom}/{si} forward done "
                  f"[{(time.time() - t0) / 60:.1f} min total]")

    import hashlib
    pool_fp = {}
    for d in DOMAINS:
        blob = b"".join(x["input_ids"].flatten().numpy().tobytes() for x in pool[d])
        pool_fp[d] = hashlib.sha256(blob).hexdigest()[:12]

    torch.save(
        {"P128_RAW": {k: v["p128"] for k, v in fwd.items()},
         "U128": U128, "pool_fp": pool_fp, "S": S, "seed": SEED,
         "n_samples": N_SAMPLES, "domains": list(DOMAINS),
         "D": m.config.hidden_size,
         "source": "local-cpu-recompute (exp27b_dense functions)"},
        "results/exp27b_p128_raw.pt",
    )
    # quick sanity: raw is sub-stochastic, normalized version sums to 1
    k0 = next(iter(fwd))
    raw = fwd[k0]["p128"]
    print(f"raw row sum: {raw[0].sum().item():.4f} (sub-stochastic expected)")
    print(f"[OK] exp27b_p128_raw.pt "
          f"({len(fwd)} x [{S}, 128]) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
