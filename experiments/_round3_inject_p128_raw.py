"""Local (CPU) injection of P128_RAW into the exp27b J-cache.

The J-cache ckpts store P128 normalized (sum=1) and lack the raw
sub-stochastic p128 row-sums that exp34's sentinel needs (sum<0.5). A full
re-stream of the pool is impossible here (streaming datasets are non-deterministic),
but exp27b_pool_ids_L23.pt holds the EXACT S=128 prefix ids (fingerprints match
the ckpt pool_fp), so we run one forward per sample and gather raw p128 over
U128 — reproducing the raw rows that _forward_dense would have produced.

Writes a backup + in-place P128_RAW field into each exp27b_jcache_L{l}.pt.
"""

import os
import sys
import torch

MODEL = "Qwen/Qwen3.5-0.8B"
CKPT_DIR = "/data/prism_kernels/exp27b/results"
LAYER_CKPT = CKPT_DIR + "/exp27b_jcache_L{0}.pt"
POOL_IDS = CKPT_DIR + "/exp27b_pool_ids_L23.pt"
LAYERS = (6, 12, 18, 22, 23)
S = 128


def main():
    from transformers import AutoModelForCausalLM

    torch.manual_seed(42)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    )
    m.eval()

    pi = torch.load(POOL_IDS, map_location="cpu")
    ids = pi["ids"]
    ref_fp = pi["meta"]["pool_fp"]

    # U128 + pool_fp from a reference ckpt (all layers share them).
    ck0 = torch.load(LAYER_CKPT.format(6), map_location="cpu")
    U128 = ck0["U128"]
    ck_fp = ck0["pool_fp"]
    assert ck_fp == ref_fp, (ck_fp, ref_fp)
    u = torch.as_tensor(U128, dtype=torch.long)
    print(f"U128 {len(U128)}, pool_fp match: {ck_fp}")

    # One forward per sample -> raw p128 [S,128] (sub-stochastic over U128).
    raw: dict[tuple, torch.Tensor] = {}
    for dom, lst in ids.items():
        for si in range(len(lst)):
            x = torch.as_tensor(lst[si]).view(1, -1).long()     # [1,128]
            assert x.shape[1] == S
            with torch.no_grad():
                out = m(input_ids=x, attention_mask=torch.ones_like(x))
            prow = torch.softmax(out.logits[0].float(), dim=-1)[:, u]  # [128,128]
            for t in range(S):
                # t indexes from the end: p128 row at position S-1-t
                raw[(dom, si, t)] = prow[S - 1 - t].cpu()
            print(f"  {dom}/{si} forward done", flush=True)
    print(f"raw rows: {len(raw)} (expect {len(ids)}x8x128 = {sum(len(v) for v in ids.values())*S})")

    # Sanity: normalized P128 in ckpt must equal raw / raw.sum().
    k0 = ("wiki", 0, 0)
    p_norm = ck0["P128"][k0]
    p_raw = raw[k0]
    rel = (p_raw / p_raw.sum() - p_norm).abs().max().item()
    print(f"normalized-vs-raw/sum max abs diff @wiki/0/0: {rel:.2e}")

    # Inject into each layer ckpt (backup first).
    for l in LAYERS:
        p = LAYER_CKPT.format(l)
        bak = p + ".pre_raw.bak"
        if not os.path.exists(bak):
            os.replace(p, bak)
        ck = torch.load(bak, map_location="cpu")
        # keys must align with the raw rows (same domains/n_samples/S)
        assert list(ck["domains"]) == list(ids.keys()), (ck["domains"], list(ids.keys()))
        ck["P128_RAW"] = raw
        ck["params_raw"] = {"model": MODEL, "source": "local-cpu-injection from exp27b_pool_ids_L23.pt",
                            "pool_fp": ref_fp, "S": S}
        torch.save(ck, p)
        # verify round-trip
        ck2 = torch.load(p, map_location="cpu")
        assert "P128_RAW" in ck2 and len(ck2["P128_RAW"]) == len(ck2["JR"])
        print(f"  L{l}: P128_RAW injected ({len(ck2['P128_RAW'])} rows), backup -> {os.path.basename(bak)}")
    print("[OK] P128_RAW injected into all layer ckpts")


if __name__ == "__main__":
    main()
