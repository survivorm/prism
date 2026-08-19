"""Metric battery for pull-experiments (Exp16–19).

All metrics operate on the *edited* small model vs. original small model
and the large (senior) model.  Designed to run on CPU with optional GPU paths.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kl_div(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p ‖ q)  on probability vectors  [U].  Stable log-version."""
    eps = 1e-12
    return (p * torch.log((p + eps) / (q + eps))).sum().item()


def _perplexity(logits: torch.Tensor, ids: torch.Tensor) -> float:
    """Perplexity on a single sequence (logits [S, V], ids [S])."""
    loss = F.cross_entropy(logits[:-1].float(), ids[1:], reduction="mean")
    return float(torch.exp(loss).item())


def _topk_set(logits: torch.Tensor, k: int) -> list[int]:
    """Return top-k token indices (no gradients)."""
    return logits.topk(k, dim=-1).indices[0].cpu().tolist()


def _topp_set(logits: torch.Tensor, p: float) -> list[int]:
    """Return minimal set whose cumulative softmax probability >= p."""
    probs = F.softmax(logits.float(), dim=-1)
    vals, idx = probs.sort(descending=True)
    cum = vals.cumsum(0)
    cut = (cum >= p).float().argmin().item() + 1
    return idx[:cut].cpu().tolist()


# ---------------------------------------------------------------------------
# Core battery
# ---------------------------------------------------------------------------

def eval_battery(
    m_edited,
    m_orig,
    mL,
    prefixes: list[dict[str, torch.Tensor]],
    *,
    token_set_cache: Optional[dict] = None,
    K: int = 64,
    device: Optional[str] = None,
) -> dict:
    """Compute the full metric battery on *prefixes*.

    Parameters
    ----------
    m_edited : nn.Module
        The edited small model.
    m_orig : nn.Module
        The unedited small model (same architecture).
    mL : nn.Module
        The large / senior model.
    prefixes : list[dict]
        Calibration samples with ``input_ids`` and ``attention_mask``.
    token_set_cache : optional dict
        Precomputed per-sample target cache ``{"J_tgt_p", "J_S_p", "p", "pair"}``.
        If provided, bridge-per-pair metrics are computed.
    K : int
        Token set size for building token sets from logits.
    device : optional str
        Target device; auto-detected from model if ``None``.

    Returns
    -------
    dict  with keys:
        - A_L_top15, A_S_top15, A_L_topp, A_S_topp  (sets of ints per prefix)
        - U_top15, U_topp  (union sets)
        - |A_S ∩ A_L|, |U|  per variant
        - bridge_per_pair  (if token_set_cache provided) — list of {pair, bridge}
        - KL_ed_L          KL(p_ed ‖ p_L) averaged over prefixes
        - P_ed_A_L         mean probability mass m_edited assigns to A_L tokens
        - rank_top10_L_in_ed  mean rank of top-10 senior tokens in p_ed
        - hit5_in_U        fraction of top-5 p_ed tokens falling in U
        - P_ed_U           probability mass p_ed assigns to U
        - KL_ed_orig       KL(p_ed ‖ p_orig)  — leash metric
        - PPL_ed_orig      perplexity on 2 natural sequences
        - (GSM8K acc only on GPU; omitted on CPU)
    """
    if device is None:
        device = str(next(m_edited.parameters()).device)

    m_edited.eval()
    m_orig.eval()
    mL.eval()

    # -- Per-prefix token sets from the *large* and *original small* ---------
    all_A_L_top15: list[set] = []
    all_A_S_top15: list[set] = []
    all_A_L_topp: list[set] = []
    all_A_S_topp: list[set] = []

    kl_ed_L_sum = 0.0
    kl_ed_orig_sum = 0.0
    p_ed_A_L_sum = 0.0
    rank_sum = 0.0
    hit5_sum = 0.0
    P_ed_U_sum = 0.0
    n_valid = 0

    with torch.no_grad():
        for s in prefixes:
            ids = s["input_ids"].to(device)
            mask = s["attention_mask"].to(device)

            # Logits from all three models
            logits_L = mL(input_ids=ids, attention_mask=mask).logits[:, -1, :]
            logits_S = m_orig(input_ids=ids, attention_mask=mask).logits[:, -1, :]
            logits_ed = m_edited(input_ids=ids, attention_mask=mask).logits[:, -1, :]

            # Softmax probs on full vocab
            p_L = F.softmax(logits_L.float(), dim=-1)
            p_S = F.softmax(logits_S.float(), dim=-1)
            p_ed = F.softmax(logits_ed.float(), dim=-1)

            # Token sets top-15 and top-p 0.9
            A_L_top15 = set(_topk_set(logits_L, 15))
            A_S_top15 = set(_topk_set(logits_S, 15))
            A_L_topp = set(_topp_set(logits_L, 0.9))
            A_S_topp = set(_topp_set(logits_S, 0.9))

            all_A_L_top15.append(A_L_top15)
            all_A_S_top15.append(A_S_top15)
            all_A_L_topp.append(A_L_topp)
            all_A_S_topp.append(A_S_topp)

            # Union top-15
            U_top15 = A_L_top15 | A_S_top15
            # Union top-p
            U_topp = A_L_topp | A_S_topp

            # -- KL(p_ed ‖ p_L) ------------------------------------------
            kl_ed_L_sum += _kl_div(p_ed[0], p_L[0])

            # -- KL(p_ed ‖ p_orig) ---------------------------------------
            kl_ed_orig_sum += _kl_div(p_ed[0], p_S[0])

            # -- P_ed(A_L)  on top-15 ------------------------------------
            mask_AL = torch.zeros(p_ed.shape[-1], device=device, dtype=torch.bool)
            mask_AL[list(A_L_top15)] = True
            p_ed_A_L_sum += p_ed[0, mask_AL].sum().item()

            # -- Mean rank of top-10 senior tokens in p_ed ----------------
            top10_L = _topk_set(logits_L, 10)
            ranks = []
            for t in top10_L:
                r = (p_ed[0] > p_ed[0, t]).sum().item() + 1
                ranks.append(r)
            rank_sum += sum(ranks) / len(ranks)

            # -- hit5_in_U (top-5 of p_ed that land in U_top15) ----------
            top5_ed = _topk_set(logits_ed, 5)
            hits = sum(1 for t in top5_ed if t in U_top15)
            hit5_sum += hits / 5.0

            # -- P_ed(U_top15) -------------------------------------------
            p_ed_U_sum += p_ed[0, list(U_top15)].sum().item()

            n_valid += 1

    N = max(n_valid, 1)
    result = {
        "A_L_top15": all_A_L_top15,
        "A_S_top15": all_A_S_top15,
        "A_L_topp": all_A_L_topp,
        "A_S_topp": all_A_S_topp,
        "KL_ed_L": kl_ed_L_sum / N,
        "KL_ed_orig": kl_ed_orig_sum / N,
        "P_ed_A_L": p_ed_A_L_sum / N,
        "rank_top10_L_in_ed": rank_sum / N,
        "hit5_in_U": hit5_sum / N,
        "P_ed_U_top15": p_ed_U_sum / N,
    }

    # -- Bridge per pair (if cache provided) -----------------------------
    if token_set_cache is not None:
        bridges = []
        for pair_key, cache in token_set_cache.items():
            J_tgt_p = cache["J_tgt_p"].to(device)
            # Recompute J_S_p on edited model
            from prism.jacobian import (  # noqa: PLC0414
                compute_jacobian_for_tokens,
                softmax_jacobian,
            )
            J_S_ed = compute_jacobian_for_tokens(
                m_edited, cache.get("l_S", 0),
                prefixes[0]["input_ids"].to(device),
                prefixes[0]["attention_mask"].to(device),
                cache["token_set"],
            )
            J_S_ed_p = softmax_jacobian(J_S_ed, cache["p"].to(device))
            # Bridge = cosine dissimilarity between J_tgt_p and J_S_ed_p
            cos = F.cosine_similarity(J_tgt_p.float(), J_S_ed_p.float(), dim=1)
            bridges.append({"pair": pair_key, "bridge": float(cos.mean().cpu())})
        result["bridge_per_pair"] = bridges

    # -- PPL on 2 natural sequences ---------------------------------------
    if len(prefixes) >= 2:
        ppl_vals = []
        for s in prefixes[:2]:
            ids = s["input_ids"].to(device)
            mask = s["attention_mask"].to(device)
            with torch.no_grad():
                logits_ed = m_edited(input_ids=ids, attention_mask=mask).logits
            ppl_vals.append(_perplexity(logits_ed[0], ids[0]))
        result["PPL_ed_orig"] = sum(ppl_vals) / len(ppl_vals)

    # -- Intersection sizes (context) -------------------------------------
    result["intersection_top15"] = [
        len(aL & aS) for aL, aS in zip(all_A_L_top15, all_A_S_top15)
    ]
    result["union_top15_size"] = [
        len(aL | aS) for aL, aS in zip(all_A_L_top15, all_A_S_top15)
    ]

    # -- GSM8K (GPU only) -------------------------------------------------
    if device.startswith("cuda"):
        result["gsm8k_accuracy"] = _gsm8k_few_shot(m_edited, device)
    else:
        result["gsm8k_accuracy"] = None  # skipped on CPU

    return result


# ---------------------------------------------------------------------------
# GSM8K few-shot helper (GPU only)
# ---------------------------------------------------------------------------

def _gsm8k_few_shot(model, device: str, n_fewshot: int = 3, n_test: int = 20) -> float:
    """Few-shot GSM8K accuracy from the real openai/gsm8k dataset (GPU only)."""
    import re
    from datasets import load_dataset  # noqa: PLC0414
    from transformers import AutoTokenizer  # noqa: PLC0414

    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    except Exception:
        return None  # dataset unavailable

    # Load few-shot examples (take first n_fewshot that parse correctly)
    fewshot_examples = []
    for raw in ds:
        q = raw["question"].strip()
        a_text = raw["answer"].strip()
        nums = re.findall(r"The answer is (\d+)", a_text)
        if nums and q:
            fewshot_examples.append(f"Question: {q}\nAnswer: {a_text}")
        if len(fewshot_examples) >= n_fewshot:
            break

    if not fewshot_examples:
        return None

    fewshot_prompt = "\n\n".join(fewshot_examples)

    # Load test examples (skip the ones used as few-shot)
    test_ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    used_questions = {e.split("\n")[0] for e in fewshot_examples}
    test_examples = []
    for raw in test_ds:
        q = raw["question"].strip()
        if q in used_questions:
            continue
        nums = re.findall(r"The answer is (\d+)", raw["answer"].strip())
        if nums and q:
            test_examples.append((q, int(nums[0])))
        if len(test_examples) >= n_test:
            break

    if not test_examples:
        return None

    correct = 0
    for q, answer in test_examples:
        prompt = fewshot_prompt + "\n\nQuestion: " + q + "\nAnswer:"
        tokens = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        try:
            generated = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(generated[0][ids.shape[-1]:], skip_special_tokens=True)
        except Exception:
            continue
        # Extract last number from generation
        nums = re.findall(r"\d+", text)
        if nums:
            pred = int(nums[-1])
            if pred == answer:
                correct += 1

    return correct / len(test_examples)
