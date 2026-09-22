# Exp 27 — Noise-Floor Calibration and Robustness of the J-Field: Findings

Date: 2026-09-22. Data: `results/exp27_noise_floor.json` (v11, Kaggle
`voroninmichael/prism-exp27-noise-floor`, commit `97ce2cb` line of script
66272e3, run 8147s). Model Qwen3.5-0.8B float32, 32 samples (8 per domain),
layers 6/12/18/19/22/23, K_max=128, baseline S=64 K=64 pos0.
Checkpoint `exp27_jcache.pt` (per-domain pool_fp, reusable across runs) —
this is the first per-domain ckpt; future runs with identical pools reuse it.

## 1. v10 -> v11 changes

- Math pool: full 320-word shift + dedup. v10 legacy concat had a 120-word
  overlap (320 window, 200 shift), so the 8 math samples reduced to 2 unique
  texts (bit-identical si 0,1,2,3,5,7 and si 4,6 in v10). v11: 8/8 unique
  (no dedup WARNING in log).
- Truncation test 6.2 (pair 64,65) was retired: pos64@S128 sees exactly
  tokens 0..63 = pos0@S64, so v10 measured an identity (ratio 1.0000 at all
  layers, |diff| ~1e-7). Not a truncation probe.
- Dip diagnostic added: pair (32,33)@S128 — genuinely different contexts
  (pos32 sees 95 tokens, pos33 sees 96).
- J/H cache keys got the layer (v9 bug: 6 layers identical), checkpoint
  resume + `--resume` local metrics recompute.

## 2. Baseline (S=64, K=64, pos0) — verdict table

Noise floor F = 1/sqrt(r_eff*(K+D-r_eff)), r_eff entropic. K=64: F 0.0046-0.0071.
Tiers: ORTHOGONAL C<F, WEAK F<=C<3F, COHERENT C>=3F.

| Layer | r_eff | F | C_a(pos-pk) pair0_1 | C_b | tier C_a | tier C_b |
|---|---|---|---|---|---|---|
| L6 | 18.6 | 0.0071 | 0.061 | +0.009 | COHERENT (8x) | ORTHOGONAL |
| L12 | 26.9 | 0.0059 | 0.095 | +0.021 | COHERENT (14x) | WEAK (3x) |
| L18 | 39.6 | 0.0049 | 0.160 | +0.063 | COHERENT (29x) | COHERENT (11x) |
| L19 | 41.6 | 0.0048 | 0.167 | +0.068 | COHERENT (30x) | COHERENT (12x) |
| L22 | 45.7 | 0.0046 | 0.223 | +0.104 | COHERENT (39x) | COHERENT (18x) |
| L23 | 38.3 | 0.0050 | 0.229 | +0.098 | COHERENT (41x) | COHERENT (17x) |

r_eff 18-46 at K=64 (entropic); stable rank 3.1-7.3, participation ratio
7.6-27.4. F is 4-9x below the 1/sqrt(D)=0.0312 used in exp01 v6 — that floor
was 3-8x too high and made exp01 v6 call the field WEAK/ORTHOGONAL.

## 3. The field is fibered: coherent along a sample, orthogonal between samples

C_a(pos-pk) COHERENT at all 6 layers (12-55x F; v11 clean pool 8-41x F) and
rises monotonically with depth 0.061 -> 0.229. C_b (between-sample, same
domain) is ORTHOGONAL at L6/L12 (C<F) and only WEAK/COHERENT from L18, where
it reaches 11-18x F. Same-sample adjacent positions stay aligned 10-60x
better than between-sample alignment across all depth. This is the fibered
(bundle) structure seen in exp01 v4, now measured against the honest
r_eff-corrected floor F: the "fibered/orthogonal" verdict of exp01 v6 was
partly real (C_b is near-orthogonal in shallow layers) and partly a floor
artifact (1/sqrt(D) 0.031 vs true F ~0.005 made even 10x-F signals look
sub-threshold).

Between-samples alignment GROWS with depth (C_b 0.009 -> 0.098, L6 -> L23):
deep layers progressively pool across samples.

## 4. Sweeps (v11, clean pool)

S-sweep (pos-pk, pair 0_1): L6 0.090/0.067/0.062/0.085, L18
0.176/0.171/0.132/0.183, L23 0.248/0.258/0.165/0.263 at S=32/64/128/256.
Non-monotonic, dip at S=128 at all layers.
Position sweep (S=64): L6 pairs 0_1/1_2/2_4/4_8/8_16 =
0.067/0.116/0.032/0.019/0.011. Peak at 1_2, decay by 4_8: coherence length
~1-2 positions.
K-sweep (pos0): C_a 0.056/0.061/0.055 at K=32/64/128 (L6), F
0.0084/0.0071/0.0035 — flat in K, not a token-set artifact; gauge_share
declines with K (0.0014/0.0007/0.0005 L6).

## 5. The S=128 dip is a mid-context dip, not a boundary effect

Pair (32,33)@S128 (context 96/97, mid-sequence): L6 0.081, L12 0.112,
L18 0.132, L19 0.140, L22 0.232, L23 0.296 — above S64 and above
pos0@S128 at every layer. The dip sits at context 127 (pos0@S128) and
recovers at context 255. Hypothesis: GDN chunked-attention boundary at 128
tokens; unverified, would need architecture-level check.

## 6. Post-processing diagnostics (from JSON, no GPU)

- C_a/C_b compression: ratio falls 6.6 (L6) -> 2.1 (L22): samples are
  progressively "squeezed" toward each other with depth.
- ARI of per-sample (cpos/csame/craw) features vs domain: ~0 at all layers
  (-0.03..+0.02): no stable domain clusters in the scalar feature space.
- L23 sigma_d ~100 vs 1.7-7.2 elsewhere: `hidden_states[-1]` is pre-final-
  RMSNorm; residual norms grow ~14x through the last block. Not a bug.
- v10 -> v11: C_b collapsed ~16x (0.155 -> 0.009 at L6) — the v10 between-
  sample "COHERENT" tiers were inflated by the math duplicate pool (28 of
  112 pairs at C~1). v11 math C_b 0.039/0.063/0.106/0.117/0.145/0.158
  matches the other domains: the math artifact is gone.

## 7. Conclusions

1. The gauge-free J field is coherent at every layer (C_a 8-55x F) —
   pre-registered rule 7.1: real coherence, the field is NOT fractal, NOT
   fibered-orthogonal in the exp01 v6 sense.
2. Fibered structure: sheets per sample, near-orthogonal in shallow layers,
   progressively pooled from L18 on. Domain-conditional gramians must stay
   sample-local (exp16/19 convention); domain-level G^d is a winner-take-all
   artifact at N<=8.
3. Coherence length ~1-2 positions; S=128 mid-context dip (GDN boundary,
   unverified); K=64 representative.
4. Honest floor F (entropic r_eff) is the correct calibration: 1/sqrt(D)
   overstates it 4-9x and flips tiers.
5. The v11 per-domain checkpoint `exp27_jcache.pt` is reusable for future
   runs with identical pools (per-domain pool_fp match).

## 8. Next

- GDN 128-boundary check (S=120/128/132 micro-sweep) for the dip.
- Full J-space clustering for ARI (requires the 1.7GB ckpt, not the JSON
  scalars): do the sheets cluster by domain?
- v10 vs v11 C_b table as a "math artifact" case study in the paper.
