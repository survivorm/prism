# Exp 28 — findings: leaf structure and local curvature of the J field (10-02)

Data: exp27b dense cache (S=128, t=0..127, K=128, D=1024, 32 docs, 6 layers).
Method: Δ_SVD(t) = max principal angle of top-r right subspaces of J_t, J_{t+1};
κ(t) = ||R_01 R_12 R_02^{-1} − I||_F (Procrustes holonomy of triples); r=16 main,
sweep {8,16,32}; random baseline = B=200 Haar subspaces of R^1024 (basis-invariant
metrics — rotation of the data subspace is a no-op, TZ-baseline corrected);
MAD boundary detection; 3264... 192 docs, batched SVD, ~22 min CPU.

## Result (pooled means over 32 docs × 126/125 positions)

| layer | Δ_SVD (r=16) | κ (r=16) | Δ rand | κ rand | boundaries | flat_frac | κ_flat |
|-------|-------------|----------|--------|--------|------------|-----------|--------|
| L6    | 1.549       | 3.960    | 1.565  | 5.449  | 0/1248     | 0.000     | 0.000  |
| L12   | 1.542       | 3.469    | 1.565  | 5.429  | 0/1248     | 0.000     | 0.000  |
| L18   | 1.495       | 2.356    | 1.565  | 5.443  | 0/1248     | 0.000     | 0.000  |
| L19   | 1.483       | 2.170    | 1.565  | 5.434  | 0/1248     | 0.000     | 0.000  |
| L22   | 1.378       | 1.558    | 1.566  | 5.461  | 0/1248     | 0.000     | 0.000  |
| L23   | 0.266       | 0.018    | 1.565  | 5.429  | 277/1248   | 0.000     | 0.000  |

## Conclusions

1. **No leaves in L6–L22.** The top-16 right subspace of J_t rotates from one
   token to the next at ~90° — within noise of the random-subspace baseline
   (1.549 vs 1.565 rad at L6). The TZ "smooth sheet with boundaries" picture
   does not hold in the intermediate layers: there the J field is locally
   random as a subspace field.
2. **L23 (pre-final-RMSNorm / lm_head regime) is a true leaf.** Δ = 0.266 rad
   (15.3°), κ = 0.018 — ~300× below the random holonomy, flat profile across
   the whole document (thirds 0.261/0.269/0.268). 277 MAD-boundaries spread
   UNIFORMLY over the document (mean position 63.9; 78/107/92 per third) —
   local rough spots, not segment boundaries.
3. **Coexistence with exp27.** C_a(coherence of gauge-free J along a sample,
   8–41×F, L6–L23) is a norm/direction correlation of individual J rows;
   subspace rotation at random speed means the "sheet" is not an r-dimensional
   leaf in the intermediate layers — it is a thin filament: correlated
   direction statistics, random local frame. The leaf structure (r=16 sheet)
   appears only in the output regime.
4. **Depth gradient = untwisting.** κ falls 3.96 → 0.018 from L6 to L23
   (monotone: 3.96/3.47/2.36/2.17/1.56/0.018) — the local frame stops
   twisting with depth and rigidifies at L23. Δ stays at random level until
   L23, where it drops 5×. No mid-depth boundary: the change is a sharp
   transition at the last block, consistent with the L23 σ_d jump (pre-final-
   RMSNorm, exp27 findings).
5. **r-sweep stable.** Δ: 1.533/1.549/1.557 (L6) and 0.253/0.266/0.303 (L23)
   for r=8/16/32 — the verdict does not depend on r. κ random scales as ~√r
   (2.75/3.96/5.43 L6), data κ at L23 grows only 0.012/0.018/0.028 — flat
   field with r-growing numerical noise, not a curved one.
6. **No degenerate positions** (0 of 4032), no per-domain differences at L23
   (Δ 0.247–0.293 across domains).

## Open items
- L23 Δ=0.27 rad ≠ 0: the leaf is smooth, not rigid; a per-position profile
  (in JSON) shows no systematic growth toward the end of the sequence.
- Token types: not in the cache (TZ item skipped, recorded in params).
- Next (if pursued): check κ on U-side (left) subspaces — the field's
  functional side is K=128 tokens; right-side (hidden) rotation at random
  level may be a hidden-space artifact, not a statement about the map.
