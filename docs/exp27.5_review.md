# Диагноз

`_rotated_cosine` реализована через

```python
M2 = np.einsum('itk,jtl->ijkl', R, R)   # (n, n, K, K)
G  = np.einsum('ikd,jld->ijkl', A, A)   # (n, n, K, K)
Kraw = np.einsum('ijkl,ijkl->ij', M2, G)
```

Это `O(n² K³)` на построение `M2` + столько же на `G`, и создаёт промежуточные тензоры `(32, 32, 128, 128)` два раза. Отсюда 12.7 с.

Но математически:

```
tr(A_iᵀ R_iᵀ R_j A_j) = ⟨R_i A_i, R_j A_j⟩_F
```

т.е. достаточно один раз посчитать `R_i A_i` батч-матумножением `R @ A` и потом обычный `pair_cosine`. Комментарий в коде («O(n²KD) instead of rotating all A_i (O(nKD×D))») — неверен: стоимость старой версии `O(n²K³)`, новой `O(nK²D + n²KD)`. Для n=32, K=128, D~256 это ~2·10⁸ вместо ~3.4·10¹⁰ — ускорение в ~50–100 раз (зависит от того, насколько BLAS быстрая на батч-матумножении).

# Патч

Заменить `_rotated_cosine` целиком:

```python
def _rotated_cosine(A, rng, En=None):
    """Null cosine K_ij = <R_i A_i, R_j A_j>_F / (||A_i||_F ||A_j||_F)
    with R_i ~ Haar O(K) per sample (QR of Gaussian), LEFT on the token axis.

    Equivalent to rotating each A_i by R_i and taking the Frobenius inner
    product, since tr(A_i^T R_i^T R_j A_j) = <R_i A_i, R_j A_j>_F and
    ||R_i A_i||_F = ||A_i||_F.

    Complexity: O(n K^2 D + n^2 K D) via batched matmul (BLAS).
    Old version using einsum('itk,jtl->ijkl', R, R) was O(n^2 K^3).
    """
    n, K, D = A.shape
    R = np.linalg.qr(rng.standard_normal((n, K, K)))[0]   # (n,K,K) Haar
    RA = R @ A                                            # (n,K,D) batched
    if En is None:
        En = np.einsum('ikd,ikd->i', A, A)
    Kraw = RA.reshape(n, -1) @ RA.reshape(n, -1).T        # (n,n)
    return Kraw / np.sqrt(np.outer(En, En) + 1e-30)
```

Дополнительно: в `spectral_null` и `clustering_diagnostics` можно один раз посчитать `En` для исходного `J` и передавать его в `_rotated_cosine`, чтобы не пересчитывать нормы на каждой итерации (норма инвариантна к ортогональному повороту):

```python
def spectral_null(J, labels, rng, B=B_ROT):
    n = len(labels)
    m = mask_from_labels(labels, n)
    En = np.einsum('ikd,ikd->i', J, J)
    C0 = _pair_cosine(J)
    T0 = mean_offdiag(C0, m) - mean_offdiag(C0, ~m)
    cnt = 0
    for _ in range(B):
        Cn = _rotated_cosine(J, rng, En=En)
        ...
```

То же самое в блоке gap. Это экономит ещё ~10% на каждой итерации.

Ожидаемое время после патча: **0.2–0.5 с/итерация** (упор пойдёт в `np.linalg.qr` — 32 QR-разложения 128×128, ~10–30 мс каждое). Полный прогон `B_rot=100 + B_gap=100` на слой ≈ 40–100 с, шесть слоёв ≈ 5–10 мин.

Если хочется ещё быстрее — можно генерировать `R_i` не через QR, а через Givens-ротации или через `scipy.stats.special_ortho_group.rvs(K)`, но выигрыш там уже небольшой, QR — стандартный и корректный способ получить Haar-меру.

---

# Замечания по соответствию ТЗ

Проверил скрипт против спеки, в целом совпадает. Что бросилось в глаза:

1. **`gramian_cosine` использует `K_MAX` вместо фактического `K`.** Внутри `run_layer` срез идёт `[:K_MAX]`, т.е. `J.shape[1] == 128` при данном кэше, и всё работает. Но если однажды срез будет короче, `u = np.ones(K_MAX)` даст неверную размерность. Лучше `K = J.shape[1]; u = np.ones(K)/np.sqrt(K)`.

2. **Комментарий-докстринг `_rotated_cosine` вводит в заблуждение** («O(n²KD) per iteration instead of rotating all A_i (O(nKD×D))»). Настоящая старая стоимость — `O(n²K³)`. После патча докстринг переписать как в патче.

3. **`cophenet` обрабатывается странно:**
   ```python
   out = cophenet(Z, squareform(DG, checks=False))
   Dc = out[0] if np.ndim(out[0]) else out[1]
   ```
   `cophenet` всегда возвращает `(c, d)` где `c` — скаляр. Проще и надёжнее:
   ```python
   _, Dc = cophenet(Z, squareform(DG, checks=False))
   ```

4. **`spectral_null` пересчитывает `T0` и `E`**, хотя `C0` уже посчитан в `run_layer` (`gramian_cosine`). Экономия небольшая, но `_pair_cosine(J)` внутри нулевого теста — лишний `O(n²KD)`. Можно прокинуть готовый `C` и `En`.

5. **`entropy_quartile_test`** — корректно, но блок
   ```python
   lab = list(labels)
   for _ in range(B):
       pool = list(lab)
       rng.shuffle(pool)
       for i in range(n):
           lab[i] = pool[i]
   ```
   эквивалентен `rng.shuffle(lab)` на каждой итерации. Работает, но выглядит запутанно.

6. **`_rotated_Js` больше не нужен** — после патча `_rotated_cosine` делает ровно то же самое, только без лишнего выделения массива `(n,K,D)` (если не считать таковым `RA`). Можно удалить, чтобы не путаться.

7. **`token_gramian_cosine`:** `n, K = G.shape[0], G.shape[1]` — `K` не используется. Мелочь.

8. **Спек против реализации по блокам:**
    - A (perm, rot, half-split 70) — совпадает.
    - B (Spearman по 496 парам, энтропийные квартили 8/8/8/8, перм 1000) — совпадает.
    - C (average linkage, k∈2..8, silhouette, gap, ρ_NN, cophenetic, half-split silhouette 50) — совпадает.
    - D (g_uni, J_rest, ΔC) — совпадает.

Если пришлёшь `exp27_noise_floor.json` и хотя бы один тензор из `exp27_jcache.pt` — могу ещё и на реальных числах прогнать тайминг до/после патча, чтобы подтвердить ускорение.