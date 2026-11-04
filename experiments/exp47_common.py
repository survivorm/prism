"""E-47 common: численная кинематика листа L23 (общое ядро для exp47a/b/c/d).

Математика (полный вывод и верификация — docs/leaf_dynamics.md §2,§4):

    J(h) = r(h) * M (I - rho * u u^T),   u = h/||h||,  rho = ||h||^2/(||h||^2 + D*eps),
    M    = W_head[U128] * diag(1 + w_norm),          # Qwen RMSNorm: (1+w), НЕ w
    r(h) = rsqrt(mean(h^2) + eps)                     # скаляр -> на V не влияет

Вся зависимость V_16(h) от позиции — ЧЕРЕЗ ОДНУ u (ранг-1 наклон). Отсюда:
  * V инвариантна к масштабу h;
  * delta J от поворота u на du ⟂ u: -rho*M(du u^T + u du^T) — ранг 2;
  * скорость листа s_t пропорциональна скорости потока a_t = angle(u_t,u_{t+1})
    с резольвентным множителем 1/(sigma_i^2 - sigma_j^2) (док. gap16 мал ⇒ рама жёсткая).

Численная гигиена (урок верификации 2026-11-05): собственные векторы Gram берём от
JJ^T размера [K,K] (K=128), а не J^T J [D,D]: второе удваивает обусловленность на блоке D
и в fp32 даёт ложный пол ~1e-3 рад на масштабе s_t. Путь: eigh(JJ^T) -> U, sigma^2;
правые векторы V = J^T U[:,top] / sigma. Пол метода в fp64: 1.4e-6 рад (проверено).

Все функции чистые, torch-only (scipy.linalg.schur только для углов Шюра).
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------- Jacobian / V

def head_M(head_W: torch.Tensor, g_norm: torch.Tensor) -> torch.Tensor:
    """M = W_head[U128] diag(g), g = 1+w_norm. fp64 — для кинематики малых углов."""
    return (head_W.double() * g_norm.double().unsqueeze(0)).contiguous()


def J_analytic(A: torch.Tensor, h: torch.Tensor, eps: float) -> tuple[torch.Tensor, float, torch.Tensor]:
    """J(h) без скаляра r (на V не влияет): J = A (I - rho u u^T) = A - rho (A u) u^T.

    Возвращает (J [K,D], rho, u [D]). A = M (fp64). Одна внешняя произведение — O(K D).
    """
    hh = float(h @ h)
    D = h.numel()
    rho = hh / (hh + D * eps)
    u = h / math.sqrt(hh)
    J = A - rho * torch.outer(A @ u, u)
    return J, rho, u


def V16_from_J(J: torch.Tensor, r: int = 16):
    """Топ-r правых сингулярных подпространств через Gram JJ^T [K,K] (fp64).

    V = J^T U_k / sigma_k (U_k — собственные векторы JJ^T с бОльшими lmbda=sigma^2).
    Возвращает (V [D,r], sigma [K] — все, по порядку убывания).
    """
    G = J @ J.T
    G = 0.5 * (G + G.T)
    lam, U = torch.linalg.eigh(G)          # по возрастанию
    idx = torch.argsort(lam, descending=True)
    lam, U = lam[idx], U[:, idx]
    sig = lam.clamp_min(1e-30).sqrt()
    V = (J.T @ U[:, :r]) / sig[:r]
    return V.contiguous(), sig


def procrustes_R(Va: torch.Tensor, Vb: torch.Tensor) -> torch.Tensor:
    """Ортогональный транспорт рамы Va->Vb: R = polar(Va^T Vb) = U W^T (SVD 16x16).

    R определён корректно, когда столбцы — УПОРЯДОЧЕННЫЕ базисы (наши V из eigh — да).
    Знаки столбцов SVD-базисов тут не мешают: Va->Va даёт точную I только при совпадении
    знаков; поэтому дальше используем только инварианты R (углы Шюра), которые от знака
    базиса НЕ зависят (подправка знаком вписана в сам полярный множитель).
    """
    U, _, Wt = torch.linalg.svd(Va.T @ Vb, full_matrices=False)
    return U @ Wt


# ---------------------------------------------------------------- инварианты R

def schur_angles(R: torch.Tensor):
    """Главная часть log(R) для R in SO(16): углы 2x2-блоков вещественного Шюра.

    Вещественный Шур R = Q T Q^T: диагональные 1x1 блоки = ±1 (угол 0 или pi),
    2x2 блоки [[a,b],[c,d]]bc<0: собственные a ± i sqrt(|bc|) => theta = atan2(|b|,a)
    (берём |theta| в [0,pi)). Возвращает (theta sorted desc [m], Q — векторы Шюра,
    bsize — размеры блоков): Q[:, k] — ИНВАРИАНТНОЕ подпространство блока k
    (для 2-блока — плоскость вращения с наибольшим |theta|, если блоки отсортированы).
    """
    import numpy as np
    from scipy.linalg import schur

    T, Q = schur(R.detach().cpu().numpy(), output="real")
    n = T.shape[0]
    thetas: list[float] = []
    bsize: list[int] = []
    i = 0
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > 1e-10:
            # собственный блок Шюра: [[a,b],[c,d]], bc<0, собств. = (a+d)/2 ± i*sqrt|bc|
            re = 0.5 * (float(T[i, i]) + float(T[i + 1, i + 1]))
            im = math.sqrt(abs(float(T[i, i + 1]) * float(T[i + 1, i])))
            th = math.atan2(im, re) % math.pi
            thetas.append(th)
            bsize.append(2)
            i += 2
        else:
            th = 0.0 if float(T[i, i]) > 0 else math.pi
            thetas.append(th)
            bsize.append(1)
            i += 1
    order = sorted(range(len(thetas)), key=lambda k: -thetas[k])
    # переставляем векторы Шюра блоками в порядке убывания угла
    cols: list[int] = []
    start: list[int] = []
    s = 0
    for bs in bsize:
        start.append(s)
        s += bs
    for k in order:
        cols.extend(range(start[k], start[k] + bsize[k]))
    Q2 = torch.from_numpy(Q[:, cols])
    return torch.tensor([thetas[k] for k in order], dtype=torch.float64), Q2, [bsize[k] for k in order]


def frame_kinematics(Va: torch.Tensor, Vb: torch.Tensor):
    """Один шаг кинематики ЛИСТА (подпространства) — базис-инвариантные величины.

    ⚠ УРОК (найден первым прогоном P0 2026-11-05): углы Шюра одного шага Procrustes
    R = polar(Va^T Vb) НЕ инвариантны к внутренним базисам рам: Va->Va@Q1, Vb->Vb@Q2
    дают R -> Q1^T R Q2, собственные углы при этом меняются. Внутренний базис eigh/SVD
    при плотном кластере сингулярных чисел выбирается произвольно — одношаговый R мерил
    «рассогласование базисов» (s~π, e~π/2 на ровных данных). Инвариантен ТОЛЬКО цикличный
    произведение (голономия κ — потому E-28 и корректен). Для одного шага — ГЛАВНЫЕ УГЛЫ
    подпространств (Грассманиан):
        Va^T Vb = X diag(cos Θ) Y^T ;  s = ||Θ||_2,  p = Θ_max/s,
        собственная 2-плоскость в D-координатах: span{Va X_1, Vb Y_1} (инвариантна).
    В меловом limite Θ = углы вращения, так что семантика «скорость/планарность» сохранена.
    Возвращает (s, p, P [D,2], Θ [r] desc).
    """
    M = Va.T @ Vb
    X, c, Yh = torch.linalg.svd(M, full_matrices=False)
    Y = Yh.T
    Th = torch.acos(c.clamp(-1.0 + 1e-12, 1.0 - 1e-12))
    # svd отдаёт cos Θ по убыванию ⇒ Θ по ВОЗРАСТАНИЮ; доминирующий — последний
    order = torch.argsort(Th, descending=True)
    Th = Th[order]
    i0 = int(order[0])
    s = float(torch.sqrt((Th * Th).sum()))
    p = float(Th[0] / s) if s > 1e-12 else float("nan")
    P = torch.column_stack([Va @ X[:, i0], Vb @ Y[:, i0]]).double().contiguous()
    return s, p, P, Th


def align_to_u_plane(P: torch.Tensor, u0: torch.Tensor, u1: torch.Tensor) -> float:
    """Худший главный угол (рад) между 2-плоскостью P и span{u0,u1}; nan если span вырожден.

    Для 2-плоскости в D>4 корректнее брать ПЛОЩАДНУЮ согласованность: здесь худший
    главный угол — если P и span расходятся хотя бы в одном направлении, угол велик.
    cos главного угла = сингулярные числа P^T Q_U; acos(min) — угол самого «непокрытого»
    направления: 0 = плоскости совпадают.
    """
    B = torch.column_stack([u0, u1])
    svB = torch.linalg.svdvals(B)
    if float(svB.min()) < 1e-8 * float(svB.max()):
        return float("nan")            # u0 || u1: шага нет, выравнивание не определено
    Qb = torch.linalg.qr(B)[0]
    svP = torch.linalg.svdvals(P)
    if float(svP.min()) < 1e-6 * float(svP.max()):
        return float("nan")            # Θ мал: «плоскость» выродилась в прямую, QR бы соврал
    Qp = torch.linalg.qr(P)[0]         # P — два ПРИНЦИПАЛЬНЫХ вектора (угол между ними = Θ),
                                       # не ортонормированы; без QR svd(P^T Q) врал (>1)
    sv = torch.linalg.svdvals(Qp.T @ Qb)
    return math.acos(float(sv.min().clamp(max=1.0 - 1e-12)))


def Q_orth(X: torch.Tensor) -> torch.Tensor:  # noqa: N802 (имя из формул)
    return torch.linalg.qr(X)[0]


# ---------------------------------------------------------------- скалярные ряды

def u_series(hs: torch.Tensor) -> torch.Tensor:
    """По [T,D] -> единичные u_t (fp64)."""
    H = hs.double()
    return H / H.norm(dim=-1, keepdim=True)


def angle_series(u: torch.Tensor) -> torch.Tensor:
    """a_t = angle(u_t, u_{t+1}) [T-1]."""
    c = (u[:-1] * u[1:]).sum(-1).clamp(max=1.0 - 1e-12)
    return torch.acos(c)


def omega_geodesic(u: torch.Tensor) -> torch.Tensor:
    """Геодезическая кривизна [M-3]: |A01 + A12 - A02| по тройкам (T-2)."""
    def A(x, y):
        return torch.acos((x * y).sum(-1).clamp(max=1.0 - 1e-12))
    return (A(u[:-2], u[1:-1]) + A(u[1:-1], u[2:]) - A(u[:-2], u[2:])).abs()


def band_power(x: torch.Tensor, lo: int = 4, hi: int = 32) -> float:
    """EXCESS спектральной плотности в полосе периодов [lo,hi] против вне-полосы.

    =1 белый шум; ≫1 — периодическая компонента на этих периодах (детектор орбиты).
    Доля-в-полосе не годится: полоса покрывает почти весь спектр короткого ряда, и даже
    чистая синусоида даёт ≈1. Поэтому делим среднюю плотность В полосе на среднюю ВНЕ
    (пол с 1e-6 от общей плотности — защита от деления на ноль у чистой синусоиды).
    Полоса по умолчанию 4..32: ниже 4 — токсично (токенизированные повторы букв),
    выше 32 — ловим тренды вместо орбит."""
    x = x - x.mean()
    if x.numel() < 16 or float(x.pow(2).sum()) < 1e-20:
        return float("nan")
    f = torch.fft.rfft(x.double())
    pw = (f.real ** 2 + f.imag ** 2)[1:]        # bin k <-> период N/k, DC выброшен
    n = x.numel()
    k_lo, k_hi = max(1, n // hi), min(pw.numel() - 1, n // lo)
    mask = torch.zeros(pw.numel(), dtype=torch.bool)
    mask[k_lo:k_hi + 1] = True
    d_in = float(pw[mask].mean())
    d_out = float(pw[~mask].mean())
    d_tot = float(pw.mean())
    return d_in / max(d_out, 1e-6 * d_tot)


def autocorr_peak(x: torch.Tensor, p: int, base_slice=slice(5, None)) -> float:
    """ACF(x)[p] относительно медианы |ACF| хвоста — пик/база (детектор периода p).

    ACF нормирована на полную энергию (смещённая оценка — ряд короткий, monotone затухание
    вычитается базой медианы хвоста)."""
    xf = (x - x.mean()).double()
    if xf.numel() < 3 * p:
        return float("nan")
    n = xf.numel()
    e0 = float((xf * xf).sum())
    if e0 < 1e-20:
        return float("nan")
    acf = torch.tensor([float((xf[: n - k] * xf[k:]).sum() / e0)
                        for k in range(1, min(n - 1, 3 * p + 1))])
    base = acf[base_slice].abs().median()
    return float(acf[p - 1] / base) if base == base and float(base) > 1e-9 else float("nan")


# ---------------------------------------------------------------- детектор петель

def find_loops(tokens: list[int], max_p: int = 32, min_reps: int = 3):
    """Детерминированный детектор повторов (пререгистрация docs/leaf_dynamics.md §5.2).

    Эпизод: позиция j и ФУНДАМЕНТАЛЬНЫЙ период p (минимальный из подходящих), при котором
    блок tokens[j:j+p] повторяется k>=min_reps раз подряд вперёд ( maximal'ный прогон).
    onset = j + (min_reps-1)*p — начало ТРЕТЬЕГО повтора («петля объявила себя»). Эпизоды
    с одинаковым onset склеиваются; в выход кладётся максимальный прогон по каждому onset.
    Возвращает [(onset, start=j, end, p, k)].
    """
    n = len(tokens)
    raw: dict[int, tuple[int, int, int, int]] = {}
    for j in range(n - 2):
        for p in range(2, max_p + 1):
            block = tokens[j: j + p]
            if len(block) < p:
                break
            k = 1
            e = j + p
            while e + p <= n and tokens[e: e + p] == block:
                k += 1
                e += p
            if k >= min_reps and p * (k - 1) >= 6:   # подавление тривиальных «aa», «aab aab»
                cand = (j, e, p, k)
                prev = raw.get(j)
                if prev is None or p * k > prev[2] * prev[3]:
                    raw[j] = cand
    out = sorted((j + (min_reps - 1) * cand[2], j, cand[1], cand[2], cand[3])
                 for j, cand in raw.items())
    # склейка: эпизод, начинающийся внутри более длинного с тем же периодом, отбрасывается
    merged: list[tuple[int, int, int, int, int]] = []
    for ep in sorted(out, key=lambda x: (x[1], -x[4] * x[3])):
        if merged and merged[-1][1] <= ep[1] < merged[-1][2] and merged[-1][3] == ep[3]:
            continue
        merged.append(ep)
    return merged
