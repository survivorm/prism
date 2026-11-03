"""operator_checks / cheap_invariants / behavior endpoints — ТЗ §IV.4 (now a
PASSING test: E6 closed) + boundary conditions §III.3/§III.9/§III.10."""

import numpy as np
import pytest

from prism.behavior import KL_lin, b_lin, b_mag, b_res, lq_coupling
from prism.cheap_invariants import A_alignment, c_pred
from prism.operator_checks import (O1, apply_P, c, cos_J_PJ, frac_K,
                                   operator_fraction, s)
from prism.p_algebra import P_matrix


def test_triple_grid_8_cells():
    """Патч v2 §1.9 (T4a): тождество тройки как сетка 8 ячеек
    p_slice ∈ {p_t, p_0} × aggregation ∈ {ratio_of_means, mean} × slice ∈ {all, alive}.
    Замкнувшаяся ячейка определяет семантику таблицы; median-ячейки замыкаться
    НЕ обязаны (медиана отношения ≠ отношение сумм) — негативный контроль."""
    import json
    from pathlib import Path
    rng = np.random.default_rng(7)
    T, K, D = 24, 16, 8
    J = rng.standard_normal((T, K, D))
    logits = rng.standard_normal((T, K)) * 2.0
    p_t = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    p_0 = np.broadcast_to(p_t[0], p_t.shape)          # fixP: один p на все позиции
    alive = np.ones(T, bool); alive[T // 3:] = False  # произвольный подсрез
    def per_pos(p):
        PJ = apply_P(J, p)
        eJ = (J ** 2).sum((-1, -2))
        c_num = np.einsum("tkd,tkd->t", J, PJ)
        s2_num = (PJ ** 2).sum((-1, -2))
        f_num = ((J - PJ) ** 2).sum((-1, -2))
        return c_num, s2_num, f_num, eJ

    nclosed = 0
    for name, p in (("p_t", p_t), ("p_0", p_0)):
        c_num, s2_num, f_num, eJ = per_pos(p)
        for sl, m in (("all", np.ones(T, bool)), ("alive", alive)):
            # ratio_of_means = Σnum/Σden (НЕ среднее отношений)
            cr = c_num[m].sum() / eJ[m].sum()
            s2r = s2_num[m].sum() / eJ[m].sum()
            fr = f_num[m].sum() / eJ[m].sum()
            assert abs(fr - (1 - 2 * cr + s2r)) < 1e-12, f"rom {name}/{sl}"
            # mean = среднее попозиционных ОТНОШЕНИЙ (каждое со своим знаменателем)
            cm = (c_num[m] / eJ[m]).mean()
            s2m = (s2_num[m] / eJ[m]).mean()
            fm = (f_num[m] / eJ[m]).mean()
            assert abs(fm - (1 - 2 * cm + s2m)) < 1e-12, f"mean {name}/{sl}"
            nclosed += 2
            # негативный контроль: медиана отношений не замыкает на generic J
            med = abs(np.median(f_num[m] / eJ[m])
                      - (1 - 2 * np.median(c_num[m] / eJ[m])
                         + np.median(s2_num[m] / eJ[m])))
            assert med > 1e-9, f"median closed {name}/{sl} — вырожденные данные?"
    assert nclosed == 8
    # g_uni как контрольная точка (патч v2 §1.9): не зависит от p, два независимых
    # закоммиченных пайплайна (E-39k и E-39-suppl) обязаны совпадать.
    root = Path(__file__).resolve().parents[1]
    f1, f2 = (root / "results/exp39k_gauge_aware_importance.json",
              root / "results/exp39_supp_three_objects.json")
    if f1.exists() and f2.exists():
        a = json.load(open(f1))["layers_diag"]
        b = json.load(open(f2))["layers"]
        for L in ("23", "18"):
            ga = a[L]["g_uni_uniform_energy_share"]
            gb = b[L]["g_uni_corpus"]
            assert abs(ga - gb) / gb < 1e-4, f"g_uni checkpoint L{L}: {ga} vs {gb}"


@pytest.fixture()
def data():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(32) * 2.0
    p = np.exp(x) / np.exp(x).sum()
    J = rng.standard_normal((8, 32, 16))
    return p, J


def test_triple_identity_one_aggregation(data):
    """§IV.4 — the E6 triple. The identity was NEVER wrong on one aggregation;
    the phantom s²<0 came from mixing an alive-slice c with an all-positions
    median frac. Closed 2026-11-03; test locks the ONE-AGGREGATION semantics."""
    p, J = data
    got = frac_K(J, p)
    want = 1.0 - 2.0 * c(J, p) + s(J, p) ** 2
    assert abs(got - want) < 1e-12
    assert abs(cos_J_PJ(J, p) - c(J, p) / s(J, p)) < 1e-12   # cos ≠ c (§III.3)
    # and per-position the identity also closes, position by position.
    # NOTE: per_position returns RAW energy ratios (sqrt is a property of the
    # aggregate), so s enters as s, not s**2 — the square is already taken.
    assert np.abs(frac_K(J, p, aggregation="per_position")
                  - (1 - 2 * c(J, p, aggregation="per_position")
                     + s(J, p, aggregation="per_position"))).max() < 1e-10


def test_O1_eq_s2_and_energy_vs_norm(data):
    p, J = data
    assert abs(O1(J, p) - s(J, p) ** 2) < 1e-12     # §III: norm vs energy = square
    assert O1(J, p) < s(J, p)                        # small fractions: E < ‖·‖


def test_apply_P_matches_dense_product(data):
    p, J = data
    Pd = P_matrix(p)
    assert np.abs(apply_P(J, p) - np.stack([Pd @ Jt for Jt in J])).max() < 1e-10


def test_operator_fraction_matches_iso_on_iid(data):
    """iid-null consistency: ‖PX‖²/‖X‖² on large isotropic X → ‖P‖²_F/K."""
    p, _ = data
    rng = np.random.default_rng(1)
    Pd = P_matrix(p)
    X = rng.standard_normal((4000, 32, 64))
    got = operator_fraction(Pd, X)
    iso = float((Pd ** 2).sum() / 32)
    assert got == pytest.approx(iso, rel=0.02)


def test_c_pred_and_A(data):
    p, J = data
    cp = c_pred(p)
    assert 0.0 < cp <= 1.0
    assert A_alignment(J, p) == pytest.approx(c(J, p) / cp, rel=1e-12)


def test_lq_coupling_and_eps3_order():
    """Патч v2 §1.1: b_res = ε³⟨Jv,H[v,v]⟩ + O(ε⁴). Квадратичное отображение с
    известной Hessian-формой: масштаб b_res ~ ε³ (не ε⁴), коэффициент = lq_coupling."""
    rng = np.random.default_rng(11)
    A = rng.standard_normal((6, 5))
    B = rng.standard_normal((6, 5, 5))
    B = 0.5 * (B + B.transpose(0, 2, 1))                # симметричная билинейная
    x0 = rng.standard_normal(5)
    v = rng.standard_normal(5); v /= np.linalg.norm(v)
    Jx0 = A + np.einsum("ijk,k->ij", B, x0)            # Jacobian at x0
    Jv = Jx0 @ v
    Hvv = np.einsum("ijk,j,k->i", B, v, v)
    lq = lq_coupling(Jv, Hvv)
    assert abs(lq) > 1e-6, f"случайно нулевая связка: {lq:g}"

    def f(x):
        return A @ x + 0.5 * np.einsum("ijk,j,k->i", B, x, x)

    def bres(eps):
        dz = f(x0 + eps * v) - f(x0)
        return b_res(b_mag(dz), b_lin(Jx0, eps, v))

    e1, e2 = 1e-2, 1e-3
    r1, r2 = bres(e1), bres(e2)
    assert (r1 > 0) == (r2 > 0)                          # знак ε³-члена устойчив
    assert abs(abs(r1 / r2) / 1000.0 - 1.0) < 0.05, "b_res масштабируется как ε³"
    # редукционный контроль: b_res/(ε²‖Jv‖²) → 0 быстрее, чем ε¹·константа
    assert abs(r2) / (e2 ** 2 * (Jv ** 2).sum()) < abs(r1) / (e1 ** 2 * (Jv ** 2).sum()) * 1.05
    # лимита коэффициента: b_res/ε³ → lq с относительной ошибкой O(ε)
    assert abs(r2 / e2 ** 3 - lq) / abs(lq) < 0.05


def test_b_mag_is_square_and_b_res(data):
    _, J = data
    dz = J[0, :, 0]
    assert b_mag(dz) == pytest.approx(float(dz @ dz), rel=1e-14)   # square, not ‖·‖₂
    assert abs(b_res(b_mag(dz), 2.0) - (b_mag(dz) - 2.0)) < 1e-15


def test_b_lin_corollary_v1(data):
    _, J = data
    Jm = J[0]
    _, S, Vt = np.linalg.svd(Jm, full_matrices=False)
    assert b_lin(Jm, 0.05, Vt[0]) == pytest.approx(0.05 ** 2 * S[0] ** 2, rel=1e-9)


def test_KL_lin_fisher_factor_not_identity(data):
    """§III.10: KL_lin uses F = diag(p) − ppᵀ, so it must equal ½ε²Var_p(Jv)
    and NOT ½ε²‖Jv‖² (bare) — and vanish for Jv ∝ 1 (pure gauge)."""
    p, J = data
    Jm = J[0]
    rng = np.random.default_rng(2)
    v = rng.standard_normal(Jm.shape[1])
    Jv = Jm @ (v / np.linalg.norm(v))
    manual = 0.5 * 0.03 ** 2 * (float(np.dot(p, Jv ** 2)) - float(np.dot(p, Jv)) ** 2)
    assert KL_lin(Jm, p, 0.03, v) == pytest.approx(manual, rel=1e-12)
    f = np.arange(1.0, Jm.shape[1] + 1.0)                   # J[i, :] = f ⇒ Jv ∝ 1
    Jflat = np.tile(f, (Jm.shape[0], 1))
    assert KL_lin(Jflat, p, 0.03, v) == pytest.approx(0.0, abs=1e-10)
