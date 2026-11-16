#!/usr/bin/env python
"""E-52f: контроли τ_amp² (ревью E52, пункты §3/§5). Без модели, чистый MC.

Вопросы:
  Q1 геодезический контроль: гладкая S-траектория (линейный дрейф + tiny noise)
     — формула (a+b−c)/a обязана давать ~0 (иначе метрика сломана).
  Q2 белый контроль: iid шаги (гаусс) — E[(a+b−c)/a] = константа размера в
     R^16 (2−√2 — это размер-2 формула; точное значение — MC).
  Q3 перескоки: σ-упорядоченный случайный блок с тесными парами (перескок) —
     насколько τ_amp² выше «честного» значения того же пути без сортировки?
  Q4 fp16-шум: тот же геодезический путь, квантованный в fp16 — сдвиг τ_amp².
"""
import numpy as np

R, T, N = 16, 96, 4000
rng = np.random.default_rng(52)


def tau_amp(S):  # S: [T,R]
    d = S[1:] - S[:-1]
    a = np.linalg.norm(d[:-1], axis=1)
    b = np.linalg.norm(d[1:], axis=1)
    c = np.linalg.norm(S[2:] - S[:-2], axis=1)
    ok = a > 1e-12
    return float(np.median(((a + b - c) / a)[ok]))


# --- Q1 геодезика: S_t = s0 + v*t*(1+шум масштаба), tiny пертурбация вдоль пути
v = rng.normal(size=R); s0 = np.sort(rng.uniform(1, 10, R))[::-1]
geo = [tau_amp(s0[None, :] * 0 + (s0 + v * np.arange(T)[:, None] * 0.05) + rng.normal(0, 1e-3, (T, R)))
       for _ in range(200)]
print("Q1 геодезика (линейный дрейф + малый шум):  τ_amp2 = %.4f ± %.4f" % (np.mean(geo), np.std(geo)))

# --- Q2 белый шум: iid гауссовы шаги (тот же размер)
white = [tau_amp(rng.normal(size=(T, R))) for _ in range(N)]
print("Q2 белый iid (R=16):                        τ_amp2 = %.4f ± %.4f" % (np.mean(white), np.std(white)))

# --- Q2b случайное блуждание (сумма iid шагов — траектория, не позиция-iid)
rw = [tau_amp(np.cumsum(rng.normal(size=(T, R)), axis=0)) for _ in range(N)]
print("Q2b белое блуждание:                        τ_amp2 = %.4f ± %.4f" % (np.mean(rw), np.std(rw)))

# --- Q3 перескоки: два близких уровня σ меняются местами случайно каждый шаг
base_sig = np.sort(rng.uniform(4, 6, R))[::-1]
mix = [tau_amp(np.sort(base_sig + rng.normal(0, 0.3, (T, R)), axis=1)) for _ in range(1000)]
nomix = [tau_amp(base_sig + rng.normal(0, 0.3, (T, R))) for _ in range(1000)]
print("Q3 тесные σ с сортировкой (перескок):       τ_amp2 = %.4f" % np.mean(mix))
print("Q3' те же данные без сортировки:            τ_amp2 = %.4f" % np.mean(nomix))

# --- Q4 fp16: геодезика, квантованная в fp16 (уровень шума реальных трасс)
f16 = [tau_amp((s0 + v * np.arange(T)[:, None] * 0.05 + rng.normal(0, 1e-3, (T, R))).astype(np.float16).astype(np.float64))
       for _ in range(200)]
print("Q4 геодезика после fp16 round-trip:         τ_amp2 = %.4f" % np.mean(f16))

# --- Q5 реалистичный микс: дрейф + шум + тесные уровни, С сортировкой (как в e52e)
mixx = []
for _ in range(1000):
    S = s0 + v * np.arange(T)[:, None] * 0.05 + rng.normal(0, 0.05, (T, R))
    mixx.append(tau_amp(np.sort(S, axis=1)))
print("Q5 дрейф+шум+сортировка:                    τ_amp2 = %.4f" % np.mean(mixx))

# --- Q6 сеточный артефакт: та же смесь (Q5), квантованная symmetric RTN на lev уровней
#     (ответ на 4B rtn2: τ_amp² −0.063 — может, σ(t) просто залипает на сетке)
for lev, tag in ((1, "2-bit (lev=1)"), (7, "3-bit"), (127, "8-bit")):
    q = []
    for _ in range(1000):
        S = s0 + v * np.arange(T)[:, None] * 0.05 + rng.normal(0, 0.05, (T, R))
        sc = np.abs(S).max(axis=0, keepdims=True).clip(1e-9) / lev
        S = np.clip(np.round(S / sc), -lev, lev) * sc
        q.append(tau_amp(np.sort(S, axis=1)))
    print("Q6 сетка %s: τ_amp2 = %.4f (было %.4f)" % (tag, np.mean(q), 0.6672))
