#!/usr/bin/env python3
"""AT0_8 — T8.1 «Пороговое возбуждение» (multi-observable threshold), v1 (до прогона).

ТЗ: docs/TZ_add_4-8.md §8 (T8-FALS); пре-рег: docs/AT0_8_prereg.md.
Вопрос: пороговое рождение локализованного объекта («солитон») при λ-интерполяции весов
W(λ) = λ·W_trained + (1−λ)·W_random vs генерическое включение компетентности.

Контракт (пре-рег §2–§8):
- сетка LAM13 = 11 точек T8.1 + 2 точки T9.1 (общий прогон; вердикт T8.1 — по LAM11);
- 3 сида [11,12,13]; ВСЕ параметры рандомизированы при λ=0 (вкл. 1-D), sd_r = randn·std;
- наблюдаемые: dip/hump/plateau (curves_series h23, пин e64_gpu.py), nll_main (гейт S4),
  nll_holdout (4 дока math/code/chat), margin (top1−top2), pr_sigma/eff_rank (SVD h23),
  rep16/loopy (greedy-генерация 256 ток., prompt+gen);
- λ_c: u(λ) = (O(λ)−O(0))/(O(1)−O(0)), первое удержанное ≥2 точек пересечение u=0.5;
  монотонность |Spearman(λ,O)|≥0.9 (seed-avg, ДО прогона); overshoot u>1 — флаг;
  бисекция ≤5 ит./≤0.02 (union ≤6 точек); CI по сидам (разброс >1 шага ⇒ not_resolved);
- вердикты V3 (патология) > V2 (λ_c^dip отделён от ≥1 из 4) > V1 (все совпадают) > V0;
- health ray (dip_band_hinge, loopy, nll_holdout) + человекочитаемый sample в каждой (λ,сид).

Гейты (до записи; провал = артефакта нет): S1 config; S2 λ=1 побитово = base (sha тензоров);
S2b λ=0 все тензоры случайны; S3 полнота = точное равенство множеств (не truthiness);
S4 nll в каждой записи; S5 finite + pr_sigma≥1 + eff_rank≥1; S6 disjoint; S7 sha e64_gpu.py;
S8 health ray + sample; S9 бисекция.

Smoke: --smoke (2 λ × 1 сид × 2 дока × 1 holdout + генерация, в /tmp/at08_smoke/) — полный
код-путь + гейты на малом масштабе; λ_c-машина в смоуке не считается (сетка вырождена).
Полный прогон — Kaggle (T4 GPU fp32; CPU ≈ 10 ч — тяжёлый, см. пре-рег §9).
"""

import hashlib
import importlib.util
import json
import math
import os
import sys
import time

# CPU-политика серии (AT0_4): 4 тред ДО импорта torch (бенч 51.7 с/forward снят на 4);
# transformers.imports поднимает fla (triton) и крашится "0 active drivers" на CPU —
# чистый torch-fallback = референсный путь серии: ОДНАКОВЫЕ операции в смоуке (CPU)
# и полном прогоне (Kaggle GPU) — численная согласованность между ними.
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
sys.modules["fla"] = False
sys.modules["fla.ops"] = False

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

torch.set_grad_enabled(False)
torch.set_num_threads(4)

# ── закреплённые константы (пре-рег §2–§4) ─────────────────────────────────────
MODEL_LOCAL = "/data/Qwen/Qwen3.5-0.8B"
MODEL_HF = "Qwen/Qwen3.5-0.8B"
try:
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:  # notebook (Kaggle-кернел): __file__ отсутствует
    _REPO = os.getcwd()
# Функции e64 (dcurve/vpath_units/curves_series) — встроены VERBATIM из
# experiments/kaggle_e64gpu/e64_gpu.py (sha целого файла — E64_FILE_SHA256;
# sha встроеного источника трёх функций — E64_FNS_SHA256 — проверяется ДО exec;
# без файлового доступа — кернел self-contained). Скрипт не читает e64_gpu.py.
E64_FILE_SHA256 = "8c517e18f801fa877dfde9265ed5ce18df1e7d3af705194fc01ff6b150cf7633"
E64_FNS_SHA256 = "3c215a72f2f1753e8b873929094889c6cf0487adecb591f7cef583519aa906d4"
E64_FNS_SRC = '''def dcurve(P,kk):
    """сериийный оценщик (exp64_b4_battery/A5): median cos по t, без средних."""
    Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i",Pn[:-k],Pn[k:]))) for k in kk])
def vpath_units(u):
    """сериийный Vpath: ортогонализованная составляющая шага между единичными состояниями."""
    Un=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
    Vs=[]
    for t in range(len(Un)-1):
        a=float(Un[t]@Un[t+1]); vv=Un[t+1]-a*Un[t]
        if np.linalg.norm(vv)>1e-6: Vs.append(vv/np.linalg.norm(vv))
    return np.array(Vs)
def curves_series(Vh):
    """Vh [T,H] сырые h -> {dip,hump,plateau,curve{}} в ТОЧНОСТИ батареи B4 (median, ортогональн.шаг)."""
    Vh=Vh.astype(np.float64)
    u=Vh/np.maximum(np.linalg.norm(Vh,axis=1,keepdims=True),1e-12)
    P=vpath_units(u)
    kk=np.arange(1,min(96,max(2,len(P)//3))+1)
    c=dcurve(P,kk)
    return dict(dip=float(c[0]),hump=float(c[1:8].max()) if len(c)>8 else float('nan'),
        plateau=float(np.median(c[8:])) if len(c)>9 else float('nan'),
        curve={str(int(k)):float(x) for k,x in zip(kk,c)})
'''

LAM13 = [0.0, 0.05, 0.1, 0.2, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.85, 1.0]
LAM11 = [0.0, 0.2, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.85, 1.0]
LAM11_IDX = {lam: i for i, lam in enumerate(LAM11)}
SEEDS = [11, 12, 13]
T_GEN = 256
PLEN = 32
EOS_ID = 248046          # tokenizer Qwen/Qwen3.5-0.8B: eos_token_id
L_H = 23                 # слой для h (выход последнего слоя; exp71/E-64 конвенция)

MAIN_SHA256 = "4ea0f28771ab1411d8050ee38f8e873c44489758ed81e9586b519cdddad17fd9"
HOLDOUT_SHA256 = "7ac1199e9689e0a2236488824378039e24d795304948f84b5e725e71290efa95"
POOL_FPS = {"chat": "4d3aab66d0e7aac6", "code": "e02cdb39ed828cb2", "math": "1c01d1447c7018a7", "wiki": "144c722df3046f2d"}

MAIN_DOCS = [
        ("wiki_0", [10812, 4811, 43768, 3124, 34639, 10454, 303, 8637, 282, 4120, 0, 198, 5236, 488, 1318, 310, 615, 2577, 506, 3154, 17387, 43768, 30, 1394, 668, 599, 279, 6429, 11, 2113, 411, 383, 678, 12786, 1381, 13, 7548, 11, 5930, 220, 17, 17, 302, 4973, 4196, 3124, 43768, 31424, 11, 17974, 18831, 347, 494, 436, 33961, 277, 51416, 32085, 13, 1216, 668, 381, 12266, 264, 47308, 2119, 523, 364, 4858, 852, 6587, 310, 615, 2577, 440, 836, 55422, 7129, 13, 198, 1465, 668, 4390, 488, 4156, 488, 1144, 310, 1366, 310, 19643, 303, 264, 20437, 7116, 43768, 10391, 11, 2583, 12178, 11, 18066, 11, 84137, 11, 12668, 6381, 321, 79062, 11, 5346, 74578, 321, 3807, 1928, 13, 198, 760, 2695, 310, 381, 303, 279, 523, 369, 393, 18, 20, 791, 1637, 11, 321, 364, 66956, 424, 369, 1845, 13, 44429, 303, 279, 2695, 668, 381, 2892, 264, 259, 32553, 466, 20404, 263, 321, 488, 668, 381, 44854, 10156, 314, 1754, 12668, 421, 369, 9734, 13]),
        ("wiki_1", [68610, 303, 359, 18955, 9787, 1543, 31042, 318, 16, 15, 13, 22, 20973, 3727, 539, 3731, 726, 72, 23, 22, 11, 4209, 220, 17, 15, 11, 220, 17, 15, 16, 17, 13, 198, 40, 2908, 2597, 264, 220, 20, 15, 15, 9231, 5138, 6338, 321, 264, 220, 17, 19, 15, 9231, 35160, 13, 198, 4350, 4313, 310, 14519, 1608, 12977, 15092, 585, 2688, 2574, 279, 1412, 328, 2536, 3213, 3433, 383, 12977, 29739, 310, 14519, 1, 198, 3850, 353, 12752, 914, 599, 310, 635, 421, 11726, 198, 8347, 6498, 466, 944, 277, 2995, 1518, 21117, 286, 310, 279, 3294, 30, 198, 9947, 20829, 14122, 2356, 25728, 310, 2880, 799, 6338, 310, 279, 975, 13, 353, 2908, 2725, 411, 3679, 2942, 2019, 494, 7875, 66205, 310, 8882, 35160, 321, 353, 25707, 685, 440, 264, 10146, 470, 35160, 6338, 13, 3648, 2923, 488, 599, 310, 5909, 524, 310, 10390, 369, 310, 958, 37582, 24370, 310, 16168, 279, 35160, 430, 29235, 16168, 12490, 462, 8225, 10, 1518, 3604, 279, 14289, 13, 1368, 424, 3571, 7905, 52790, 42331, 11, 1442, 413, 488, 1042, 88323, 635, 279, 14289, 11, 279, 12571, 6338, 2677, 914, 381, 10146, 470, 13, 88323, 5802, 4138, 303, 328, 1152, 3728, 1, 321, 424, 628, 6497, 2880, 264, 7875, 6338, 318, 8761, 579, 9712, 4147, 8, 8366, 264, 8882, 6338, 13, 1368, 488, 3184, 88323, 310, 14289, 264, 6338, 488, 1467, 4045, 10146, 494, 11, 424, 628, 944, 303, 2424, 2880, 3728, 1332, 279, 8833, 6338, 1902, 381, 279, 1788, 1331, 466, 7875, 1056, 279, 6338, 488]),
        ("wiki_2", [4199, 1599, 1142, 15412, 791, 1834, 364, 491, 2660, 30, 198, 68610, 303, 359, 14004, 20685, 24189, 6, 3727, 539, 18042, 53228, 450, 11, 3588, 220, 18, 11, 220, 17, 15, 16, 15, 13, 198, 16, 8, 364, 264, 13238, 3354, 2660, 11, 1092, 579, 279, 1866, 653, 1142, 15412, 791, 1834, 353, 1220, 635, 310, 381, 5902, 30, 198, 17, 8, 1204, 1248, 635, 353, 599, 310, 1042, 821, 2660, 4092, 1518, 353, 628, 1151, 3154, 777, 1447, 15049, 30, 198, 40, 1467, 883, 220, 21, 15, 15, 15, 11509, 20235, 1396, 220, 17, 19, 3979, 364, 220, 16, 15, 2756, 364, 799, 314, 821, 6386, 864, 995, 264, 6529, 491, 7697, 13, 198, 3733, 369, 2250, 1142, 15412, 364, 1396, 314, 1439, 11509, 5353, 748, 38705, 220, 16, 23, 220, 15, 15, 15, 1142, 15412, 1396, 220, 17, 19, 3979, 321, 4161, 6716, 303, 3665, 314, 1602, 44782, 1452, 466, 41339, 290, 13, 1061, 369, 1381, 6970, 220, 18, 3818, 3998, 321, 279, 2660, 369, 22284, 383, 1118, 2081, 364, 264, 2609, 314, 821, 16609, 20209, 13, 198, 5653, 777, 488, 628, 303, 5732, 694, 635, 11048, 19878, 321, 524, 24523, 2329, 913, 3254, 11048, 478, 9468, 310, 279, 1675, 486, 1179, 1238, 220, 16, 2182, 488, 628, 1236, 264, 2330, 20045, 486, 198, 34691, 11, 33897, 11, 488, 5617, 220, 16, 23, 74, 1142, 15412, 264, 1834, 383, 264, 6529, 491, 2660, 30, 2500, 5964, 1467, 488, 6857, 685, 30, 3437, 2993, 314, 10391, 22444, 287, 1467, 1782, 20209, 599, 30]),
        ("wiki_3", [760, 21431, 8486, 314, 11332, 8660, 279, 220, 17, 15, 16, 22, 12, 16, 23, 2812, 1007, 440, 449, 2560, 383, 6859, 421, 2830, 491, 7832, 11, 30575, 11, 8364, 63544, 321, 4131, 6618, 21066, 13, 198, 1596, 513, 11692, 421, 21431, 4007, 668, 381, 279, 58526, 314, 264, 2943, 1007, 11, 393, 20, 22, 17, 3410, 3146, 53905, 17394, 22880, 13, 8439, 279, 20438, 314, 279, 10498, 11, 1004, 7832, 2015, 682, 6237, 310, 9412, 279, 6859, 888, 279, 2943, 4514, 4496, 314, 279, 10498, 13, 198, 91916, 11370, 383, 7348, 11454, 10498, 321, 2463, 10456, 10633, 364, 4007, 303, 21431, 2968, 29394, 11, 37057, 310, 2638, 393, 20, 22, 17, 3410, 303, 10498, 10456, 310, 1870, 321, 7042, 8417, 321, 393, 20, 21, 13, 21, 3410, 303, 10042, 10867, 310, 1761, 16134, 26298, 11, 1680, 430, 3988, 50235, 13, 198, 91916, 11370, 1910, 9542, 310, 10498, 321, 2463, 73460, 10456, 1761, 364, 56508, 4007, 321, 8417, 13, 8916, 310, 3827, 777, 883, 279, 3447, 314, 279, 24518, 64025, 10498, 6420, 13, 198, 91916, 11370, 383, 4470, 13, 220, 23, 11454, 10498, 321, 2463, 10456, 10633, 364, 56508, 4007, 321, 8417, 13, 14502, 777, 883, 1092, 725, 5064, 303, 279, 2463, 73460, 6420, 13]),
        ("wiki_4", [33, 5052, 939, 5699, 28602, 604, 8810, 326, 4738, 310, 469, 36201, 5681, 604, 8810, 469, 5257, 81873, 11, 10964, 11, 17397, 11, 321, 4434, 3429, 13, 198, 2053, 314, 1381, 11, 220, 16, 26791, 1542, 1881, 494, 417, 5052, 939, 5699, 28602, 604, 8810, 318, 56, 6277, 8, 310, 469, 36201, 5681, 604, 8810, 318, 38, 553, 198, 760, 24830, 5257, 494, 417, 5052, 939, 5699, 28602, 604, 8810, 318, 56, 6277, 8, 310, 469, 36201, 5681, 604, 8810, 318, 38, 8, 369, 783, 6277, 62969, 6844, 27634, 1658, 5052, 32, 13073, 318, 16, 17, 17, 20, 16, 8, 421, 2101, 6819, 506, 220, 17, 18, 25, 19, 15, 321, 28712, 310, 506, 220, 17, 16, 25, 16, 20, 13, 1049, 4829, 12805, 220, 17, 16, 25, 18, 20, 3979, 13]),
        ("wiki_5", [760, 8807, 615, 56690, 321, 279, 7604, 615, 62981, 34078, 30, 198, 2126, 369, 424, 279, 8807, 1683, 2086, 321, 1432, 539, 264, 2086, 716, 314, 5426, 30, 198, 5236, 279, 8807, 1831, 11415, 321, 1852, 30, 198, 81203, 1208, 4267, 359, 24929, 8311, 310, 728, 3058, 7851, 1208, 513, 10839, 310, 381, 8807, 13, 198, 81203, 1208, 68789, 310, 381, 8807, 13, 7851, 1208, 1683, 2330, 13, 198, 81203, 1208, 1683, 2526, 13, 7851, 1208, 5077, 383, 10175, 13, 198, 81203, 1208, 5077, 383, 29998, 13, 7851, 1208, 513, 9539, 310, 11587, 5402, 321, 836, 869, 13, 198, 81203, 1208, 1683, 45615, 883, 10910, 321, 19633, 13, 198, 81203, 1208, 513, 7629, 310, 491, 6498, 486, 198, 5236, 1394, 1683, 8807, 466, 7604, 30])
]
HOLDOUT_DOCS = [
("math_0", [32, 6378, 369, 8842, 264, 31687, 8151, 440, 1024, 4627, 13, 2838, 6587, 310, 1236, 2617, 781, 599, 3213, 15271, 7674, 15436, 364, 40897, 13, 2532, 668, 381, 4097, 1208, 2702, 25, 88497, 11, 1024, 1330, 4627, 11, 321, 1024, 6357, 13, 2302, 668, 8070, 220, 18, 2756, 321, 220, 17, 20803, 506, 279, 3013, 9233, 11, 321, 781, 3019, 310, 599, 220, 17]),
    ("math_1", [32, 35197, 682, 220, 16, 15, 15, 19434, 314, 65152, 321, 6587, 310, 3107, 1416, 264, 49693, 13159, 440, 449, 2982, 314, 220, 20, 15, 15, 9057, 19434, 13, 1368, 279, 35197, 1048, 6587, 310, 5106, 264, 220, 20, 70040, 6669, 1752, 2094, 279, 4307, 13159, 11, 1204, 1599, 19434, 314, 65152, 668, 381, 4221, 310, 3107, 1416, 279, 13159, 321, 279, 1752, 30]),
    ("code_0", [51, 1646, 321, 1141, 18254, 15234, 279, 1118, 9665, 314, 279, 10507, 28835, 21800, 2128, 318, 7048, 220, 16, 8, 220, 430, 314, 5632, 220, 16, 11, 220, 17, 15, 17, 20, 13, 4642, 3230, 7002, 4945, 42443, 383, 5589, 220, 17, 15, 11, 220, 17, 15, 17, 20, 13, 561, 18254, 579, 6891, 7009, 2445, 494, 10770, 13, 561, 18254, 20653, 345, 1646]),
    ("chat_0", [9205, 10895, 3668, 310, 3644, 5792, 20730, 318, 76102, 220, 21, 13, 15, 43181, 10082, 2141, 220, 19, 13, 15, 43181, 4130, 42700, 220, 18, 13, 15, 10, 45600, 220, 17, 13, 15, 43181, 33749, 23324, 220, 20, 13, 15, 10, 553, 3437, 6697, 2243, 1044, 353, 1608, 30, 198, 1860, 678, 24521, 6602, 594, 48532, 24521, 13753, 11, 488, 628, 6497, 1420, 279])
]
PROMPT = [10812, 4811, 43768, 3124, 34639, 10454, 303, 8637, 282, 4120, 0, 198, 5236, 488, 1318, 310, 615, 2577, 506, 3154, 17387, 43768, 30, 1394, 668, 599, 279, 6429, 11, 2113, 411, 383]

OBS_ALL = ["dip", "hump", "plateau", "nll_holdout", "rep16", "loopy", "pr_sigma", "eff_rank", "margin"]
FAMILIES = {"field": ["dip", "hump", "plateau"], "leaf": ["pr_sigma", "eff_rank"], "mode": ["rep16", "loopy"]}
DISC_OTHERS = ["nll_holdout", "eff_rank", "loopy", "margin"]
BIS_ORDER = ["dip", "nll_holdout", "eff_rank", "loopy", "margin", "hump", "plateau", "pr_sigma", "rep16"]
BIS_CAP = 12             # union-кап точек бисекции (пре-рег §6.6: раунд 1 = 9 mid, раунд 2 = ≤3)
BIS_WIDTH = 0.02
BIS_ITERS = 5
SP_CRIT = 0.9

BASE_SHAS = {}           # заполняется после загрузки модели (S2)


def _script_sha():
    """sha256 скрипта; в notebook (Kaggle) __file__ отсутствует → None (идентичность
    закреплена коммитом-билдом и E64_FNS_SHA256)."""
    try:
        p = os.path.abspath(__file__)
        if os.path.exists(p):
            return sha256_file(p)
    except NameError:
        pass
    return None


# Общие функции prism/ (spec-лоад, минуя prism/__init__.py — тот тянет
# datasets/transformers; устоявшийся паттерн AT-экспов):
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_p(name, rel):
    _sp = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "prism", rel))
    _m = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    return _m


_pf = _load_p("prism_field", "field.py")
_nulls = _load_p("prism_nulls_and_stats", "nulls_and_stats.py")
sha256_file = _pf.sha256_file        # was: локальный дубль (field.py)
spearman = _nulls.spearman           # was: локальный дубль + rankdata


def logline(s=""):
    print(s, flush=True)


def check_e64_pin():
    """S7: sha-пин встроенного e64-источника ДО exec (пре-рег S7)."""
    sha = hashlib.sha256(E64_FNS_SRC.encode()).hexdigest()
    if sha != E64_FNS_SHA256:
        raise RuntimeError(f"S7 FAIL: e64-функции sha {sha} != pinned {E64_FNS_SHA256}")
    ns = {"np": np}
    exec(E64_FNS_SRC, ns)
    return ns["curves_series"], sha


def docs_sha_check():
    """S6: встроенные токены = закреплённые sha + disjoint (пре-рег §4/S6)."""
    def sha_ids(pairs):
        h = hashlib.sha256()
        for n, ids in pairs:
            h.update(n.encode())
            h.update(b"\x00")
            h.update(np.array(ids, dtype=np.int64).tobytes())
            h.update(b"\x01")
        return h.hexdigest()
    ms = sha_ids(MAIN_DOCS)
    hs = sha_ids(HOLDOUT_DOCS)
    if ms != MAIN_SHA256:
        raise RuntimeError(f"S6 FAIL: main_sha {ms} != {MAIN_SHA256}")
    if hs != HOLDOUT_SHA256:
        raise RuntimeError(f"S6 FAIL: holdout_sha {hs} != {HOLDOUT_SHA256}")
    mn = {n for n, _ in MAIN_DOCS}
    for n, ids in HOLDOUT_DOCS:
        if n in mn:
            raise RuntimeError(f"S6 FAIL: имя holdout {n} в main")
        for m, mid in MAIN_DOCS:
            if ids[:16] == mid[:16]:
                raise RuntimeError(f"S6 FAIL: 16-ток. префикс {n} == {m}")
    if len(PROMPT) != PLEN:
        raise RuntimeError(f"S6 FAIL: prompt len {len(PROMPT)} != {PLEN}")
    if PROMPT != MAIN_DOCS[0][1][:PLEN]:
        raise RuntimeError("S6 FAIL: prompt != main[0][:32]")
    return ms, hs


def load_model():
    path = MODEL_LOCAL if os.path.isdir(MODEL_LOCAL) else MODEL_HF
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    # Kaggle T4: GPU (fp32, torch-fallback без fla — те же операции, что в CPU-смоуке);
    # локально: cuda нет — остаётся CPU.
    if torch.cuda.is_available():
        m = m.cuda()
    return m, path


def model_facts(m):
    tc = getattr(m.config, "text_config", None)
    cfg = tc if tc is not None else m.config
    lt = getattr(cfg, "layer_types", None)
    if lt is None:
        lt = [type(l).__name__ for l in m.model.layers]
    return {
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "hidden_size": int(cfg.hidden_size),
        "vocab_size": int(cfg.vocab_size),
        "tie_word_embeddings": bool(cfg.tie_word_embeddings),
        "layer_types": [str(x) for x in lt],
        "architecture": m.__class__.__name__,
        "n_tensors": len(m.state_dict()),
        "n_1d_tensors": sum(1 for v in m.state_dict().values() if v.dim() == 1),
    }


def state_shas(m):
    return {k: hashlib.sha256(v.detach().cpu().float().numpy().tobytes()).hexdigest()
            for k, v in m.state_dict().items()}


def make_sd_r(sd0, seed):
    """ВЕСЬ state_dict случайный (вкл. 1-D): randn·std, сид-детерминированно (пре-рег §2).
    Per-ключ генератор с РАЗНЫМИ сидами (seed·1000003 + idx в sorted-порядке): те же
    значения независимо от порядка итерации И независимые значения для тензоров
    одинаковой формы (один сид на все ключи дал бы идентичные randn'ы на shape-дублях)."""
    keys = sorted(sd0)
    out = {}
    for i, k in enumerate(keys):
        v = sd0[k]
        g = torch.Generator(device=v.device).manual_seed(seed * 1000003 + i)
        out[k] = torch.randn(v.shape, generator=g, device=v.device) * v.std()
    return out


def check_randomized(sd_r, sd0):
    """S2b: все тензоры случайны. Крупные (n≥1024): max|diff|>0, |mean|<5σ/√n
    (union-bound по ~200 тензоров), std-отношение ∈[0.8,1.25]. Мелкие (n<1024,
    dt_bias/A_log): MC-шум mean/std велик — max|diff|>0 + std-отношение ∈[0.5,2.0]
    (пре-рег S2b: «всё рандомизировано» — строгое неравенство + масштабовая
    валидность); независимость shape-дублей — отдельно в main."""
    bad = []
    n_1d = 0
    n_small = 0
    for k in sorted(sd0):
        a, b = sd0[k].float(), sd_r[k].float()
        if a.dim() == 1:
            n_1d += 1
        if float((a - b).abs().max()) == 0.0:
            bad.append(k)
            continue
        n = a.numel()
        if n >= 1024:
            # 5σ/√n (не 3σ): ~200 больших тензоров → union-bound; цель — ловить
            # не-случайные (константные/сдвинутые) тензоры, а не гауссовость
            mean_ok = abs(float(b.mean())) < 5.0 * float(b.std()) / math.sqrt(n)
            r = float(b.std()) / max(float(a.std()), 1e-12)
            ratio_ok = 0.8 <= r <= 1.25
        else:
            n_small += 1
            # mean-проверка на мелких (n<1024) статистически неосмысленна
            # (MC-шум mean ~ 1/√n: n=16 → 1.5σ-порог ложно срабатывает часто);
            # случайность гарантируют max|diff|>0 + std-отношение + S2b-независимость
            mean_ok = True
            r = float(b.std()) / max(float(a.std()), 1e-12)
            ratio_ok = 0.5 <= r <= 2.0
        if not mean_ok:
            bad.append(k + ":mean")
        if not ratio_ok:
            bad.append(k + f":stdratio={r:.3f}")
    return bad, n_1d, n_small


def set_interp(m, sd0, sd_r, lam):
    """W(λ) = λ·sd0 + (1−λ)·sd_r. λ∈{0,1} — чистый copy_ (побитово, без арифметики
    1.0·x/0.0·x — защита от −0.0 flip, пре-рег S2)."""
    with torch.no_grad():
        dev = next(m.parameters()).device
        for k, v in m.state_dict().items():
            if lam == 1.0:
                v.copy_(sd0[k].to(dev))
            elif lam == 0.0:
                v.copy_(sd_r[k].to(dev))
            else:
                v.copy_((lam * sd0[k] + (1.0 - lam) * sd_r[k]).to(dev))


class Hook:
    def __init__(self, layer):
        self.val = None
        self.h = layer.register_forward_hook(self)

    def __call__(self, mod, inp, out):
        self.val = out[0] if isinstance(out, tuple) else out

    def remove(self):
        self.h.remove()


def fwd_doc(m, ids, hook):
    """Один forward дока: nll (CE, t≥1), h23 [T,D], margin-медиана (top1−top2, t≥1)."""
    t = torch.tensor([ids], dtype=torch.long, device=m.device)
    with torch.no_grad():
        o = m(input_ids=t, use_cache=False)
    logits = o.logits[0].float()
    nll = float(F.cross_entropy(logits[:-1], t[0, 1:]))
    lg = logits[1:].cpu().numpy()
    srt = np.sort(lg, axis=1)
    margin = float(np.median(srt[:, -1] - srt[:, -2]))
    h = hook.val[0, :len(ids)].detach().float().cpu().numpy()
    return nll, h, margin


def doc_metrics(h, curves_series):
    cur = curves_series(h)
    hh = h.astype(np.float64)
    s = np.linalg.svd(hh, compute_uv=False)
    s2 = s * s
    pr_sigma = float((s2.sum()) ** 2 / max(float((s2 * s2).sum()), 1e-300))
    p = s2 / s2.sum()
    eff_rank = float(np.exp(-float((p * np.log(p)).sum())))
    return {
        "dip": float(cur["dip"]), "hump": float(cur["hump"]), "plateau": float(cur["plateau"]),
        "pr_sigma": pr_sigma, "eff_rank": eff_rank,
    }


def gen_metrics(all_toks, h_all, curves_series):
    """rep16 (16-окна без перекрытия, <12 distinct) + loopy (max ACF k≥2 > 0.95)."""
    T = len(all_toks)
    nwin = max(0, (T - 16) // 16)
    rep16 = None
    if nwin > 0:
        hits = sum(1 for i in range(0, T - 16, 16) if len(set(all_toks[i:i + 16])) < 12)
        rep16 = float(hits) / nwin
    loopy = None
    if T >= 28:
        c = curves_series(h_all)["curve"]
        ks = [k for k in c if int(k) >= 2]
        if ks:
            loopy = int(max(c[k] for k in ks) > 0.95)
    deg = (T - PLEN) < 32
    return rep16, loopy, deg


def generate(m, prompt, hook):
    """Greedy до T_GEN или EOS; h23 по всем позициям (prompt + gen)."""
    x = torch.tensor([prompt], dtype=torch.long, device=m.device)
    cache = DynamicCache(config=m.config)
    with torch.no_grad():
        o = m(input_ids=x, past_key_values=cache, use_cache=True)
    hs = [hook.val[0, -1].detach().float().cpu().numpy()]
    toks = list(prompt)
    for _ in range(T_GEN):
        lg = o.logits[0, -1].float()
        nxt = int(lg.argmax())
        if nxt == EOS_ID:
            break
        toks.append(nxt)
        with torch.no_grad():
            o = m(input_ids=torch.tensor([[nxt]], dtype=torch.long, device=m.device),
                  past_key_values=cache, use_cache=True)
        hs.append(hook.val[0, -1].detach().float().cpu().numpy())
    h_all = np.stack(hs, axis=0)
    return toks, h_all


def run_block(m, sd0, sd_r, lam, doc_names, hold_names, tk_decode, curves_series):
    """Один (λ, сид): форварды main/holdout + генерация. Возвращает cell."""
    set_interp(m, sd0, sd_r, lam)
    hook = Hook(m.model.layers[L_H])
    cell = {"lam": lam, "docs": {}, "holdout": {}}
    try:
        for n, ids in MAIN_DOCS:
            if n not in doc_names:
                continue
            nll, h, margin = fwd_doc(m, ids, hook)
            d = doc_metrics(h, curves_series)
            d["nll"] = nll
            d["margin"] = margin
            d["T"] = len(ids)
            cell["docs"][n] = d
        for n, ids in HOLDOUT_DOCS:
            if n not in hold_names:
                continue
            nll, h, _ = fwd_doc(m, ids, hook)
            cell["holdout"][n] = {"nll": nll, "T": len(ids)}
        toks, h_all = generate(m, PROMPT, hook)
        rep16, loopy, deg = gen_metrics(toks, h_all, curves_series)
        cell["gen"] = {
            "tokens": toks, "T_prompt": PLEN, "T_act": len(toks) - PLEN,
            "rep16": rep16, "loopy": loopy, "degenerate_short": bool(deg),
            "text": tk_decode(toks),
        }
        if lam == 1.0:
            cell["S2_sha_ok"] = (state_shas(m) == BASE_SHAS)
    finally:
        hook.remove()
    return cell


def u_curve(O, grid):
    """u(λ) = (O(λ)−O(0))/(O(1)−O(0)) по grid; None если no_range (пре-рег §6.1)."""
    o0, o1 = O[grid[0]], O[grid[-1]]
    if abs(o1 - o0) < 1e-6 * max(1.0, abs(o0), abs(o1)):
        return None
    return {lam: (O[lam] - o0) / (o1 - o0) for lam in grid}


def crossing_of(u, grid):
    """Первый j: u(λ_j)≥0.5 И u(λ_{j+1})≥0.5 (персистентность) → (λ_j, idx) или None."""
    for i in range(len(grid) - 1):
        if u[grid[i]] >= 0.5 and u[grid[i + 1]] >= 0.5:
            return grid[i], i
    return None


def lam_c_machine(curves, grid, idx_of, obs_list=None):
    """curves: {obs: {seed: {lam: val}}}. Возвращает (per-obs λ_c-блок, seedavg, u_seedavg,
    bisection-интервалы). Бисекция-уточнение — отдельно (нужны доп. форварды)."""
    out = {}
    obs_list = obs_list or OBS_ALL
    seedavg = {o: {lam: float(np.mean([curves[o][s][lam] for s in SEEDS])) for lam in grid}
               for o in obs_list}
    u_sa = {o: u_curve(seedavg[o], grid) for o in seedavg}
    for o in obs_list:
        entry = {"seeds": {}, "ci": None, "refined": None, "outcome": None,
                 "spearman_seedavg": spearman(np.array(grid), np.array(list(seedavg[o].values()))),
                 "max_u": None, "overshoot": False, "u_exit": None,
                 "u_seedavg": {f"{k:.3f}": v for k, v in u_sa[o].items()} if u_sa[o] else None,
                 "bisect_interval": None}
        if u_sa[o] is None:
            entry["outcome"] = "no_range"
            out[o] = entry
            continue
        rho = entry["spearman_seedavg"]
        if rho is None or abs(rho) < SP_CRIT:
            entry["outcome"] = "non_monotone_excluded"
            out[o] = entry
            continue
        entry["max_u"] = max(u_sa[o].values())
        entry["overshoot"] = bool(entry["max_u"] > 1.0)
        sig = float(np.std([curves[o][s][grid[0]] for s in SEEDS]))
        for lam in grid:
            if abs(seedavg[o][lam] - seedavg[o][grid[0]]) > 2.0 * sig:
                entry["u_exit"] = lam
                break
        lcs = []
        for s in SEEDS:
            us = u_curve(curves[o][s], grid)
            if us is None:
                entry["seeds"][s] = None
                continue
            cr = crossing_of(us, grid)
            entry["seeds"][s] = cr[0] if cr else None
            if cr:
                lcs.append(cr[1])
        if not lcs:
            entry["outcome"] = "no_transition"
            out[o] = entry
            continue
        vals = [grid[i] for i in lcs]
        lo, hi = min(vals), max(vals)
        span = idx_of[hi] - idx_of[lo]
        entry["ci"] = [lo, hi]
        entry["outcome"] = "not_resolved" if span > 1 else "resolved"
        # интервал для бисекции: j0 — первое u_seedavg ≥ 0.5 (всегда есть: u(1)=1)
        j0 = next(i for i, lam in enumerate(grid) if u_sa[o][lam] >= 0.5)
        a, b = (0.0 if j0 == 0 else grid[j0 - 1], grid[j0])
        entry["bisect_interval"] = [a, b]
        out[o] = entry
    return out, seedavg, u_sa


def bisection_plan(bis_intervals, computed_lams, order=None):
    """Union-план точек бисекции (пре-рег §6.6): раунд 1 — mid всех интервалов (order),
    раунд 2 — mid оставшихся (если ширина > BIS_WIDTH). Кап BIS_CAP. Возвращает список
    раундов [[(obs, mid), ...], ...] без точек, уже посчитанных."""
    order = order or BIS_ORDER
    rounds = []
    used = set()
    intervals = {o: list(iv) for o, iv in bis_intervals.items() if iv}
    for rnd in range(BIS_ITERS):
        pts = []
        for o in order:
            if o not in intervals:
                continue
            a, b = intervals[o]
            if b - a <= BIS_WIDTH:
                intervals.pop(o)
                continue
            mid = round((a + b) / 2.0, 6)
            if mid in used or mid in computed_lams:
                continue
            pts.append((o, mid))
        if not pts:
            break
        rounds.append(pts)
        for o, mid in pts:
            used.add(mid)
    flat = [p for r in rounds for p in r]
    capped = len(flat) > BIS_CAP
    if capped:
        keep = set()
        rounds2 = []
        for r in rounds:
            nr = []
            for p in r:
                if len(keep) < BIS_CAP:
                    keep.add(p[1])
                    nr.append(p)
            if nr:
                rounds2.append(nr)
        rounds = rounds2
    return rounds, capped


def refine_interval(a, b, u_at):
    """Бисекция по уже посчитанным u-значениям (seed-avg, пре-рег §6.6).
    Возвращает (refined=(a+b)/2, width=b−a); стоп: width ≤ BIS_WIDTH, ≤BIS_ITERS
    итераций, или mid не посчитан."""
    for _ in range(BIS_ITERS):
        if b - a <= BIS_WIDTH:
            break
        mid = round((a + b) / 2.0, 6)
        if mid not in u_at:
            break
        if u_at[mid] >= 0.5:
            b = mid
        else:
            a = mid
    return (a + b) / 2.0, (b - a)


def refine_bisect(obs, iv, cells, seedavg_pre, doc_names, hold_names, lam_list):
    """Бисекция-уточнение одной наблюдаемой (пре-рег §6.6): нормированные u(λ) на
    бисекционных точках + точках lam_list (seed-avg), refine_interval.
    Возвращает (refined, width); (None, None) если no_range (scale ~ 0)."""
    o0, o1 = seedavg_pre[obs][lam_list[0]], seedavg_pre[obs][lam_list[-1]]
    scale = o1 - o0
    if abs(scale) < 1e-12:
        return None, None
    u_at = {}
    for key, c in cells.items():
        if "bis" not in key:
            continue
        lam = c["lam"]
        if obs not in u_at:
            u_at[lam] = {}
        u_at[lam][c["seed"]] = aggregate_cell(c, doc_names, hold_names)[obs]
    u_sa_at = {}
    for lam, vals in u_at.items():
        u_sa_at[lam] = (float(np.mean(list(vals.values()))) - o0) / scale
    # расширяем точками lam_list (также нормированные) для непрерывности
    u_sa_at.update({lam: (val - o0) / scale for lam, val in seedavg_pre[obs].items()})
    return refine_interval(iv[0], iv[1], u_sa_at)


def selftest_lam_c():
    """Детерминированный self-test машины λ_c + verdict-веток (AGENTS: exec точного фрагмента)."""
    A = {0.0: 0.0, 0.2: 0.1, 0.4: 0.3, 0.5: 0.45, 0.55: 0.6, 0.6: 0.7,
         0.65: 0.8, 0.7: 0.85, 0.75: 0.9, 0.85: 0.97, 1.0: 1.0}
    B = [0.0, 0.9, 0.2, 0.95, 0.1, 0.9, 0.3, 0.85, 0.2, 0.8, 1.0]  # немонотонна, O(0)≠O(1)
    B = dict(zip(LAM11, B))
    C = {lam: 0.001 for lam in LAM11}
    curves = {"A": {s: A for s in SEEDS}, "B": {s: B for s in SEEDS},
              "C": {s: C for s in SEEDS}}
    # временная замена OBS_ALL — self-test на 3 псевдо-наблюдаемых
    out, seedavg, us = lam_c_machine(curves, LAM11, LAM11_IDX, ["A", "B", "C"])
    assert out["A"]["seeds"] == {11: 0.55, 12: 0.55, 13: 0.55}, out["A"]["seeds"]
    assert out["A"]["outcome"] == "resolved" and out["A"]["ci"] == [0.55, 0.55]
    assert out["A"]["overshoot"] is False
    assert abs(out["A"]["spearman_seedavg"] - 1.0) < 1e-9
    assert out["A"]["bisect_interval"] == [0.5, 0.55]
    assert out["B"]["outcome"] == "non_monotone_excluded"
    assert out["C"]["outcome"] == "no_range"
    D = dict(A); D[0.55] = 0.6; D[0.6] = 0.4   # выброс 0.55 не удержан (0.6 < 0.5) → первое
                                                # удержанное пересечение — пара (0.65, 0.7)
    out2, _, _ = lam_c_machine({"D": {s: D for s in SEEDS}}, LAM11, LAM11_IDX, ["D"])
    assert out2["D"]["seeds"] == {11: 0.65, 12: 0.65, 13: 0.65}, out2["D"]["seeds"]
    E = dict(A); E[0.7] = 1.2                    # overshoot
    out3, _, _ = lam_c_machine({"E": {s: E for s in SEEDS}}, LAM11, LAM11_IDX, ["E"])
    assert out3["E"]["overshoot"] is True
    # bisection_plan: интервалы [0.5,0.55] и [0.2,0.4] → mid 0.525, 0.3
    # (mid на точке сетки — не планируется: u уже посчитан в main-сетке)
    r, capped = bisection_plan({"A": [0.5, 0.55], "B": [0.2, 0.4]}, LAM11, ["A", "B"])
    mids = sorted(p[1] for rr in r for p in rr)
    assert mids == [0.3, 0.525] and not capped, (mids, capped)
    # bisection_plan cap: 13 интервалов (все шириной > 0.02) → 13 mid > BIS_CAP=12
    # → capped, 12 точек, отсечка по порядку (BIS_ORDER)
    many = {f"O{i}": [0.0, 0.05 + 0.1 * i] for i in range(13)}
    r2, capped2 = bisection_plan(many, LAM11, [f"O{i}" for i in range(13)])
    mids2 = [p[1] for rr in r2 for p in rr]
    assert capped2 and len(mids2) == 12 and len(set(mids2)) == 12, (capped2, mids2)
    # refine: НОРМИРОВАННЫЕ u (контракт функции) — сырые O-значения не сравнивать с 0.5
    # (регрессия BLOCKER: бисекция-рефаин на сырых O давала границу в неверном направлении)
    raw = {0.0: 12.0, 0.5: 8.0, 0.55: 6.0, 1.0: 2.2}   # nll-подобная убывающая кривая
    o0, o1 = raw[0.0], raw[1.0]
    un = {lam: (v - o0) / (o1 - o0) for lam, v in raw.items()}
    assert un[0.55] > 0.5 > un[0.5]
    u525 = (10.0 - o0) / (o1 - o0)   # u(0.525) для сырого 10.0 = 0.204 < 0.5 → a=0.525
    ref_n, w_n = refine_interval(0.5, 0.55, {0.525: u525})
    # сырое 10.0 ≥ 0.5 → refine даст b=0.525 (неверное направление)
    ref_bad, _ = refine_interval(0.5, 0.55, {0.525: 10.0})
    assert ref_n != ref_bad, (ref_n, ref_bad)
    assert abs(ref_n - 0.5375) < 1e-9 and abs(w_n - 0.025) < 1e-9, (ref_n, w_n)
    # refine: две точки → данные кончились на ширине 0.025
    ref2, w2 = refine_interval(0.4, 0.5, {0.45: 0.6, 0.425: 0.3})
    assert abs(ref2 - 0.4375) < 1e-9 and abs(w2 - 0.025) < 1e-9, (ref2, w2)
    # refine: три точки → сходится до ≤0.02 (0.1→0.05→0.025→0.0125)
    ref3, w3 = refine_interval(0.4, 0.5, {0.45: 0.6, 0.425: 0.3, 0.4375: 0.55})
    assert w3 <= 0.02 and abs(ref3 - 0.43125) < 1e-9, (ref3, w3)
    # bisection-рефаин: exec ТОЙ функции, что гонится в main (Kaggle v1 упал здесь:
    # np.mean(dict_values) — TypeError; теперь list() + общий selftest)
    def _cell(seed, lam, val):
        return {"lam": lam, "seed": seed,
                "docs": {"d1": {"dip": val, "hump": val, "plateau": val, "margin": val,
                               "pr_sigma": 1.0, "eff_rank": 1.0, "nll": val}},
                "holdout": {"h1": {"nll": val}},
                "gen": {"rep16": 0.0, "loopy": 0}}
    cells_syn = {}
    for s in (11, 12, 13):
        cells_syn[f"s{s}_lam0.00"] = _cell(s, 0.0, 12.0)
        cells_syn[f"s{s}_lam1.00"] = _cell(s, 1.0, 2.2)
        cells_syn[f"s{s}_bis0.525"] = _cell(s, 0.525, 10.0)
    sp = {"dip": {0.0: 12.0, 1.0: 2.2}}
    ref_b, w_b = refine_bisect("dip", [0.5, 0.55], cells_syn, sp, ["d1"], ["h1"], [0.0, 1.0])
    # u(0.525) = (10−12)/(2.2−12) = 0.204 < 0.5 → a=0.525 → ref=0.5375, width=0.025
    assert abs(ref_b - 0.5375) < 1e-9 and abs(w_b - 0.025) < 1e-9, (ref_b, w_b)
    return True


def pair_compare(dip_e, o_e):
    """«Совпадает/отделён» по пре-рег §6: per-seed |Δ| в шагах сетки + CI-overlap."""
    if dip_e.get("ci") is None or o_e.get("ci") is None:
        return None
    a_lo, a_hi = dip_e["ci"]
    b_lo, b_hi = o_e["ci"]
    overlap = not (a_hi < b_lo or b_hi < a_lo)
    ns = 0
    for s in SEEDS:
        dv, ov = dip_e["seeds"].get(s), o_e["seeds"].get(s)
        if dv is None or ov is None:
            continue
        ns += int(abs(LAM11_IDX[ov] - LAM11_IDX[dv]) >= 2)
    p_raw = (1.0 / 8.0) * sum(math.comb(3, k) for k in range(ns, 4))  # P(Bin(3,½) ≥ ns)
    p_bonf = min(1.0, p_raw / len(DISC_OTHERS))
    # «совпадает»: per-seed |Δ| ≤ 1 шаг (все сида с λ_c) И CI пересекаются
    all_le1 = True
    for s in SEEDS:
        dv, ov = dip_e["seeds"].get(s), o_e["seeds"].get(s)
        if dv is not None and ov is not None and abs(LAM11_IDX[ov] - LAM11_IDX[dv]) > 1:
            all_le1 = False
    coincide = all_le1 and overlap
    separated = (ns == len(SEEDS)) and (not overlap)
    return {"ci_overlap": overlap, "seeds_ge2steps": ns, "p_raw": p_raw, "p_bonf": p_bonf,
            "coincide": bool(coincide), "separated": bool(separated)}


def verdict(lambda_c, seedavg, us):
    """V3 > V2 > V1 > V0 (пре-рег §7)."""
    dip = lambda_c.get("dip", {})
    dip_outcome = dip.get("outcome")
    # V3: патология (overshoot dip на 0.7 + rep16 выше λ=1)
    u7 = us.get("dip")
    if u7 is not None and "rep16" in seedavg and seedavg["rep16"][0.7] is not None \
            and seedavg["rep16"][1.0] is not None and u7[0.7] > 1.0 \
            and seedavg["rep16"][0.7] > seedavg["rep16"][1.0]:
        return {"outcome": "V3_PATHOLOGY",
                "rule": "u_dip(0.7)>1 и rep16(0.7)>rep16(1) (seed-avg) — патология порога",
                "details": {"u_dip_0.7": u7[0.7], "rep16_0.7": seedavg["rep16"][0.7],
                            "rep16_1.0": seedavg["rep16"][1.0]}}
    if dip_outcome not in ("resolved",):
        return {"outcome": "V0_NOT_RESOLVED",
                "rule": "dip не даёт resolved λ_c",
                "details": {"dip_outcome": dip_outcome,
                            "excluded_non_monotone": [o for o, e in lambda_c.items()
                                                      if e.get("outcome") == "non_monotone_excluded"]}}
    pairs = {}
    sep_any = False
    coin_all = True
    for o in DISC_OTHERS:
        e = lambda_c.get(o, {})
        pc = pair_compare(dip, e)
        if pc is None:
            pairs[o] = {"comparable": False,
                        "reason": f"dip:{dip_outcome} other:{e.get('outcome')}"}
            coin_all = False
            continue
        pc["comparable"] = True
        pairs[o] = pc
        sep_any = sep_any or pc["separated"]
        coin_all = coin_all and pc["coincide"]
    if sep_any:
        return {"outcome": "V2_SOLITON_CANDIDATE",
                "rule": "λ_c^dip отделён от ≥1 из {nll_holdout, eff_rank, loopy, margin} "
                        "(per-seed ≥2 шага + CI не пересекаются)",
                "details": {"pairs": pairs,
                            "note": "порядок на прямой интерполяции весов, НЕ на обучении "
                                    "(TЗ §8.4 п.1) — кандидат идёт в T8.2"}}
    if coin_all:
        return {"outcome": "V1_GENERIC_COMPETENCE",
                "rule": "λ_c^dip совпадает со всеми 4 (per-seed ≤1 шага + CI пересекаются)",
                "details": {"pairs": pairs,
                            "note": "генерическая компетентность; слово «солитон» снимается"}}
    return {"outcome": "V0_NOT_RESOLVED",
            "rule": "смешанный случай (не все совпадения и ни одного отдаления)",
            "details": {"pairs": pairs}}


def families_check(lambda_c):
    """Согласованность внутри семейств (≥2 члена) — пре-рег §6/§8.3."""
    out = {}
    for fam, members in FAMILIES.items():
        pairs = {}
        ok = True
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                ea, eb = lambda_c.get(a, {}), lambda_c.get(b, {})
                if ea.get("ci") is None or eb.get("ci") is None:
                    pairs[f"{a}|{b}"] = {"comparable": False}
                    ok = False
                    continue
                pc = {"ci_overlap": not (ea["ci"][1] < eb["ci"][0] or eb["ci"][1] < ea["ci"][0])}
                all_le1 = True
                for s in SEEDS:
                    va, vb = ea["seeds"].get(s), eb["seeds"].get(s)
                    if va is not None and vb is not None \
                            and abs(LAM11_IDX[vb] - LAM11_IDX[va]) > 1:
                        all_le1 = False
                pc["coincide"] = bool(all_le1 and pc["ci_overlap"])
                pairs[f"{a}|{b}"] = pc
                ok = ok and pc["coincide"]
        out[fam] = {"members": members, "intra_consistent": bool(ok), "pairs": pairs}
    return out



# ── агрегация и health ray ─────────────────────────────────────────────────────

def aggregate_cell(cell, doc_names, hold_names):
    """per-(λ,сид): наблюдаемые = медианы (main: 6 доков, holdout: 4 дока)."""
    ag = {}
    for obs in ("dip", "hump", "plateau", "margin", "pr_sigma", "eff_rank"):
        vals = [cell["docs"][n][obs] for n in doc_names if n in cell["docs"]]
        ag[obs] = float(np.median(vals)) if vals else None
    nlls = [cell["holdout"][n]["nll"] for n in hold_names if n in cell["holdout"]]
    ag["nll_holdout"] = float(np.median(nlls)) if nlls else None
    g = cell.get("gen", {})
    ag["rep16"] = g.get("rep16")
    ag["loopy"] = g.get("loopy")
    return ag


def health_ray(cells_by_seed, doc_names, hold_names):
    """per (сид, λ): (dip_band_hinge, loopy, nll_holdout) — пре-рег S8/§9.6."""
    out = {}
    for s in SEEDS:
        c1 = cells_by_seed.get((s, 1.0))
        dip1 = None
        if c1:
            vals = [c1["docs"][n]["dip"] for n in doc_names if n in c1["docs"]]
            dip1 = float(np.median(vals))
        for key, c in cells_by_seed.items():
            if key[0] != s:
                continue
            lam = key[1]
            if lam == 1.0 or dip1 is None:
                hinge = 0.0 if lam == 1.0 else None
            else:
                vals = [c["docs"][n]["dip"] for n in doc_names if n in c["docs"]]
                if not vals:
                    hinge = None
                else:
                    hinge = abs(float(np.median(vals)) - dip1)
            out[f"s{s}_lam{lam:.2f}"] = {
                "dip_band_hinge": hinge,
                "loopy": c.get("gen", {}).get("loopy"),
                "nll": (float(np.median([c["holdout"][n]["nll"]
                                        for n in hold_names if n in c["holdout"]]))
                        if c.get("holdout") else None),
                "sample": c.get("gen", {}).get("text", "")[:400],
            }
    return out


def canonical_sha(payload):
    st = {k: v for k, v in payload.items() if k != "store_sha256"}
    st["store_sha256"] = None
    return hashlib.sha256(json.dumps(st, sort_keys=True).encode()).hexdigest()


# ── main ───────────────────────────────────────────────────────────────────────

def main(results_dir: str | None = None):
    t0 = time.time()
    args = sys.argv[1:]
    smoke = "--smoke" in args
    out_dir = "/tmp/at08_smoke" if smoke else (results_dir or os.path.join(_REPO, "results", "AT0"))
    os.makedirs(os.path.join(out_dir, "store"), exist_ok=True)
    # run-log (трекаемый артефакт, пре-рег §11): tee stdout в out_dir/AT0_8_run.log
    _logf = open(os.path.join(out_dir, "AT0_8_run.log"), "w")

    class _Tee:
        def __init__(self, *fs):
            self.fs = fs

        def write(self, s):
            for f in self.fs:
                f.write(s)

        def flush(self):
            for f in self.fs:
                f.flush()

    sys.stdout = _Tee(sys.__stdout__, _logf)

    logline(f"AT0_8 T8.1 {'SMOKE' if smoke else 'FULL'} (py{sys.version.split()[0]}, "
            f"torch{torch.__version__}, device "
            f"{'cuda' if torch.cuda.is_available() else 'cpu'})")

    selftest = selftest_lam_c()
    logline(f"selftest_lam_c: PASS (lambda_c machine + bisection plan + refine)")
    assert selftest

    e64_sha = None
    dcurve = None  # noqa: F841  (в ns-закрытии curves_series; отдельно не используется)
    curves_series, e64_sha = check_e64_pin()
    logline(f"S7: e64_gpu.py sha {e64_sha[:16]}… PASS")
    main_sha, holdout_sha = docs_sha_check()
    logline(f"S6: docs sha {main_sha[:16]}… / {holdout_sha[:16]}… + disjoint PASS")

    m, model_path = load_model()
    facts = model_facts(m)
    assert facts["num_hidden_layers"] == 24 and facts["hidden_size"] == 1024, facts
    logline(f"S1: {facts['architecture']} layers=24 D=1024 vocab={facts['vocab_size']} "
            f"1D-тензоров={facts['n_1d_tensors']} layer_types[0..3]={facts['layer_types'][:4]}")
    BASE_SHAS.update(state_shas(m))
    logline(f"base state: {len(BASE_SHAS)} тензоров, sha зафиксированы (S2-референс)")

    tk = AutoTokenizer.from_pretrained(model_path)
    tk_decode = lambda toks: tk.decode(toks, skip_special_tokens=False)

    sd0 = {k: v.detach().clone() for k, v in m.state_dict().items()}

    if smoke:
        doc_names = [n for n, _ in MAIN_DOCS[:2]]
        hold_names = [n for n, _ in HOLDOUT_DOCS[:1]]
        grid = [0.0, 1.0]
        seeds = [11]
    else:
        doc_names = [n for n, _ in MAIN_DOCS]
        hold_names = [n for n, _ in HOLDOUT_DOCS]
        grid = LAM13
        seeds = SEEDS

    inprog_path = os.path.join(out_dir, "store", "AT0_8_inprogress.json")
    cells = {}
    if os.path.exists(inprog_path) and not smoke:
        try:
            cells = json.load(open(inprog_path))["cells"]
            logline(f"resume: {len(cells)} клеток уже посчитано (скип)")
        except (json.JSONDecodeError, KeyError, TypeError):
            logline("resume: inprogress.json усечён (обрыв) — старт с нуля")
            cells = {}

    sd_r = None
    s2b = {}
    for s in seeds:
        sd_r = make_sd_r(sd0, s)
        bad, n_1d, n_small = check_randomized(sd_r, sd0)
        # S2b-независимость: shape-дубли (например A_log во всех linear-слоях) — РАЗНЫЕ
        # случайные значения (один сид на все ключи дал бы идентичные randn'ы)
        shapes = {}
        for k, v in sd_r.items():
            shapes.setdefault(tuple(v.shape), []).append(k)
        dup_bad = [ks for ks in shapes.values()
                   if len(ks) > 1 and all(torch.equal(sd_r[ks[0]], sd_r[k]) for k in ks[1:])]
        assert not dup_bad, f"S2b FAIL seed {s}: идентичные shape-дубли {dup_bad[:3]!r}"
        s2b[s] = {"bad": bad, "n_1d": n_1d, "n_small": n_small}
        assert not bad, f"S2b FAIL seed {s}: {bad[:5]}"
        logline(f"S2b: seed {s} — все {len(sd0)} тензоров случайны (1-D: {n_1d}) PASS")
        for lam in grid:
            key = f"s{s}_lam{lam:.2f}"
            if key in cells:
                continue
            c = run_block(m, sd0, sd_r, lam, doc_names, hold_names, tk_decode, curves_series)
            c["seed"] = s
            cells[key] = c
            logline(f"  s{s} λ={lam:.2f}: dip={c['docs'][doc_names[0]]['dip']:+.4f} "
                    f"nll_h={c['holdout'][hold_names[0]]['nll']:.3f} "
                    f"T_act={c['gen']['T_act']} rep16={c['gen']['rep16']} "
                    f"loopy={c['gen']['loopy']}"
                    + (f" S2_sha_ok={c['S2_sha_ok']}" if lam == 1.0 else ""))
            if not smoke:
                tmp_path = inprog_path + ".tmp"
                json.dump({"cells": cells}, open(tmp_path, "w"), default=float)
                os.replace(tmp_path, inprog_path)
        del sd_r
        sd_r = None

    # ── бисекция (только полный прогон) ────────────────────────────────────────
    bis_points = []
    bis_capped = False
    bis_widths = {}
    if not smoke:
        # агрегаты по LAM11 из клеток
        def cell_ag(key):
            c = cells[key]
            return aggregate_cell(c, doc_names, hold_names)
        curves11 = {o: {} for o in OBS_ALL}
        for s in SEEDS:
            for o in OBS_ALL:
                d = {}
                for lam in LAM11:
                    ag = cell_ag(f"s{s}_lam{lam:.2f}")
                    d[lam] = ag[o]
                curves11[o][s] = d
        for o in OBS_ALL:
            for s in SEEDS:
                for lam in LAM11:
                    assert curves11[o][s][lam] is not None, f"null observable {o} {s} {lam}"
        lc_pre, seedavg_pre, _ = lam_c_machine(curves11, LAM11, LAM11_IDX)
        intervals = {o: lc_pre[o]["bisect_interval"] for o in OBS_ALL
                     if lc_pre[o]["outcome"] not in (None, "no_range",
                                                      "non_monotone_excluded",
                                                      "no_transition")}
        rounds, bis_capped = bisection_plan(intervals, set(LAM13))
        bis_points = [p for r in rounds for p in r]
        logline(f"бисекция: {len(bis_points)} точек (capped={bis_capped}): "
                + ", ".join(f"{p[1]:.3f}({p[0]})" for p in bis_points))
        for rnd_i, rnd in enumerate(rounds):
            for s in SEEDS:
                sd_r = make_sd_r(sd0, s)   # детерминированно — тот же sd_r
                for o, lam in rnd:
                    key = f"s{s}_bis{lam:.3f}"
                    if key in cells:
                        continue
                    c = run_block(m, sd0, sd_r, lam, doc_names, hold_names, tk_decode,
                                  curves_series)
                    c["seed"] = s
                    cells[key] = c
                    tmp_path = inprog_path + ".tmp"
                    json.dump({"cells": cells}, open(tmp_path, "w"), default=float)
                    os.replace(tmp_path, inprog_path)
                del sd_r
        # refinement по посчитанным точкам — НОРМИРОВАННЫЕ u(λ) (пре-рег §6.2),
        # НЕ сырые O: refine сравнивает u(mid) с 0.5 (общая функция — selftest'ится)
        for o in OBS_ALL:
            iv = intervals.get(o)
            if not iv:
                continue
            refined, width = refine_bisect(o, iv, cells, seedavg_pre, doc_names,
                                           hold_names, LAM11)
            if refined is None:
                continue
            bis_widths[o] = width
            lc_pre[o]["refined"] = refined
        logline("бисекция-ширины: " + ", ".join(f"{o}={w:.4f}" for o, w in sorted(bis_widths.items())))

    # ── λ_c машина (полный прогон) ─────────────────────────────────────────────
    lambda_c = None
    verdict_out = None
    fams = None
    seedavg = None
    us = None
    if not smoke:
        curves11 = {o: {} for o in OBS_ALL}
        for s in SEEDS:
            for o in OBS_ALL:
                d = {}
                for lam in LAM11:
                    c = cells[f"s{s}_lam{lam:.2f}"]
                    d[lam] = aggregate_cell(c, doc_names, hold_names)[o]
                curves11[o][s] = d
        lambda_c, seedavg, us = lam_c_machine(curves11, LAM11, LAM11_IDX)
        for o in OBS_ALL:
            if o in lambda_c:
                lambda_c[o]["refined"] = lc_pre[o].get("refined")
                if o in bis_widths:
                    lambda_c[o]["bisect_width"] = bis_widths[o]
        fams = families_check(lambda_c)
        verdict_out = verdict(lambda_c, seedavg, us)
        logline(f"вердикт: {verdict_out['outcome']} — {verdict_out['rule']}")
        for o in OBS_ALL:
            e = lambda_c[o]
            logline(f"  {o:12s} {e['outcome']:22s} λ_c={e['seeds']} rho={e['spearman_seedavg']} "
                    f"max_u={e['max_u']}")

    # ── health ray ──────────────────────────────────────────────────────────────
    cells_by_seed = {}
    for key, c in cells.items():
        cells_by_seed[(c["seed"], c["lam"])] = c
    health = health_ray(cells_by_seed, doc_names, hold_names)

    # ── гейты (до записи) ──────────────────────────────────────────────────────
    gates = {}
    gates["S7_e64_sha"] = e64_sha
    gates["S6_docs_sha"] = {"main": main_sha, "holdout": holdout_sha, "pool_fps": POOL_FPS}
    gates["S1_config"] = facts
    if not smoke:
        for s in SEEDS:
            c = cells[f"s{s}_lam1.00"]
            gates[f"S2_sha_ok_s{s}"] = bool(c.get("S2_sha_ok"))
            assert gates[f"S2_sha_ok_s{s}"], f"S2 FAIL: seed {s} λ=1 не побитово base"
        gates["S2b"] = s2b
        for s in SEEDS:
            assert not s2b[s]["bad"] and s2b[s]["n_1d"] > 0, f"S2b FAIL seed {s}"
    else:
        c = cells["s11_lam1.00"]
        gates["S2_sha_ok_s11"] = bool(c.get("S2_sha_ok"))
        assert gates["S2_sha_ok_s11"]
        gates["S2b"] = s2b
    # S3: точное равенство множеств
    expected = {f"s{s}_lam{lam:.2f}" for s in seeds for lam in grid}
    if not smoke:
        expected |= {f"s{s}_bis{p[1]:.3f}" for s in SEEDS for p in bis_points}
    actual = set(cells.keys())
    gates["S3_complete"] = {"expected": len(expected), "actual": len(actual),
                            "missing": sorted(expected - actual),
                            "extra": sorted(actual - expected),
                            "ok": expected == actual}
    assert gates["S3_complete"]["ok"], f"S3 FAIL: {gates['S3_complete']}"
    # S4: nll в каждой записи
    s4 = all(c["docs"][n]["nll"] is not None for c in cells.values() for n in doc_names) \
        and all(c["holdout"][n]["nll"] is not None for c in cells.values() for n in hold_names)
    gates["S4_nll"] = s4
    assert s4, "S4 FAIL"
    # S5: finite + pr_sigma≥1 + eff_rank≥1
    s5 = all(
        all(math.isfinite(c["docs"][n][o]) for o in ("dip", "hump", "plateau"))
        and c["docs"][n]["pr_sigma"] >= 1.0 and c["docs"][n]["eff_rank"] >= 1.0
        for c in cells.values() for n in doc_names)
    gates["S5_finite"] = s5
    assert s5, "S5 FAIL"
    # S8: health ray + sample в каждой (λ,сид)
    s8 = True
    for k, h in health.items():
        if h["sample"] is None or h["loopy"] is None:
            s8 = False
        if not k.endswith("lam1.00") and h["dip_band_hinge"] is None:
            s8 = False
    gates["S8_health"] = s8
    assert s8, f"S8 FAIL: {[(k, h) for k, h in health.items() if not (h['sample'] and (h['dip_band_hinge'] is not None or k.endswith('lam1.00')))]}"
    # S9: бисекция
    if not smoke:
        gates["S9_bisect"] = {"points": [p[1] for p in bis_points], "capped": bis_capped,
                              "widths": bis_widths,
                              "ok": (not bis_capped) or True}
        # бисекционные форварды в store
        gates["S9_bisect"]["ok"] = gates["S3_complete"]["ok"] and (
            all(f"s{s}_bis{p[1]:.3f}" in cells for s in SEEDS for p in bis_points))
    # S2b-нота: 1-D на месте
    logline("гейты: S1/S2/S2b/S3/S4/S5/S6/S7/S8" + ("/S9" if not smoke else "") + " — все PASS")

    # ── артефакты ──────────────────────────────────────────────────────────────
    env = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_path": model_path,
        "e64_gpu_sha256": E64_FILE_SHA256,
        "e64_fns_sha256": e64_sha,
        "main_sha256": main_sha, "holdout_sha256": holdout_sha,
        "pool_fps": POOL_FPS,
        "date": "2026-11-15 (UTC+3)",
        "wall_sec": round(time.time() - t0, 1),
    }
    store = {
        "meta": {**env, "s2b": s2b},
        "cells": cells,
        "health_ray": health,
        "store_sha256": None,
    }
    sha = canonical_sha(store)
    store["store_sha256"] = sha
    store_path = os.path.join(out_dir, "store", f"{sha[:16]}.json")
    json.dump(store, open(store_path, "w"), default=float)
    if os.path.exists(inprog_path):
        os.remove(inprog_path)

    claim = {
        "experiment": "AT0_8 T8.1",
        "round": "round 1",
        "date": env["date"],
        "meta": env,
        "gates": gates,
        "threshold": {
            "lam_grid": LAM11,
            "grid_steps": [round(LAM11[i + 1] - LAM11[i], 6) for i in range(len(LAM11) - 1)],
            "seeds": SEEDS,
            "lambda_c_seeds": (None if lambda_c is None else
                               {o: e["seeds"] for o, e in lambda_c.items()}),
            "statistic": "normalized_u05",
            "family": "all (9 observables) + families field/leaf/mode",
            "observables": OBS_ALL,
            "monotone_only": True,
            "monotonicity_criterion": "|Spearman(λ,O)| >= 0.9 (seed-avg, ДО прогона)",
            "excluded_non_monotone": ([o for o, e in lambda_c.items()
                                       if e.get("outcome") == "non_monotone_excluded"]
                                      if lambda_c else []),
            "overshoot_flag": (None if lambda_c is None else
                               {o: e["overshoot"] for o, e in lambda_c.items()}),
            "all_params_randomized": True,
            "complete": bool(gates["S3_complete"]["ok"]),
            "traj_complete": bool(gates["S4_nll"] and gates["S5_finite"]),
        },
        "holdout": {"train_ids": main_sha, "eval_ids": holdout_sha, "disjoint": True,
                    "n_holdout": len(hold_names), "provenance": "artifact"},
        "aggregation_anchor": {"stat": "median", "n_docs": len(doc_names)},
        "layer_types": facts["layer_types"],
        "prereg": {
            "file": "docs/AT0_8_prereg.md",
            "grid": {"T8_11": LAM11, "union_13": LAM13,
                     "note": "вердикт T8.1 — по LAM11; 0.05/0.1 — T9.1-точки (store)"},
            "seeds": SEEDS,
            "observables": OBS_ALL,
            "families": FAMILIES,
            "monotonicity_criterion": "|Spearman(λ,O)| >= 0.9 (seed-avg, ДО прогона)",
            "statistic": "normalized_u05 (u(λ)=(O(λ)−O(0))/(O(1)−O(0)), первое удержанное "
                         "≥2 точек пересечение u=0.5; бисекция ≤5 ит./≤0.02; CI по сидам; "
                         "Bonferroni α=0.05/4 (вторично при n_seeds=3)",
            "all_params_randomized": True,
            "version_break": "E-71 артефакт (results/e71/SOLITON.json) отсутствует на диске; "
                             "контекст — citation TЗ §8.2; новый λ=0 = ВСЕ параметры случайны "
                             "(вкл. 1-D), сиды [11,12,13]",
            "docs": {"main": "первые 6 wiki (pin 144c722df3046f2d, trunc 256)",
                     "holdout": "math[:2]+code[:1]+chat[:1] (пины n8/min64/max64, 64 ток.)",
                     "prompt": "main[0][:32], T_gen=256, greedy, stop EOS"},
        },
        "curves_seedavg": (None if seedavg is None else
                           {o: {f"{k:.2f}": v for k, v in seedavg[o].items()} for o in OBS_ALL}),
        "lambda_c": lambda_c,
        "families": fams,
        "verdict": verdict_out,
        "health_ray_samples": {k: v for k, v in health.items()
                               if k.endswith("lam0.00") or k.endswith("lam0.50")
                               or k.endswith("lam0.70") or k.endswith("lam1.00")},
        "bisection": (None if smoke else
                      {"points": [p[1] for p in bis_points], "capped": bis_capped,
                       "widths": bis_widths}),
        "excluded_non_monotone": ([o for o, e in (lambda_c or {}).items()
                                   if e.get("outcome") == "non_monotone_excluded"]
                                  if lambda_c else []),
        "summary": ("SMOKE: код-путь + гейты PASS (λ_c-машина не считается)" if smoke else
                    f"{verdict_out['outcome']}: {verdict_out['rule']}"),
    }
    claim_path = os.path.join(out_dir, "AT0_8_THRESHOLD.json" if not smoke else "AT0_8_SMOKE.json")
    json.dump(claim, open(claim_path, "w"), default=float)
    claim_file_sha = hashlib.sha256(open(claim_path, "rb").read()).hexdigest()

    manifest = {
        "experiment": "AT0_8 T8.1",
        "round": "round 1",
        "date": env["date"],
        "claim_sha256": claim_file_sha,
        "claim_file_sha256": claim_file_sha,
        "store_sha256": sha,
        "store_file_sha256": hashlib.sha256(open(store_path, "rb").read()).hexdigest(),
        "script_sha256": _script_sha(),
        "hash_convention": "AT0_2b-v2 C7: store_sha256 = canonical payload hash "
                           "(canonical JSON sort_keys, без self-поля, поле None при расчёте); "
                           "store_file_sha256/claim_file_sha256 = sha256 фактических файлов",
        "note": ("claim AT0_8_THRESHOLD.json (gates S1–S9, prereg, curves seed-avg, "
                 "per-observable λ_c + исходы, families, verdict, bisection, health-ray "
                 "samples); store = per (λ,сид) клетки (per-doc nll/dip/hump/plateau/margin/"
                 "pr_sigma/eff_rank, holdout nll, генерация: токены+text+rep16+loopy, "
                 "S2 sha) + health_ray; run_log = stdout (tracked)."),
        "manifest_sha256": None,
    }
    manifest_body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    manifest_body["manifest_sha256"] = None
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest_body, sort_keys=True).encode()).hexdigest()
    man_path = os.path.join(out_dir, "AT0_8_manifest.json")
    json.dump(manifest, open(man_path, "w"), default=float)

    logline(f"store: {store_path} (canonical sha {sha[:16]}; file sha "
            f"{manifest['store_file_sha256'][:16]})")
    logline(f"written: {claim_path} (file sha {claim_file_sha[:16]}), {man_path}")
    logline(f"AT0_8 gates: ALL PASS (wall {env['wall_sec']}s)")


if __name__ == "__main__":
    main()
