"""Протокол переноса вычислений prism на iGPU (Iris Xe / SYCL / torch.xpu) — v1.0.

Этот файл — машинно-читаемая часть. Человеческая часть: `PROTOCOL.md`.
Проверка артефакта: `python3 protocol/check_artifact.py <artifact.json>`.

Смысл протокола. После AT0_4 (решающая клетка 3×2) главный эффект серии стал МАЛЫМ:
  обученная модель: r(real) − r(mix) ≈ 0.0158 (P3 confirmed);
  случайные модели: |r(real) − r(mix)| ≈ 0.001–0.002 (P4 null, 6/6).
То есть объект — не «поле вообще», а его ТЕКСТОВАЯ ЧУВСТВИТЕЛЬНОСТЬ, а она по величине
близка к шумовому полу. Значит дальше решает не «посчитать ещё раз», а «посчитать
достаточно null'ов, чтобы разрешить разницу ~0.001» ⇒ ⇒ именно нулевой слой
(перестановочные null'и, бутстрапы) становится главным по объёму, и именно он
идеально параллелится. Поэтому протокол существует.

ЖЁСТКИЕ ПРАВИЛА (нарушение = артефакт невалиден):
  R1. Ни одно число не публикуется без `backend` и `device` в артефакте.
  R2. Молчаливый фолбэк на CPU запрещён: если упало/не поддержано — пишется
      `fallback_reason`, и число помечается как CPU-произведённое.
  R3. Наблюдаемая статистика и её null считаются в ОДНОМ формате (dtype) либо
      расхождение объявлено явно (`systematic_note`).
  R4. GEMM на iGPU — только fp32. TF32-подобное ускорение запрещено: на Intel его нет,
      но если появится — портит малые KL (10 бит мантиссы).
  R5. fp64- reductions допустимы только там, где их стоимость мала; крупные
      fp64-сводки по словарю (248k) остаются на CPU.
  R6. Перенос не меняет определение объекта: k-алфавит, T-нормировка, агрегатор,
      фильтр вырожденности — как в `prism/field.py` (канон sha bcc607c5).
  R7. Каждый перенос проходит три контроля (см. `controls`): аналитический,
      паритетный, инвариантный. Провал любого ⇒ перенос отклоняется.
  R8. Критерий приёмки: выигрыш ≥ acceptance.min_speedup НА ЦЕЛЕВОЙ ФОРМЕ
      (не на синтетике меньшего размера) + все контроли зелёные.
"""

PROTOCOL = {
    "protocol": "prism-gpu-transfer",
    "version": "1.0",
    "date": "2026-11-14",
    "motivating_result": {
        "status": "PRELIMINARY - цикл ревью AT0_4 не завершён (ТЗ-ревью r3 запущен, результата нет)",
        "cell": "AT0_4 (3x2: R0 trained / R1 random trunk / R2 random all x real / mix)",
        "observed": {
            "P1_R2_real_orthogonal_walk": "vacuous 3/3 (r_max=0.0137..0.3128) - предсказано априори, информации нет",
            "P2_R1_vs_R2_real": "|Δr| = 0.00101..0.00180 < 0.1 => 'same' (порог априорный, в 10-30x выше наблюдаемых)",
            "P3_R0_real_vs_mix": "median Δr = +0.01581; 6/8 положительны (+0.0094..+0.0231); own_w0 -0.0535, own_w4 -0.0397",
            "P4_random_real_vs_mix": "медиана 0.00074..0.00183 => null; burst-доки own_w0 0.086..0.110, own_w4 0.119..0.131",
            "P5_R0_f_DC": "+4.127e-05 - зеркало P3 по G3, не независим",
        },
        "what_survives": (
            "на 6 не-burst доках trained Δr = +0.009..+0.023 против random |Δr| <= 0.0023 (~10x): "
            "текстовая чувствительность выучена - предварительно (правило 'медиана>0' слабое, n=6, "
            "пер-доковой мощности нет)"
        ),
        "what_does_not_survive": (
            "'база поля архитектурна' НЕ лицензировано: тот же порог 0.1 сказал бы 'same' и для "
            "trained-vs-random (различие того же порядка ~0.002)"
        ),
        "implication": (
            "Измеряемая величина мала (0.001-0.02) => решает шум и объём null'ей, а не число "
            "прогонов => перенос нулевого слоя на iGPU - условие разрешимости."
        ),
    },
    "required_artifact_fields": [
        "backend",              # "cpu" | "sycl" | "torch_xpu" | "mixed"
        "device",               # имя устройства / "cpu"
        "protocol_version",     # "1.0"
        "gemm_dtype",           # "fp32" (R4: никакого TF32)
        "reduce_dtype",         # "fp32" | "fp64"
        "control_passed",       # bool: три контроля из `controls` зелёные
        "parity_max_abs",       # максимум |CPU - GPU| по контролируемым величинам
        "speedup_vs_cpu",       # float (или None, если измерялось не на целевой форме)
    ],
    "optional_fields": [
        "fallback_reason",      # обязателен, если backend != "cpu" и вычисление упало
        "systematic_note",      # обязателен, если dtype наблюдаемой != dtype null
        "per_doc_exceptions",   # обязателен, если применяется медианный тест (см. AT0_4)
        "target_shape",         # форма, на которой мерился выигрыш (T, D, K, ...)
    ],
    "stages": {
        "base_pre": {
            "label": "GEMM по словарю + logsumexp (fp32), fp64-сводки",
            "device": "gpu_ok",
            "gemm_dtype": "fp32",
            "reduce_dtype": "fp32",
            "controls": ["E45c_analytic", "parity", "order_invariance"],
            "why": "вычислительно-интенсивен (N x D x V), fp32 на Intel без TF32",
        },
        "dense_metrics": {
            "label": "метрики правки: KL/TV/flip по 248k-словарю",
            "device": "gpu_ok",
            "gemm_dtype": "fp32",
            "reduce_dtype": "fp64",
            "controls": ["E45c_analytic", "parity"],
            "why": "сводки по словарю в fp64 (ошибка на 248k слагаемых) ⇒ либо CPU, "
                   "либо явная запись systematic_note",
        },
        "sparse_metrics": {
            "label": "разреженные метрики правки",
            "device": "gpu_ok",
            "gemm_dtype": "fp32",
            "reduce_dtype": "fp64",
            "controls": ["E45c_analytic", "parity"],
        },
        "teacher_forced_forward": {
            "label": "teacher-forced форвард + per-position hidden states",
            "device": "gpu_ok",
            "gemm_dtype": "fp32",
            "reduce_dtype": "fp32",
            "controls": ["head_acc_liveness", "parity"],
            "why": "prefill-режим на iGPU силён (E-93: pp512 = 174..307 tok/s), "
                   "decode слаб ⇒ генерация корпусов на iGPU не выносим",
        },
        "dcurve": {
            "label": "кривая прибора (mean/median per-t косинусов)",
            "device": "gpu_ok",
            "gemm_dtype": "n/a",
            "reduce_dtype": "fp32",
            "controls": ["parity", "G1", "G3"],
        },
        "vpath": {
            "label": "касательные прибора (ортогонализованный тангенс)",
            "device": "gpu_ok",
            "gemm_dtype": "n/a",
            "reduce_dtype": "fp32",
            "controls": ["parity"],
        },
        "acf_moments": {
            "label": "ACF прибора (c(k), r, f_DC, rho_Tk)",
            "device": "gpu_ok",
            "gemm_dtype": "n/a",
            "reduce_dtype": "fp32",
            "controls": ["parity", "G1", "G3", "G7_order_invariance"],
        },
        "perm_null": {
            "label": "перестановочный null-слой (block-2, лаги 3,4,9..K)",
            "device": "gpu_ok",
            "gemm_dtype": "n/a",
            "reduce_dtype": "fp32",
            "controls": ["parity", "rng_stream_unchanged"],
            "why": "ГЛАВНЫЙ объём: 60 сидов x 53 дока x 3 объекта x 90 лагов; "
                   "на 32k-игле ≈ 20 часов на CPU ⇒ на iGPU — минуты. "
                   "RNG остаётся на хосте (numpy default_rng), на GPU уезжает арифметика",
        },
        "bootstrap_ci": {
            "label": "пер-центный бутстрап по документам",
            "device": "gpu_ok",
            "gemm_dtype": "n/a",
            "reduce_dtype": "fp32",
            "controls": ["parity"],
        },
    },
    "never_transfer": {
        "kappa_of": "fp64 eigh грамма: на Xe-LP fp64 = 1/64 скорости ⇒ остаётся на CPU",
        "spectral_flags": "то же (eigh в fp64)",
        "drift_med": "‖ΔJ‖/‖J‖ в fp64",
        "autograd_jacobian": "нужен autograd + кастомные CUDA-ядра GDN ⇒ T4/Kaggle",
        "hessian": "fp64 плотная линейная алгебра",
    },
    "controls": {
        "E45c_analytic": {
            "what": "правка W → W + 1·cᵗ ⇒ softmax по словарю инвариантен ⇒ "
                    "KL/TV/flip обязаны быть 0",
            "tolerances": {"max_kl": 1e-4, "max_tv": 1e-4, "flip": 0},
            "ground_truth": True,
        },
        "parity": {
            "what": "|CPU - GPU| по контролируемым величинам на целевой форме",
            "tolerances": {"fp32": 1e-4, "fp16": 5e-3},
        },
        "order_invariance": {
            "what": "G7: r и Σρ̂ инвариантны к перестановке приращений (полный размах)",
            "tolerances": {"r": 1e-9, "sum_rho_full": 1e-12},
        },
        "G1": {"what": "r = 1 + 2Σρ̂ (полный размах)", "tolerances": {"abs": 1e-9}},
        "G3": {"what": "r = N·f_DC", "tolerances": {"abs": 1e-9}},
        "G7_order_invariance": {"what": "см. order_invariance"},
        "rng_stream_unchanged": {
            "what": "индексы перестановок совпадают побитово с numpy-протоколом",
            "tolerances": {"exact": True},
        },
        "head_acc_liveness": {
            "what": "head-accuracy обученной модели ≥ 0.90 (S1 из AT0_4)",
            "tolerances": {"min": 0.90},
        },
    },
    "acceptance": {
        "min_speedup": 2.0,
        "target_shape_required": True,
        "note": "выигрыш измеряется на ЦЕЛЕВОЙ форме (напр. N=4096, V=248k), "
                "не на синтетике малого размера; иначе перенос не принимается",
    },
    "forbidden": [
        "молчаливый фолбэк на CPU без записи в артефакт",
        "TF32/bf16-ускорение GEMM в метриках, где чувствительны малые KL",
        "смешение форматов наблюдаемой и null без явной systematic_note",
        "перенос определения объекта (k-алфавит, T-нормировка, агрегатор, фильтр)",
        "публикация числа без backend/device в артефакте",
    ],
}
