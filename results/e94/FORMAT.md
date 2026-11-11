# E94 SPIKE — схема ответа, гейты, доступность top-k

## Запуск
- gguf: `qwen3.5-0.8B-f16.gguf`
- argv: `/home/survivor/work/llama.cpp/build/bin/llama-server -m /data/Qwen/qwen3.5-0.8B-f16.gguf -sm none -mg 0 -ngl 99 -fa on -c 2048 -np 1 --slots --host 127.0.0.1 --port 8110 --log-disable`
- port=8110 server_pid=29425 load=10.1s total=-1.0s
- fd-гейт: 20/20 сэмплов (0.05с) с /dev/dri во время decode (n_predict=64, wall=4.8s); union=['/dev/dri/renderD128']; fd after health (контекст): ['/dev/dri/renderD128', '/dev/dri/renderD128']

## Фактическая схема ответа /completion (полные имена ключей, w0)
- `index`
- `content`
- `tokens`
- `id_slot`
- `stop`
- `model`
- `tokens_predicted`
- `tokens_evaluated`
- `generation_settings`
- `generation_settings.seed`
- `generation_settings.temperature`
- `generation_settings.dynatemp_range`
- `generation_settings.dynatemp_exponent`
- `generation_settings.top_k`
- `generation_settings.top_p`
- `generation_settings.min_p`
- `generation_settings.top_n_sigma`
- `generation_settings.xtc_probability`
- `generation_settings.xtc_threshold`
- `generation_settings.typical_p`
- `generation_settings.repeat_last_n`
- `generation_settings.repeat_penalty`
- `generation_settings.presence_penalty`
- `generation_settings.frequency_penalty`
- `generation_settings.dry_multiplier`
- `generation_settings.dry_base`
- `generation_settings.dry_allowed_length`
- `generation_settings.dry_penalty_last_n`
- `generation_settings.dry_sequence_breakers`
- `generation_settings.mirostat`
- `generation_settings.mirostat_tau`
- `generation_settings.mirostat_eta`
- `generation_settings.adaptive_target`
- `generation_settings.adaptive_decay`
- `generation_settings.stop`
- `generation_settings.max_tokens`
- `generation_settings.n_predict`
- `generation_settings.n_keep`
- `generation_settings.n_discard`
- `generation_settings.ignore_eos`
- `generation_settings.stream`
- `generation_settings.logit_bias`
- `generation_settings.n_probs`
- `generation_settings.min_keep`
- `generation_settings.grammar`
- `generation_settings.grammar_lazy`
- `generation_settings.grammar_triggers`
- `generation_settings.preserved_tokens`
- `generation_settings.chat_format`
- `generation_settings.reasoning_format`
- `generation_settings.reasoning_in_content`
- `generation_settings.generation_prompt`
- `generation_settings.samplers`
- `generation_settings.speculative.types`
- `generation_settings.timings_per_token`
- `generation_settings.post_sampling_probs`
- `generation_settings.backend_sampling`
- `generation_settings.lora`
- `prompt`
- `has_new_line`
- `truncated`
- `stop_type`
- `stopping_word`
- `tokens_cached`
- `timings`
- `timings.cache_n`
- `timings.prompt_n`
- `timings.prompt_ms`
- `timings.prompt_per_token_ms`
- `timings.prompt_per_second`
- `timings.predicted_n`
- `timings.predicted_ms`
- `timings.predicted_per_token_ms`
- `timings.predicted_per_second`
- `completion_probabilities`
- `completion_probabilities[0].id`
- `completion_probabilities[0].token`
- `completion_probabilities[0].bytes`
- `completion_probabilities[0].logprob`
- `completion_probabilities[0].top_logprobs`
- `completion_probabilities[0].top_logprobs[0].id`
- `completion_probabilities[0].top_logprobs[0].token`
- `completion_probabilities[0].top_logprobs[0].bytes`
- `completion_probabilities[0].top_logprobs[0].logprob`
- `completion_probabilities[1].id`
- `completion_probabilities[1].token`
- `completion_probabilities[1].bytes`
- `completion_probabilities[1].logprob`
- `completion_probabilities[1].top_logprobs`
- `completion_probabilities[1].top_logprobs[0].id`
- `completion_probabilities[1].top_logprobs[0].token`
- `completion_probabilities[1].top_logprobs[0].bytes`
- `completion_probabilities[1].top_logprobs[0].logprob`

## Гейты
- fd_gate: {"n_samples": 20, "interval_s": 0.05, "n_samples_with_dri": 20, "dri_union": ["/dev/dri/renderD128"], "request_wall_s": 4.834, "request_timings": {"cache_n": 0, "prompt_n": 64, "prompt_ms": 506.038, "prompt_per_token_ms": 7.90684375, "prompt_per_second": 126.47271548776969, "predicted_n": 64, "predicted_ms": 4235.783, "predicted_per_token_ms": 67.2346507936508, "predicted_per_second": 14.873283168660905}, "ok": true}
- gate_prompt_count: {"ok": true, "detail": "tokens_evaluated=64 sent=64"}
- gate_schema: {"ok": true, "detail": "cp_len=64 tokens_len=64"}
- gate_top_logprobs_len: {"tz_required": 2, "observed": {"1": 64}, "ok": false, "treated_as": "finding (steering): определяет контракт Q2, не hard-гейт"}
- gate_generation_settings: {"ok": true, "missing_field": false, "mismatch": {}, "got": {"seed": 8181, "temperature": 0.0, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_last_n": 0, "dry_multiplier": 1.0, "dry_base": 1.0, "xtc_probability": 0.0, "mirostat": 0, "n_predict": 64}, "note": "logprobs:true(bool)→n_probs=1; top_logprobs вне legacy-схемы (server-schema.cpp:179 поле n_probs, alias logprobs)"}
- gate_tokenize: null
- gate_cache:
  - k=64: {"wall_s_r2": 0.421, "per_pos_s": 0.421, "slots": {"n_prompt_tokens": 64, "n_prompt_tokens_processed": 0, "n_prompt_tokens_cache": 0}, "ok": false}
  - k=128: {"wall_s_r2": 0.349, "per_pos_s": 0.349, "slots": {"n_prompt_tokens": 64, "n_prompt_tokens_processed": 0, "n_prompt_tokens_cache": 0}, "ok": false}
  - k=192: {"wall_s_r2": 0.349, "per_pos_s": 0.349, "slots": {"n_prompt_tokens": 64, "n_prompt_tokens_processed": 0, "n_prompt_tokens_cache": 0}, "ok": false}

## Свип top-k (w0, n_predict=4) — len(top_logprobs) по позициям
- tl2 (overrides={'top_logprobs': 2}): len_hist={1: 4}
- tl5 (overrides={'top_logprobs': 5}): len_hist={1: 4}
- tl20 (overrides={'top_logprobs': 20, 'n_probs': 20}): len_hist={20: 4}
- np2 (overrides={'n_probs': 2}): len_hist={2: 4}
- np5 (overrides={'n_probs': 5}): len_hist={5: 4}
- A (tl20, цепочка как есть): len_hist={20: 4}; N(A)=400 (проба n_probs=400)
- B (tl20+n_probs=20+top_k:400,top_p:1.0,min_p:0.0,typical_p:1.0,xtc:0.0): len_hist={20: 4}; N(B)=400 (проба n_probs=400)

## (A) vs (B): p_max и ренормировка
- A: [{"pos": 0, "id": 220, "p_max": 0.25749010867949856, "n_tl": 20, "sum_list": 0.6900501835809862}, {"pos": 1, "id": 18, "p_max": 0.1904362835964774, "n_tl": 20, "sum_list": 0.9995544300751253}, {"pos": 2, "id": 20580, "p_max": 0.3046322001440712, "n_tl": 20, "sum_list": 0.9144538103565173}, {"pos": 3, "id": 523, "p_max": 0.22423374087449635, "n_tl": 20, "sum_list": 0.7148762187406601}]
- B: [{"pos": 0, "id": 220, "p_max": 0.2573825444314792, "n_tl": 20, "sum_list": 0.6900194627982923}, {"pos": 1, "id": 18, "p_max": 0.19041578499780593, "n_tl": 20, "sum_list": 0.9995542532021126}, {"pos": 2, "id": 20580, "p_max": 0.3046783963113737, "n_tl": 20, "sum_list": 0.9144724355568377}, {"pos": 3, "id": 523, "p_max": 0.22413480548904868, "n_tl": 20, "sum_list": 0.7148155365679821}]
- median |p_max(B)−p_max(A)| = 7.256577637508588e-05 (per_pos=[0.00010756424801938724, 2.0498598671481183e-05, 4.619616730250353e-05, 9.893538544766822e-05]) → (A) ренормирована по выжившему множеству

## (C) жадность
- C1 (Q1-тело, 384 токов, wall=26s) vs C2 ((B)-тело, 384, wall=26s): agree 384/384 = 100.0%; first_diff=None; Q1 переводится на (B): False

## Выводы (канон)
- None
- None
- None
- КАНОН: тело запроса для Q1/Q2/Q3 = (B) {n_probs:20, top_logprobs:20, top_k:400, top_p:1.0, min_p:0.0, typical_p:1.0, xtc_probability:0.0} + поля ТЗ (temperature:0, stream:false, return_tokens:true, cache_prompt, seed:8181, нейтральные штрафы), потому что только в (B) len(top_logprobs)=20 стабилен, p_max нормирована по фиксированному N=400, а жадность совпадает с Q1-телом на 100%%

## Время на позицию (клиентский wall-clock, с)
- main (n_predict=64): wall=4.8 → per_pos=0.0755
- cache k=64: per_pos=0.421
- cache k=128: per_pos=0.349
- cache k=192: per_pos=0.349

## Рекомендованный скоуп Q2
t_per_pos ≈ 0.4210 с (max). Полный: 13 точек k × 16 доков = 208 запросов (decode ≈ 88 с); минимальный: 2 дока (w0, w8) × 13 точек. Сторона transformers: fp32 .venv-референс (валидирован: max|Δ| против REPLAY.gen_p1 = 2.2e-6, top1 6144/6144) — teacher-forced один forward на (prompt+gen) даёт p на всех k сразу, поэтому Q2 не ограничено 90 мин.

## Отклонения
- `--slots` оставлен явно, но он включён ПО УМОЛЧАНИЮ (common/common.h:671 endpoint_slots = true).
- тело запроса: `top_logprobs` не является полем legacy-/completion-схемы и игнорируется; реальное поле — `n_probs` (alias `logprobs`; bool true → 1). Фиксируется свипом.
- серверный лог: /tmp/e94spike_server.log; raw: /tmp/e94raw/*.json

## Метрики Q1/Q2/Q3
- TODO-заглушки в этом прогоне (spike only); см. docs/dynamic/0_E94_EXEC_NOTES.md.

## HALT
- rc=2 halt_reason=cache gate failed: {"64": {"wall_s_r2": 0.421, "per_pos_s": 0.421, "slots": {"n_prompt_tokens": 64, "n_prompt_tokens_processed": 0, "n_prompt_tokens_cache": 0}, "ok": false}, "128": {"wall_s_r2": 0.349, "per_pos_s": 0.349, "slots": {"n_prompt_tokens": 64, "n_prompt_tokens_processed": 0, "n_prompt_tokens_cache": 0}, "ok": false}, "192": {"wall_s_r2": 0.349, "per_pos_s": 0.349, "slots": {"n_prompt_tokens": 64, "n_prompt_tokens_processed": 0, "n_prompt_tokens_cache": 0}, "ok": false}}
