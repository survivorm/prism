# E93 SYCL smoke+bench — qwen3.5-0.8B на Iris Xe (level-zero через ggml-SYCL)

Дата 2026-11-11. Модель `/data/Qwen/qwen3.5-0.8B-f16.gguf` и `-f32.gguf`. Билд `3e88e2517 (b10800)`,
бэкенд SYCL (register: `SYCL0 Intel(R) Iris(R) Xe Graphics` + CPU). Запуск строго по
`~/work/llama.cpp/docs/backend/SYCL.md`: `source /opt/intel/oneapi/setvars.sh`, `ZES_ENABLE_SYSMAN=1`,
флаги `-ngl 99 -sm none -mg 0 --load-mode auto -e -fa on`. `$GGML_SYCL_DEVICE` не ставить (доки:
`--main-gpu` его заменяет). `SYCL_CACHE_PERSISTENT` не включать (доками запрещён).

## История прогонов (важно для доверия к цифрам)

1. **run1 (19:07, кривая команда, убит мной в 19:25)**: `llama-cli` без подавления диалога
   ушёл в REPL на /dev/null-stdin и напечатал 493 МБ приглашений `> `; генерации не было,
   GPU-ядра простаивали (наблюдение пользователя в intel_gpu_top: процесс есть, GPU не
   используется). Артефакт stdout обрезан до 293 КБ и лежит как `smoke_gen_head.txt` (в git не
   класть — мусор-доказательство). Ветвь не считать результативной.
2. **run2 (19:37, канонический)**: `llama-completion` (диалога нет по построению; в этом b10800
   `-no-cnv` из доков уже отвергается парсером, `error: invalid argument: -no-cnv`).
   `RUN_RC=0`, `SMOKE_DONE`. Генерация 256 токенов, greedy — в `smoke_gen.txt` (344 B):
   корректный список Jupiter/Saturn/Uranus.
   - fd-доказательство GPU во время работы (`fd_check.txt`, pid 3446343): два `/dev/dri/renderD128`.
   - Тайминги из `smoke_err.log` (llama-completion их печатает, в отличие от cli run1):
     load 21138 мс (первичный JIT), prompt eval 447.6 мс / 25 ток. = **55.9 t/s**,
     eval 11629.7 мс / 73 = **6.28 t/s**.
3. **bench (e93b/e93s mini)**: таблицы ниже; в e93b f32 fa=off таблица оборвалась (нет строк
   результатов) — помечаем N/A, не догадками.

## Бенч (tg64/pp512, -r 2, -mmp 0; строки из bench_sycl.txt, bench_all.txt, /tmp/e93s.log)

| формат | fa | pp512 t/s | tg64 t/s | источник |
|---|---|---|---|---|
| f16 | on  | 318.0 ± 19.4 | 8.60 ± 3.61 | mini-bench run2 (e93s.log) |
| f16 | on  | 279.9 ± 3.8  | 6.20 ± 2.52 | bench_all.txt (e93b) |
| f16 | off | 193.8 ± 6.5  | 8.93 ± 3.76 | bench_all.txt |
| f32 | on  | 173.8 ± 0.7  | 2.93 ± 0.02 | bench_all.txt |
| f32 | off | N/A (обрыв)  | N/A        | bench_all.txt |

## Сравнение с прежними движками той же карты (qwen3.5-0.8B)

| движок | pp512 t/s | tg t/s |
|---|---|---|
| CPU (llama, этот билд) | 98 | 0.92 |
| Vulkan f32 (прежний билд) | 259 | 5.1 |
| Vulkan f16 (прежний билд) | 196 | 10.5 |
| **SYCL f16 fa=on (этот)** | **280–318** | **6.2–8.6** |

Вывод-минимум: SYCL подтверждён как рабочий GPU-путь (pp заметно быстрее и VK, и CPU; tg того же
порядка, что VK f16). TG-первенство VK за этим дымом не закреплено.

## Кэйвэты (читатель, не сужай по одному числу)

- Билд с **asserts**: `warning: asserts enabled, performance may be affected` в bench_err.log —
  все цифры занижены, межконфигурационные сравнения внутри этого билда корректнее абсолютных.
- Первый прогон компилирует ядра JIT (load 21 с); разброс tg частично — разогрев внутри r=2.
  Кандидаты на следующий заход: AOT `GGML_SYCL_DEVICE_ARCH` (детект `intel_gpu_adl_p`), release-
  пересборка без asserts, `-r 5`.
- fa-эффект неустойчив (pp on>off, tg on<off) — на 2 повторах не решать.
- `[TEMP-SDPA] has_xmx=0 arch=12582912 family=0` — ADL без XMX, форк сам логирует; не ошибка.

## Не тронуто / не свершено

- exp91 и e83 (CPU) не трогались, чужие процессы не убивались.
- L2 полки (logits-all стрим по 16 докам) и L1 (--slot-save-path/REPLAY-двойник) — следующие
  прогоны, к этому коммиту отношения не имеют.

S93_SMOKE_DONE
