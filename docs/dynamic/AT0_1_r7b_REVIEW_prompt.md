# Повторное внешнее ревью (round 3b): AT0_1 r7 v2 — MINOR-фиксы

Роль: внешний ревьюер. Предыдущее ревью (round 3, этот же файл: docs/dynamic/AT0_1_r7_REVIEW.md)
дало PASS с 6 MINOR. Исполнитель применил фиксы — claim изменился
`2230226141a9a697fa8ba6118f0a5585efc1b8ea650e8f3a3a377e75a8d04db2` →
`82f44f0813f6adde047f11b97568f08ca19807fb41af145b2977f3fd450999c5` (r7 v2).

Задача: независимая верификация, что (1) изменения — только текст/даты (НИЧЕГО численного),
(2) все 6 MINOR закрыты по существу, (3) цепочка пинов согласована.

## Файлы
- Старый r7-claim (v1): `results/AT0/store/claim_r7_2230226141a9a697.json`
  (если удалён — sha 22302261…; восстановить нельзя — тогда проверять по ответке)
- Новый r7-claim (v2, текущий): `results/AT0/T0_ACCOUNTING.json`
  (пин `results/AT0/store/claim_r7_82f44f0813f6adde.json`)
- Manifest: `results/AT0/AT0_1_manifest.json`
- Pины: `results/AT0/AT0_1_artifact_hashes.json`
- Скрипт r7: `experiments/at0_1_r7_annotation.py` (обновлён)
- R6-скрипт (новый пин): `experiments/at0_1_r6_annotation.py` (sha 0d52eda5f1f4128aebe1bfdce9f3c325…;
  оригинал — /tmp/recovered/AT0_1_r6/apply_r6_edits.py, если доступен — сверь bit-eq)
- Ответка: `docs/AT0_1_review_r7_answer.md` (секция «Round 3»)
- Лог запуска v2: `/tmp/at01_r7b.log` (RC=0, маркер R7_ANNOTATION_DONE)

## Обязательные проверки
1. **Числовой diff v1→v2:** json-сравнение по всем листьям — перечисли ВСЕ отличающиеся пути.
   Критерий: отличающиеся листья — только строковые поля (notes/status/provenance/date/
   r6_annotation_script_sha256/R3-строка в changes); ни одно числовое значение (z, median,
   mean, p, d_eff, se_factor, roundoff, gram dev, n_*, sha в claim-телах кроме chain) не
   изменилось. Любой числовой diff = MAJOR.
2. **MINOR-1:** meta.r7_annotation.date = «2026-11-15»; reconstruction_roundoff_identity_detail.
   provenance содержит «2026-11-15».
3. **MINOR-2:** c1_c4_note содержит ноту о thread-pinned bit-eq (d_eff/z_excess_eff/
   se_factor_median; OpenBLAS/MKL=6; ≤3.4e-15 rel в другом env).
4. **MINOR-3:** experiments/at0_1_r6_annotation.py существует; sha256 =
   meta.r7_annotation.chain.r6_annotation_script_sha256; (если /tmp/recovered доступен)
   bit-eq с apply_r6_edits.py.
5. **MINOR-4:** t0_headline_note + identity_check_G1.headline: r/f_dc ≡ N (trivial) отмечено;
   headline = r per object (r_v ≈ 24.9, r_du ≈ 0.0048).
6. **MINOR-5/6:** c1_c4_note: guard th is None — defensive note; дубль median_z/z_raw_band —
   note «цитировать z_raw_band».
7. **Цепочка:** manifest = claim v2 + manifest-блок (claim_sha256 = 82f44f08…); пин = claim
   bit-eq; AT0_1_artifact_hashes.json: claim_sha256 = 82f44f08…, pins полные (r5/r6/r7);
   store 23281c64 и script de305869 — не тронуты (sha).
8. **Регрессия:** z_raw_band medians (381.3249665142485 / −0.21574175798491185 /
   −0.21562117252716537), z_excess v_tangent median 10.547543645305291, T_none_full identity
   (−0.5023582212969491 / −12.868956825537097), roundoff 3.7939126356469717e-16 — в v2
   нетронуты (bit-eq с v1).

## Ограничения
- НЕ менять файлы репо (чтение + вычисления в /tmp; запись — только addendum в
  docs/dynamic/AT0_1_r7_REVIEW.md).
- Числа — только из фактических пересчётов/артефактов.

## Формат вывода
В конец docs/dynamic/AT0_1_r7_REVIEW.md дописать секцию:

```
## 4. Addendum round 3b (r7 v2, claim 82f44f08)
- Числовой diff v1→v2: <n> листьев, только текст: <список путей или «только notes/dates/chain»>
- MINOR-1..6: <закрыт/не закрыт — доказательство>
- Цепочка: <OK/NA>
- Регрессия: <OK/NA>
- Вердикт: PASS/FAIL (+ MAJOR/MINOR, если есть)
```

Финальный маркер в stdout: `REVIEW_DONE`.
