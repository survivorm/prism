"""Rebuild docs/paper.md from paper_old.md sections + current guide + §5/§6 + §4 blocks."""
import re

cur = open('docs/paper.md', encoding='utf-8').read().splitlines(keepends=True)
old = open('docs/paper_old.md', encoding='utf-8').read().splitlines(keepends=True)

g0 = next(i for i, l in enumerate(cur) if l.startswith('## 0.'))
g5 = next(i for i, l in enumerate(cur) if l.startswith('## 5.'))
head = ''.join(cur[:g0])
s56 = ''.join(cur[g5:])

def sec(a, b):
    return ''.join(old[a - 1:b])

s1, s2, s3 = sec(52, 70), sec(71, 134), sec(135, 165)
s4, s5, s6, s7, s8 = sec(166, 203), sec(204, 233), sec(234, 283), sec(284, 328), sec(329, 398)

remap = {
    '1': '1', '2': '2',
    '3': '1.2',
    '4': '3.1', '5': '3.2', '6': '3.3', '7': '3.4', '8': '3.5',
    '9': 'ref-impl §1', '10': 'ref-impl §2', '11': 'ref-impl §3', '12': 'ref-impl §4',
    '13': 'ref-exp §1', '14': 'ref-exp §2',
    '15': 'ref-exp 4.3', '16': 'ref-exp 4.4', '17': 'ref-exp 4.6',
    '18': 'ref-exp 4.5,4.7', '19': 'ref-exp 4.8',
    '20': 'paper §5', '21': 'paper §6.1', '22': 'paper §6.2',
}

def remap_refs(t):
    t = re.sub(r'v3\.3\s+(§\d{1,2}(?:\.\d+)?)', lambda m: 'v3.3 \x00' + m.group(1), t)
    t = re.sub(r'(?<![0-9.])§(\d{1,2})(?!\d)',
               lambda m: '§' + remap.get(m.group(1), m.group(1)), t)
    t = t.replace('\x00', '')
    t = t.replace('Приложение A', 'ref-math §2').replace('Приложение B', 'ref-exp')
    t = t.replace('Приложение C', 'ref-impl').replace('Приложение D', 'ref-math §3')
    t = t.replace('Приложение E', 'ref-math §4')
    t = t.replace('Math.md §3', 'ref-math §3')
    t = t.replace('в Части II', 'в paper §2–3')
    t = t.replace('Часть V', 'ref-exp')
    return t

p1 = (remap_refs(s1).replace('## 1. Мотивация, цели, статус',
                             '## 1. Мотивация и архитектура\n\n### 1.1 Мотивация, цели, статус'))
p3 = (remap_refs(s3).replace('## 3. Архитектурный контекст: Qwen3.5',
                             '### 1.2 Архитектурный контекст: Qwen3.5'))
p2 = remap_refs(s2)
p3m = '\n'.join([
    '## 3. Метод: Якобиан, Грамиан-коррекция, ESG, gap/bridge, gauge',
    '',
    remap_refs(s4).replace('## 4. Якобиан и модели', '### 3.1 Якобиан и модели'),
    remap_refs(s5).replace('## 5. Лемма о размерностях и Gramian-коррекция',
                           '### 3.2 Лемма о размерностях и Грамиан-коррекция'),
    remap_refs(s6).replace('## 6. ESG, спектральные метрики, ранги',
                           '### 3.3 ESG, спектральные метрики, ранги'),
    remap_refs(s7).replace('## 7. Gap и bridge: определения и направления',
                           '### 3.4 Gap и bridge: определения и направления'),
    remap_refs(s8).replace('## 8. Gauge-оператор и производные блоков',
                           '### 3.5 Gauge-оператор и производные блоков'),
])

blocks = """## 4. Применение: 4 блока экспериментов

**Блок A: метод (E-0, E-13, E-33).** Валидация метода (калькулятор) и эталонные пороги:
E-0 (калибровка), E-13 (топология/домены), E-33 (gauge-режим; редизайн E-33r).
Статус: E-0, E-13, E-33r — выполнены; E-33 preflight — закрыт (отрицательный контроль).

**Блок B: обучение (E-34, E-35).** Применение к L1: cross-coupling токенов (E-34:
rho_gf = 0.064–0.069 на L6–L23 — per-token защита спасаема в первом порядке),
листовая геометрия (E-35: L23 — единственный устойчивый лист, плоский, kappa = 0.018).

**Блок D: модели (E-16, E-17).** L3-сравнение: bridge как causal lever (E-16a:
readout-редакция движет мост; E-16b: Goodhart на точечной метрике), LoRA-дистилляция
(Exp 17–19 — P2, §6.2).

**Блок F: фреймворки (E-14, E-15).** L2: gauge share немонотонна (E-14: peak 0.48 (L3)
→ 0.08 (L23)), конвергенция моста не артефакт фазы (E-15).

**Блоки C (E-21..E-24, диагностика) и E (E-25, E-26, топология/шум)** — `ref-exp.md`.
Все карточки (26) — `ref-exp.md`, «Индекс экспериментов и реестр гипотез»;
paper содержит только блок-саммари и ссылки.

---

"""

out = head + p1 + '\n' + p3 + '\n' + p2 + '\n' + p3m + '\n---\n\n' + blocks + s56
open('docs/paper.md', 'w', encoding='utf-8').write(out)
print('total lines:', len(out.splitlines()))
