"""Общий execution-логгер для эксперимент-скриптов.

Вынос из ~14 дублей в experiments/ (at0_7_core, at06_v22_pinned, AT0_8, AT0_1, …):

    _log = []
    def logline(s=""):
        print(s, flush=True)
        _log.append(s)

Drop-in:

    _log = ExpLog()
    logline = _log
    ...
    _log.write_to(path)      # == f.write("\n".join(lines) + "\n")

Семантика: print с flush (по умолчанию) + буфер ``.lines``. Вариант без flush
(legacy AT0_3) — ``ExpLog(flush=False)``.

Зависимостей нет (stdlib) — можно spec-лоадить, минуя prism/__init__.py
(тот тянет datasets/transformers; паттерн AT-экспов — spec_from_file_location).
"""

__all__ = ["ExpLog"]


class ExpLog:
    """Buffered logger: print(flush) + буфер .lines + write_to."""

    def __init__(self, flush: bool = True) -> None:
        self.lines: list = []
        self._flush = flush

    def __call__(self, s: str = "") -> None:
        print(s, flush=self._flush)
        self.lines.append(s)

    def write_to(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self):
        return iter(self.lines)
