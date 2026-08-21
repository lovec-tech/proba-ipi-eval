#!/usr/bin/env python3
"""The triviality floor: regexes for the most quoted Russian injection phrases.

WHAT IT IS FOR. Nobody ships this, and it is not competing. It exists so every number in every
report has something to be read against: "recall 13%" means nothing on its own; "recall 13% where
a handful of regexes already get N%" says whether the detector is worth more than a grep. Two
things fall out of it:

  * a lower bound on the corpus — a detector scoring below the floor is not detecting anything;
  * a reading of the corpus itself: the floor's own recall is roughly the share of this set that is
    quotable boilerplate. That it is small is the point.

TRANSLATED, NOT COPIED. Quadrat-IPI's floor is five English phrases ("ignore previous
instructions", "system prompt", …). Run against Russian carriers it reads ~0% and would say
nothing about this corpus — the number would be about the language, not about triviality. So the
phrases here are the Russian equivalents, in the forms a Russian-language injection actually uses,
with the English ones kept alongside because an injection written in Russian text may still quote
the English formula. The two floors are therefore NOT comparable line-for-line, and the card must
not present them as one number measured twice.

BINARY, and declared so. The runner picks a threshold at a target FPR for scored detectors, and on
a 0/1 vector that degenerates: with fewer firings than the budget allows it selects tau = 0 and
`score >= tau` flags the whole corpus — reported as 100% recall at 100% FPR, which looks like a
result. `binary = True` routes this past threshold selection to a fixed 0.5.

Called `floor`, not `baseline`: every system measured here is a baseline in the paper's sense.
The file is not named `keyword.py` — that shadows a standard-library module `collections` imports.
"""
from __future__ import annotations

import re

from ..detector import Detector, register

#: The quotable formulas. Kept few and obvious on purpose: a floor that grew until it competed
#: would stop being a floor and start being an undocumented detector.
RX = re.compile(
    r"(?i)("
    # --- Russian ---
    r"игнорируй\s+(все\s+)?(предыдущие|прежние|выше)"
    r"|не\s+обращай\s+внимани\w*\s+на\s+(предыдущ|прежн|выше)"
    r"|забудь\s+(все\s+)?(предыдущие|прежние|инструкции)"
    r"|(системн\w+\s+(промпт|инструкц\w+))"
    r"|теперь\s+ты\s+"
    r"|как\s+языковая\s+модель"
    r"|отмен\w+\s+(все\s+)?(предыдущие|прежние)\s+инструкц"
    # --- English, quoted verbatim inside Russian text ---
    r"|ignore\s+(all\s+)?(previous|prior)\s+instructions"
    r"|disregard\s+the\s+above"
    r"|you\s+are\s+now"
    r"|system\s+prompt"
    r"|as\s+an\s+AI\s+language\s+model"
    r")")


@register("floor", version="ru-1.0")
class Floor(Detector):
    display = "Regex floor (triviality baseline, Russian)"
    binary = True
    notes = "quotable injection formulas, RU + EN — the triviality floor"

    def score(self, docs):
        for d in docs:
            yield 1.0 if RX.search(d.text) else 0.0
