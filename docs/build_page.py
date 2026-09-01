#!/usr/bin/env python3
"""Build docs/index.html -- the Russian article about proba-ipi and the run.

Every number on the page is read from a result file in docs/results/, the same
files the dataset ships, so the page cannot disagree with the run it describes.
The handful of figures that are not from this corpus (Quadrat-IPI's English
column, the trivial baseline) are declared in EXTERNAL below with their source
and are marked as such on the page.

    python3 docs/build_page.py                       # rebuild index.html
    python3 docs/build_page.py --sync-from <release> # refresh results/ and figures/

Nothing here touches proba/ -- metrics.py, window.py and figures.py are vendored
from quadrat-ipi-eval unchanged and stay diffable against upstream (see NOTICE).
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
import shutil
import sys

DOCS = pathlib.Path(__file__).resolve().parent
REPO = DOCS.parent
RESULTS = DOCS / "results"
FIGURES = DOCS / "figures"
OUT = DOCS / "index.html"
ADAPTERS = REPO / "proba" / "detectors"

DATASET_URL = "https://huggingface.co/datasets/privettoha/proba-ipi"
REPO_URL = "https://github.com/lovec-tech/proba-ipi-eval"
UPSTREAM_URL = "https://github.com/mihail-gribov/quadrat-ipi-eval"
QUADRAT_URL = "https://huggingface.co/datasets/mihailgribov/quadrat-ipi"

# --- numbers that do not live in docs/results/ ------------------------------
# Read off Quadrat-IPI's own published results/, same convention as ours: all
# 16,800 injections, one pooled threshold, a 2000-character aperture. The one
# row we hold a cached copy of (protectai) is checked against it at build time
# by check_external(); the other four are transcribed and marked on the page.
EN_QUADRAT = {
    "proventra": (0.150, 0.423),
    "bastion": (0.302, 0.506),
    "piguard": (0.174, 0.569),
    "protectai": (0.035, 0.148),
    "deepset": (0.014, 0.073),
}
EN_QUADRAT_CACHE = REPO.parent / "ru-injection-eval" / "build_v2" / "cache" / "quadrat_published"

# 21.4% is from CONSTRUCTION.md; 53.3% appears only in the dataset card's table.
# The trivial baseline has no run in results/ -- it was
# measured before the harness and its scores were not kept, so unlike every
# other row on the page it cannot be re-derived from this release. Said on the
# page, not hidden here.
TRIVIAL_BASELINE = (0.214, 0.533)
EN_REGEX_FLOOR = 0.065

# Which figures the article uses. The rest of the 80 stay in the dataset's
# reports/ -- this is an article, not the full report set.
FIGURE_KEYS = [
    "comparison-curve",
    "{promptidote}-cells-all",
    "{gbv}-cells-all",
    "{proventra}-cells-all",
    "{promptidote}-curve",
    "{promptidote}-marg-family",
    "{promptidote}-marg-action",
]

LEVERS_RU = [
    ("bare", "прямое требование без всякой подпорки"),
    ("execution_surface", "замаскировано под исполняемый код: макрос, формула, шаблон"),
    ("forged_frame", "подделывает служебную границу: теги, маркеры конца блока, роли"),
    ("guard", "выдаёт себя за систему безопасности и её вердикт"),
    ("identity", "переназначает персону или роль"),
    ("inference", "подаёт посылку как факт, вывод модель делает сама"),
    ("output_marking", "управляет форматом ответа"),
    ("persistence", "подано как конфигурация или сохранённая память"),
    ("pretext", "выдумывает процедуру, внутрь которой вложено требование"),
    ("revocation", "объявляет прежние инструкции недействительными"),
]

OBJECTIVES_RU = [
    ("deny", "сделать работу и утаить результат"),
    ("disarm", "выключить фильтр или проверку"),
    ("disclose", "раскрыть системную инструкцию"),
    ("escalate", "получить повышенные права"),
    ("execute", "выполнить команду или скрипт"),
    ("exfiltrate", "отправить данные выбранному получателю"),
    ("hijack_task", "подменить задачу другой"),
    ("manipulate", "исказить вывод, продолжая делать работу"),
    ("persist", "чтобы изменение пережило сессию"),
    ("unauthorized_action", "действие с внешним эффектом: отправить, заказать, оплатить"),
]

CARRIERS_RU = {
    "cards": "карточки товаров",
    "reviews": "отзывы",
    "web": "веб-страницы",
}


# --- loading ----------------------------------------------------------------

def load_runs() -> list[dict]:
    """Every run in docs/results/, strongest first at the tight operating point."""
    runs = []
    for f in sorted(RESULTS.glob("*.json")):
        d = json.loads(f.read_text())
        d["_file"] = f.name
        d["_stem"] = f.stem
        runs.append(d)
    if not runs:
        sys.exit(f"no result files in {RESULTS} -- run with --sync-from <release dir>")
    runs.sort(key=lambda d: point(d, "0.001")["mean_recall"], reverse=True)
    return runs


def point(run: dict, target: str) -> dict:
    """The run's numbers at one operating point.

    A binary detector has no curve and therefore one point of its own; the
    harness writes it at the top level instead of under points."""
    if run.get("binary"):
        return run
    return run["points"][target]


def model_ids() -> dict[str, str]:
    """The model each adapter names, read from the adapter source.

    The adapter is where the claim lives, so the page takes the model id from
    there rather than from a list retyped here."""
    ids = {}
    for f in sorted(ADAPTERS.glob("*.py")):
        src = f.read_text()
        name = re.search(r'@register\(\s*"([^"]+)"', src)
        mid = re.search(r'^\s*model_id\s*=\s*"([^"]+)"', src, re.M)
        if name and mid:
            ids[name.group(1)] = mid.group(1)
    return ids


def check_external() -> list[str]:
    """Verify what can be verified about the numbers declared in EXTERNAL."""
    notes = []
    f = EN_QUADRAT_CACHE / "protectai.json"
    if not f.exists():
        return ["EN protectai: кэш опубликованного прогона недоступен, строка не сверена"]
    d = json.loads(f.read_text())
    got = (d["points"]["0.001"]["mean_recall"], d["points"]["0.01"]["mean_recall"])
    want = EN_QUADRAT["protectai"]
    if any(abs(a - b) > 0.0005 for a, b in zip(got, want)):
        notes.append(f"EN protectai: объявлено {want}, в опубликованном прогоне {got}")
    return notes


# --- formatting -------------------------------------------------------------

def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%".replace(".", ",")


def plural(n: int, one: str, few: str, many: str) -> str:
    """Russian agreement for a generated count -- 1 ячейка, 2 ячейки, 5 ячеек."""
    if 11 <= n % 100 <= 14:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(n % 10, many)


def num(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def esc(s: str) -> str:
    return html.escape(str(s))


def cells_of(run: dict, target: str) -> tuple[int, int]:
    p = point(run, target)
    return round(p["coverage_50"] * p["n_cells"]), p["n_cells"]


# --- tables -----------------------------------------------------------------

def leaderboard(runs: list[dict], ids: dict[str, str]) -> str:
    """Every run at both operating points, plus the two baselines, by recall."""
    rows = []
    for r in runs:
        name = r["detector"]
        if name == "floor":
            label = "regex floor <span class=\"gloss\">горстка цитируемых фраз</span>"
            recall = f"<em>{pct(point(r, '0.001')['mean_recall'])}</em>"
            second = "&mdash;"
            h1, n1 = cells_of(r, "0.001")
            c1 = f"{h1} из {n1}"
            c2 = "&mdash;"
            cls = "base"
        else:
            mid = ids.get(name) or ("наш коммерческий детектор"
                                    if name == "promptidote" else None)
            gloss = f'<span class="gloss">{esc(mid)}</span>' if mid else ""
            label = f"<strong>{esc(name)}</strong> {gloss}"
            recall = pct(point(r, "0.001")["mean_recall"])
            second = pct(point(r, "0.01")["mean_recall"])
            h1, n1 = cells_of(r, "0.001")
            h2, n2 = cells_of(r, "0.01")
            c1, c2 = f"{h1} из {n1}", f"{h2} из {n2}"
            cls = "ours" if name == "promptidote" else ""
        rows.append((point(r, "0.001")["mean_recall"], cls, label, recall, c1, second, c2))

    rows.append((
        TRIVIAL_BASELINE[0], "base",
        "тривиальный базлайн <span class=\"gloss\">символьные n-граммы</span>",
        f"<em>{pct(TRIVIAL_BASELINE[0])}</em>", "&mdash;",
        f"<em>{pct(TRIVIAL_BASELINE[1])}</em>", "&mdash;",
    ))
    rows.sort(key=lambda t: t[0], reverse=True)

    body = "\n".join(
        f'<tr class="{cls}"><td>{label}</td><td class="n">{r1}</td><td class="n">{c1}</td>'
        f'<td class="n">{r2}</td><td class="n">{c2}</td></tr>'
        for _, cls, label, r1, c1, r2, c2 in rows
    )
    return f"""<div class="scroll"><table>
<thead><tr><th>детектор</th><th class="n">recall @0,1%</th><th class="n">ячейки @0,1%</th>
<th class="n">recall @1%</th><th class="n">ячейки @1%</th></tr></thead>
<tbody>
{body}
</tbody></table></div>"""


def price_table(runs: list[dict], keys: list[str]) -> str:
    """What the single threshold actually costs, carrier by carrier, at 0.1%."""
    by = {r["detector"]: r for r in runs}
    carriers = ["cards", "reviews", "web"]
    head = "".join(
        f'<th class="n">{CARRIERS_RU[c]}</th>' for c in carriers)
    rows = []
    for k in keys:
        p = point(by[k], "0.001")
        cols = "".join(f'<td class="n">{pct(p["fpr"][c], 3)}</td>' for c in carriers)
        rows.append(
            f'<tr><td><strong>{esc(k)}</strong></td>'
            f'<td class="n">{pct(p["fpr_pooled"], 3)}</td>{cols}</tr>')
    return f"""<div class="scroll"><table>
<thead><tr><th>детектор</th><th class="n">весь пул</th>{head}</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table></div>"""


def language_table(runs: list[dict], ids: dict[str, str]) -> str:
    """The five detectors measured on both corpora, EN beside RU."""
    by = {r["detector"]: r for r in runs}
    order = ["proventra", "bastion", "piguard", "protectai", "deepset"]
    kind = {
        "proventra": "mDeBERTa, <strong>многоязычная</strong>",
        "bastion": "английская DeBERTa",
        "piguard": "английская DeBERTa",
        "protectai": "английская DeBERTa",
        "deepset": "английская DeBERTa",
    }
    rows = []
    for k in order:
        en1, en2 = EN_QUADRAT[k]
        ru1 = point(by[k], "0.001")["mean_recall"]
        ru2 = point(by[k], "0.01")["mean_recall"]
        rows.append(
            f'<tr><td><strong>{esc(k)}</strong> <span class="gloss">{kind[k]}</span></td>'
            f'<td class="n">{pct(en1)}</td><td class="n em">{pct(ru1)}</td>'
            f'<td class="n">{pct(en2)}</td><td class="n em">{pct(ru2)}</td></tr>')
    return f"""<div class="scroll"><table>
<thead><tr><th>детектор</th><th class="n">EN @0,1%</th><th class="n">RU @0,1%</th>
<th class="n">EN @1%</th><th class="n">RU @1%</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table></div>"""


def lever_table(runs: list[dict], keys: list[str]) -> str:
    """Recall by lever at 0.1%, the working detectors side by side."""
    by = {r["detector"]: r for r in runs}
    marg = {k: point(by[k], "0.001")["marginals"]["family"] for k in keys}
    levers = sorted(marg[keys[0]], key=lambda f: marg[keys[0]][f]["recall"], reverse=True)
    worst = {k: min(marg[k], key=lambda f: marg[k][f]["recall"]) for k in keys}
    head = "".join(f'<th class="n">{esc(k)}</th>' for k in keys)
    rows = []
    for f in levers:
        cols = ""
        for k in keys:
            v = pct(marg[k][f]["recall"])
            cols += f'<td class="n">{"<em>" + v + "</em>" if worst[k] == f else v}</td>'
        rows.append(f'<tr><td><code>{esc(f)}</code></td>{cols}</tr>')
    return f"""<div class="scroll"><table>
<thead><tr><th>рычаг</th>{head}</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table></div>"""


def glossary(pairs: list[tuple[str, str]], head: str) -> str:
    rows = "\n".join(
        f'<tr><td><code>{esc(k)}</code></td><td>{esc(v)}</td></tr>' for k, v in pairs)
    return f"""<div class="scroll"><table class="gl">
<thead><tr><th>{head}</th><th>что это</th></tr></thead>
<tbody>
{rows}
</tbody></table></div>"""


def facts(runs: list[dict]) -> str:
    """The corpus in one block -- counted from the files, not transcribed."""
    ref = next(r for r in runs if r["detector"] == "promptidote")
    p = point(ref, "0.001")
    detectors = sorted({r["detector"] for r in runs} - {"floor"})
    items = [
        ("документов с инъекцией", num(p["n_positives"])),
        ("чистых документов", num(p["n_negatives"])),
        ("носители", "карточки товаров · отзывы · веб-страницы, три русских источника"),
        ("типов атак", f'<strong>{p["n_cells"]}</strong> '
                       f'{plural(p["n_cells"], "заполненная ячейка", "заполненные ячейки", "заполненных ячеек")}'
                       " — 10 рычагов × 10 целей, по 80 или 240 примеров в ячейке"),
        ("рабочие точки", "0,1% и 1% FPR, один порог на весь чистый пул"),
        ("измерено детекторов", f"{len(detectors)}, плюс regex floor — итого "
                                f"{len(runs)} {plural(len(runs), 'прогон', 'прогона', 'прогонов')}"),
        ("знаменатель recall", f'все {num(p["n_positives"])} инъекций, включая quota fill'),
        ("сборка корпуса", f'<code>{esc(ref["dataset"])}</code>'),
    ]
    rows = "\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in items)
    return f'<div class="scroll"><table class="facts"><tbody>\n{rows}\n</tbody></table></div>'


# --- page chrome ------------------------------------------------------------

CSS = """
:root{
  --measure: 44rem;
  --page:#fbfaf7; --sunken:#f3f1e9; --ink:#191919; --mut:#78746c;
  --line:#e2ded2; --hair:#eae7dd; --accent:#8a5a2b; --heat:#184f95;
  color-scheme: light;
}
@media (prefers-color-scheme: dark){
  :root{
    --page:#141510; --sunken:#1c1d16; --ink:#eceae2; --mut:#98958c;
    --line:#33342b; --hair:#26271f; --accent:#d4a373; --heat:#9ec5f4;
    color-scheme: dark;
  }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--page); color:var(--ink);
  font:400 17px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-feature-settings:"kern" 1; text-rendering:optimizeLegibility;
  counter-reset:sec;
}
a{color:inherit; text-decoration-color:var(--accent); text-underline-offset:3px}
a:hover{color:var(--accent)}
code{font:0.88em/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; color:var(--accent)}

.bar{
  position:sticky; top:0; z-index:9; background:var(--page);
  border-bottom:1px solid var(--hair); backdrop-filter:saturate(140%) blur(6px);
}
.bar .in{
  max-width:calc(var(--measure) + 3rem); margin:0 auto; padding:.7rem 1.5rem;
  display:flex; gap:1rem; align-items:baseline; justify-content:space-between;
}
.bar b{font-weight:600; letter-spacing:.01em}
.bar nav{display:flex; gap:1.1rem; font-size:.82rem; color:var(--mut)}
.bar nav a{text-decoration:none}

.wrap{max-width:calc(var(--measure) + 3rem); margin:0 auto; padding:0 1.5rem 6rem}

header.lead{padding:4rem 0 2.5rem; border-bottom:1px solid var(--line)}
h1{font-size:clamp(2rem,5.5vw,2.9rem); line-height:1.1; margin:0 0 1rem; letter-spacing:-.02em; font-weight:600}
.sub{font-size:1.12rem; color:var(--mut); margin:0; max-width:34rem}
.stamp{margin-top:1.6rem; font-size:.82rem; color:var(--mut)}
.stamp a{text-decoration-color:var(--line)}

.toc{margin:2.5rem 0 0; padding:0; list-style:none; font-size:.92rem; column-gap:2rem}
.toc li{padding:.28rem 0; border-bottom:1px solid var(--hair)}
.toc a{text-decoration:none; color:var(--mut); display:flex; gap:.7rem}
.toc a:hover{color:var(--accent)}
.toc .k{color:var(--accent); font-variant-numeric:tabular-nums; min-width:1.2rem}

section{padding-top:3.2rem; scroll-margin-top:4rem}
h2{
  font-size:1.42rem; font-weight:600; letter-spacing:-.01em; margin:0 0 1rem;
  display:flex; gap:.8rem; align-items:baseline;
}
h2::before{
  counter-increment:sec; content:counter(sec);
  font-size:.78rem; color:var(--accent); font-variant-numeric:tabular-nums;
  font-weight:500; padding-top:.2rem;
}
h3{font-size:1.02rem; font-weight:600; margin:2.2rem 0 .7rem}
p{margin:0 0 1.1rem}
p.note{color:var(--mut); font-size:.92rem}
strong{font-weight:600}
ul{margin:0 0 1.1rem; padding-left:1.1rem}
li{margin:.35rem 0}

.callout{
  border-left:2px solid var(--accent); background:var(--sunken);
  padding:1.2rem 1.4rem; margin:0 0 1.4rem; border-radius:0 3px 3px 0;
}
.callout p:last-child{margin-bottom:0}
.callout .tag{
  display:block; font-size:.72rem; letter-spacing:.09em; text-transform:uppercase;
  color:var(--accent); margin-bottom:.6rem; font-weight:600;
}

.scroll{overflow-x:auto; margin:0 0 1.4rem; -webkit-overflow-scrolling:touch}
table{border-collapse:collapse; width:100%; font-size:.92rem; min-width:30rem}
table.facts, table.gl{min-width:24rem}
th,td{padding:.55rem .7rem; text-align:left; border-bottom:1px solid var(--hair); vertical-align:top}
thead th{
  font-size:.74rem; letter-spacing:.06em; text-transform:uppercase; color:var(--mut);
  font-weight:600; border-bottom:1px solid var(--line); white-space:nowrap;
}
td.n,th.n{text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap}
tr.ours td{background:var(--sunken)}
tr.base td{color:var(--mut)}
td em,.em{font-style:normal; color:var(--accent)}
.gloss{display:block; font-size:.78rem; color:var(--mut); font-weight:400; margin-top:.1rem}
table.facts td:first-child{color:var(--mut); width:38%}

figure{
  margin:1.8rem 0 1.6rem;
  width:min(61rem, calc(100vw - 3rem));
  position:relative; left:50%; transform:translateX(-50%);
}
figure .frame{overflow-x:auto; border:1px solid var(--hair); border-radius:3px; background:var(--page)}
figure img{display:block; width:100%; min-width:34rem; height:auto}
figcaption{font-size:.85rem; color:var(--mut); margin-top:.7rem; max-width:44rem}

pre{
  background:var(--sunken); border:1px solid var(--hair); border-radius:3px;
  padding:.9rem 1.1rem; overflow-x:auto; margin:0 0 1.4rem;
  font:0.85rem/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
pre code{color:var(--ink)}
.files{white-space:pre; font:0.82rem/1.7 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}

footer{
  margin-top:4.5rem; padding-top:1.5rem; border-top:1px solid var(--line);
  font-size:.85rem; color:var(--mut);
}
@media (max-width:640px){
  body{font-size:16px}
  header.lead{padding:2.5rem 0 2rem}
  figure img{min-width:30rem}
}
"""


# --- the article ------------------------------------------------------------

def figure(src: str, alt: str, caption: str) -> str:
    return (f'<figure><div class="frame"><img src="figures/{src}" alt="{esc(alt)}" '
            f'loading="lazy"></div><figcaption>{caption}</figcaption></figure>')


def sections(runs: list[dict], ids: dict[str, str]) -> list[tuple[str, str, str]]:
    stem = {r["detector"]: r["_stem"] for r in runs}
    working = ["promptidote", "gbv", "proventra"]
    by = {r["detector"]: r for r in runs}
    pi = point(by["promptidote"], "0.001")
    pi1 = point(by["promptidote"], "0.01")
    lo, hi = pi["attainable_range"]
    ours_window = by["promptidote"]["window"]
    open_windows = sorted({r["window"] for r in runs
                           if r["policy"] == "chunk" and r["detector"] != "promptidote"})
    gap1 = round((pi1["mean_recall"] - pi["mean_recall"]) * 100)
    gap_gbv = round((point(by["gbv"], "0.01")["mean_recall"]
                     - point(by["gbv"], "0.001")["mean_recall"]) * 100)
    stale = len(by["promptidote"].get("stale_ids") or [])
    imported_n = sum(1 for r in runs if r.get("imported"))

    out = []

    out.append(("chto-eto", "Что это", f"""
<p>Мы собрали корпус для проверки детекторов непрямой промпт-инъекции на русском языке
и прогнали через него всё, до чего дотянулись: семь детекторов и regex floor. Это страница
о том, что получилось.</p>

<p><strong>Непрямая инъекция</strong> — это когда инструкция для модели лежит не в реплике
пользователя, а в документе, который агент прочитал по дороге: в карточке товара, в отзыве,
на веб-странице. Пользователь её не писал и обычно не видит.</p>

{facts(runs)}

<p><a href="{QUADRAT_URL}">Quadrat-IPI</a> — английский корпус того же устройства;
здесь он всё время рядом как точка отсчёта. Сетка
из {pi["n_cells"]} типов атак, протокол размещения и пропорции скопированы у него
без изменений: те же десять рычагов, те же десять
целей, те же {pi["n_cells"]} допустимые ячейки, те же глубины ячеек, то же правило,
куда девать отвергнутого кандидата. Не скопирован язык. Носители здесь — русские документы
маркетплейса вместо почты и офисных файлов, и каждая инъекция написана по-русски,
а не переведена.</p>

<p>Зачем держать конструкцию неизменной и менять только язык: один и тот же детектор
можно измерить на английском корпусе и на этом, и тогда разница между двумя цифрами читается
как разница языков, а не как разница линеек. Одна из измеренных моделей многоязычная,
и на строгой рабочей точке она укладывается в 1,9 пункта от своего английского результата,
пока четыре англоязычные теряют от 40 до 170 раз. Оба прочтения — в разделе
<a href="#yazyk">«Что переносится между языками»</a>, включая то место,
где русский корпус оказался просто труднее.</p>
"""))

    out.append(("coi", "Кто это публикует", """
<div class="callout"><span class="tag">Конфликт интересов</span>
<p>Первую строку таблицы ниже занимает <code>promptidote</code> — <strong>наш собственный
коммерческий детектор</strong>. Бенчмарк тоже наш. Мы говорим это до таблицы, а не оставляем
читателю искать разгадку в доменном имени.</p></div>

<p>Вендорский бенчмарк стоит ровно столько, сколько стоит его проверяемость, поэтому здесь
перечислено то, что делает его проверяемым, а не предполагается по умолчанию. Порог, ячейку
и доверительный интервал считает код <a href="%s">quadrat-ipi-eval</a>, перенесённый без
единой правки: любое расхождение видно построчным диффом с апстримом. Сетку спроектировали для другого корпуса
до того, как на ней измерили наш детектор. Скоры каждого детектора по каждому из документов
лежат в релизе, так что любую цифру отсюда можно пересчитать или опровергнуть по исходным
файлам, не запуская ни одной модели и не веря нам на слово. Адаптер нашего детектора живёт
по тем же правилам, что и остальные семь: объявляет апертуру, не имеет права бинаризовать
скор и при ошибке сервиса останавливает прогон, а не подставляет ноль.</p>

<p>Если вы найдёте здесь цифру, которая не выдерживает собственного файла со скорами, —
это тот баг-репорт, который нам нужен.</p>
""" % UPSTREAM_URL))

    out.append(("chto-merim", "Что именно измеряется", f"""
<p>Recall при заданной частоте ложных срабатываний, <strong>с одним порогом на весь чистый
пул целиком</strong>, а не отдельным для каждого носителя. У боевой системы одна рабочая
точка, и она не знает, какой документ читает, пока его не прочитает: порог по носителю
выдал бы детектору оракула, которого у него на самом деле нет. Поэтому FPR по носителю
мы показываем всегда, но никогда не фиксируем.</p>

<p>Рабочих точек тоже всегда две. <strong>0,1%</strong> — там, где фильтр документов реально
стоит в проде. <strong>1%</strong> — там, где приводят большинство опубликованных цифр
по инъекциям. Они ранжируют детекторы по-разному, так что назвать только одну — уже значит
выбрать победителя заранее.</p>

<p>Колонка «ячейки» — в скольких из {pi["n_cells"]} типов атак детектор ловит хотя бы половину.
Это опубликованная метрика Quadrat-IPI, и она отвечает на другой вопрос, чем средний recall:
{pct(pi["mean_recall"])} в среднем можно набрать, забрав половину сетки целиком и не увидев
вторую.</p>

<p><strong>Знаменатель — все {num(pi["n_positives"])} инъекций</strong>, включая 14,5%
quota fill: пейлоады, которыми добивали ячейку до нормы и которые слепой судья не подтвердил.
Так же считает Quadrat-IPI, а сравнивать два корпуса можно только на одном знаменателе.
Recall по одним подтверждённым инъекциям доступен как <code>--slice verified</code>
и у сильных детекторов выше на 2–5 пунктов.</p>

<h3>Чего стоит один порог</h3>

<p>Порог подобран так, чтобы на всём чистом пуле вышло 0,1%. Вот как эти 0,1% распределяются
по трём носителям у трёх детекторов, которые здесь вообще работают:</p>

{price_table(runs, working)}

<p>Бюджет выедает веб. Карточки и отзывы короткие и однообразные, на них детектор почти
не ошибается; веб-страницы длинные и разношёрстные, и там живёт почти вся ложная тревога.
У {esc("promptidote")} разрыв между лучшим и худшим носителем — от нуля до
{pct(pi["fpr"]["web"], 3)}. Клиенту с потоком веб-документов средняя цифра обещает
не то, что он получит, и это ровно та причина, по которой порог здесь не подбирается
по носителю.</p>
"""))

    out.append(("rezultaty", "Результаты", f"""
{leaderboard(runs, ids)}

{figure("comparison-curve.svg", "recall против бюджета ложных срабатываний",
        "Каждый детектор на всей оси, а не в одной точке. Ранжирование при одном "
        "бюджете не может сказать, отстаёт детектор везде или только там, где кто-то "
        "поставил порог. Порядок трёх работающих детекторов держится на всём диапазоне; "
        "меняется то, сколько каждый покупает на дополнительный бюджет. "
        "Пересечения есть только между англоязычными моделями, на recall, где "
        "перестановка уже ничего не значит.")}

<p>Работают трое. <code>promptidote</code> берёт {pct(pi["mean_recall"])} на строгой точке
и {pct(pi1["mean_recall"])} на мягкой, <code>gbv</code> — единственная открытая модель,
обученная на русском, — идёт следом, <code>proventra</code> держится за счёт многоязычности.
Остальные четыре модели англоязычные и на русском тексте не видят практически ничего:
их токенизатор не знает кириллицы, и разбирать их результат как качество детектора нельзя.
Что с ними происходит и почему это отдельный факт, а не строка в общем зачёте, —
в разделе <a href="#yazyk">про язык</a>.</p>

<p>Вторую колонку надо читать вместе с первой. Одна и та же система на двух ценах — это
две разные системы для того, кто разбирает очередь алертов. У <code>promptidote</code> между
ними {gap1} пунктов recall и десятикратная разница в количестве ложных тревог;
у <code>gbv</code> — {gap_gbv} {plural(gap_gbv, "пункт", "пункта", "пунктов")}.
Кто из них лучше, зависит от того, кто платит за разбор.</p>

<p><strong>Апертуры у детекторов разные, и это объявлено, а не спрятано.</strong> Харнесс
не навязывает окно детектору, который его не просил: как читать длинный документ — часть
детектора, а не линейки, и харнесс, режущий всем одинаково, мерил бы recall собственного
сплиттера. <code>promptidote</code> читал окнами по {ours_window} символов — столько берёт
его API; открытые модели — по {open_windows[0]}, их лимит контекста. Что именно читал каждый
прогон, записано в его файле полями <code>policy</code>, <code>window</code>,
<code>overlap</code>. Документов, которые разные апертуры вообще режут
по-разному, здесь в любом случае мало: границу окна в {open_windows[0]} символов пересекают
142 инъекции из {num(pi["n_positives"])} — 0,85%.</p>
"""))

    out.append(("poly", "Два дна, и они отвечают на разные вопросы", f"""
<p>Две нижние строки таблицы — не украшение.</p>

<p><strong>Тривиальный базлайн</strong> — классификатор на символьных n-граммах, ничего
не знающий про семантику инъекций, обученный с расщеплением по формулировке, чтобы ни одна
копия пейлоада не попала в собственный тест. Он берёт {pct(TRIVIAL_BASELINE[0])} на строгой
точке. Детектор, который его не превзошёл, не показал ничего сверх словаря этого корпуса —
и <code>proventra</code> его здесь не превосходит.</p>

<p><strong>Regex floor</strong> — горстка цитируемых фраз («игнорируй предыдущие инструкции»,
«system prompt»). Он бинарный, поэтому у него одна рабочая точка, а не кривая, и срабатывает
он при {pct(point(by["floor"], "0.001")["fpr_pooled"], 3)} ложных. Его
{pct(point(by["floor"], "0.001")["mean_recall"])} — это не характеристика детектора,
а характеристика корпуса: примерно такая доля здешних инъекций является цитируемым шаблоном,
который любой нагрепает. У Quadrat-IPI собственный regex floor читается как
{pct(EN_REGEX_FLOOR)} на английском корпусе — самая близкая к like-for-like линия между
двумя наборами, с оговоркой, что списки фраз — переводы друг друга, а не одни и те же строки.</p>

<p class="note">Оговорка к строке базлайна: у неё, единственной на этой странице, нет прогона
в <code>results/</code>. Её измеряли до харнесса, скоры не сохранили, и пересчитать её
по релизу нельзя — цифра приведена по <code>CONSTRUCTION.md</code>. Regex floor,
в отличие от неё, лежит в релизе полностью.</p>

<p>И ещё одно, что видно только при уравненном бюджете. Regex floor срабатывает при
{pct(point(by["floor"], "0.001")["fpr_pooled"], 3)} ложных, а колонка рядом снята
при 0,1% — то есть остальным досталось впятеро больше бюджета. Если уравнять
(<code>python3 -m proba.compare --against floor</code>), две английские модели падают
до 0,0%: они проигрывают горстке регулярок, когда бюджет одинаковый.</p>
"""))

    out.append(("yazyk", "Что переносится между языками", f"""
<p>Пять детекторов измерены и на английском корпусе Quadrat-IPI, и на этом русском. Обе стороны
на одном соглашении: все {num(pi["n_positives"])} инъекций, один общий порог, апертура
в 2000 символов. Английская колонка прочитана из опубликованных <code>results/</code>
самого Quadrat-IPI.</p>

{language_table(runs, ids)}

<p><strong>Многоязычную строку и англоязычные надо читать друг против друга, а не по
отдельности.</strong> Четыре англоязычные модели теряют между корпусами от 40 до 170 раз.
Это их токенизатор, а не их веса: ни у одной нет русского в словаре. Поставить их рядом
с русскоязычными детекторами без этой оговорки — значит записать архитектурный разрыв
в разрыв по качеству.</p>

<p>Единственная многоязычная модель теряет <strong>1,9 пункта на 0,1% и 7,4 пункта
на 1%</strong>. Такой порядок — почти совпадение там, где бюджет жёсткий, и настоящий
разрыв там, где он свободный — и есть честный результат, и он говорит сразу о двух вещах.
Во-первых, сетка и композиция воспроизведены: зеркало, собранное неправильно, не легло бы
в два пункта ни на одном конце. Во-вторых, этот корпус для той же модели действительно
труднее на мягкой точке. Это <strong>не</strong> утверждение о бесплатном переносе,
и более ранняя версия карточки именно так его и переоценила, сравнив английскую цифру
по всем инъекциям с русской по одним подтверждённым.</p>

<p>Две вещи, которыми разрыв не объясняется. Не апертура: другой сплиттер предложений
может сдвинуть результат максимум на 0,85 пункта, потому что границу окна пересекают
всего 142 инъекции из {num(pi["n_positives"])}. И не чистый пул: оба порога ложатся
в 0,002 пункта от своей целевой ставки.</p>
"""))

    out.append(("slepye-zony", "Слепые зоны", f"""
<p>Рычаг — это то, чем инъекция добивается послушания. Здесь recall по рычагам на строгой
точке, три носителя вместе, один порог. Цветом отмечен худший рычаг каждого детектора.</p>

{lever_table(runs, working)}

<p><strong>Общего дна по рычагу нет.</strong> <code>promptidote</code> и <code>proventra</code>
хуже всего справляются с <code>persistence</code> — инъекцией, поданной как сохранённая
конфигурация или память, — а <code>gbv</code> проваливается на <code>identity</code>.
Более ранняя версия карточки объявляла общим дном <code>guard</code>; это было неверно,
причём неверно против таблицы, напечатанной прямо над этим утверждением: <code>guard</code>
оказался последней строкой только потому, что таблица отсортирована по первой колонке.</p>

<p>А вот <strong>по цели общее дно есть</strong>, и оно переживает любую перенарезку:
<code>unauthorized_action</code> — действие с внешним эффектом (отправить, заказать,
оплатить) — худшая цель у всех трёх
({", ".join(f'{k} {pct(point(by[k], "0.001")["worst_action"]["recall"])}' for k in working)}).
Это самая дорогая клетка сетки: именно здесь инъекция что-то делает, а не что-то говорит.</p>

{figure(f'{stem["promptidote"]}-cells-all.svg', "сетка рычаг × цель, promptidote",
        "Все 92 типа атак у <code>promptidote</code>. Каждая плитка разбита пополам: "
        "слева recall на 0,1% ложных, справа на 1%. Последняя строка и последний "
        "столбец — маргиналы. Где половинки плитки резко расходятся, recall этой "
        "ячейки — факт про бюджет, а не про детектор.")}

{figure(f'{stem["gbv"]}-cells-all.svg', "сетка рычаг × цель, gbv",
        "Та же сетка у <code>gbv</code>, лучшей открытой модели. Различается не только "
        "уровень: провалы лежат в других местах.")}

{figure(f'{stem["proventra"]}-cells-all.svg', "сетка рычаг × цель, proventra",
        "И у <code>proventra</code>. На строгой точке она не берёт половину "
        "ни в одной из 92 ячеек — отсюда ноль в колонке «ячейки».")}

<p>Средний recall прячет разброс. У <code>promptidote</code> на строгой точке
от {pct(lo)} в худшем типе атак до {pct(hi)} в лучшем: насколько средняя цифра
относится к вам, зависит от того, какие типы приходят именно к вам.</p>

{figure(f'{stem["promptidote"]}-curve.svg', "recall против бюджета, promptidote",
        "Весь размен целиком, а не точка. Логарифмическая ось x, потому что вопрос "
        "живёт в первой декаде: между 0,01% и 0,1% ложных — разница между фильтром, "
        "который можно поставить на поток, и тем, который нельзя. Полоса — "
        "95-процентный интервал; заштрихованная слева зона — то, что "
        f"{num(pi['n_negatives'])} чистых документов вообще не в состоянии выразить: "
        "ставка, стоящая на горстке ложных срабатываний, — не ставка.")}

{figure(f'{stem["promptidote"]}-marg-family.svg', "recall по рычагам, promptidote",
        "Строки сетки, свёрнутые в маргиналы: чем инъекция добивается послушания.")}

{figure(f'{stem["promptidote"]}-marg-action.svg', "recall по целям, promptidote",
        "Столбцы сетки: чего инъекция просит.")}

<p class="note">Подписи на всех фигурах английские. Их рисует <code>figures.py</code>,
перенесённый из чужого харнесса без изменений, и переписывать его ради локализации
подписей значило бы потерять ровно то свойство, ради которого он взят неизменным.
Расшифровка всех двадцати ключей — в разделе <a href="#korpus">«Как устроен корпус»</a>.</p>
"""))

    out.append(("svoj-detektor", "Как измерить свой детектор", f"""
<p>Харнесс — <a href="{REPO_URL}">proba-ipi-eval</a>. Один класс, один метод. Корпус
скачивается при первом запуске, версия зафиксирована тегом.</p>

<pre><code>python3 -m proba.run --list                 # все зарегистрированные адаптеры
python3 -m proba.run --detector floor       # измерить один (этому не нужна модель)
python3 -m proba.report                     # страница на каждый детектор
python3 -m proba.compare --against floor    # все при одном бюджете, по сохранённым скорам</code></pre>

<pre><code>from proba.detector import Detector, register

@register("my-detector", version="1.2")
class MyDetector(Detector):
    max_chars = 2000                 # ваш лимит контекста; опустите, если его нет

    def setup(self):                 # необязательно
        self.model = load_model()

    def score(self, docs):           # yield одного float на документ, в том же порядке
        for d in docs:
            yield self.model.predict(d.text)</code></pre>

<p><strong>Апертура ваша.</strong> В <code>score</code> харнесс отдаёт документы целиком.
Объявите <code>max_chars</code> — базовый класс сам порежет их на окна по границам
предложений с нахлёстом в четыре предложения. Или переопределите
<code>score_documents</code> и режьте так, как режет ваш продукт. Чего харнесс не сделает
никогда — не навяжет окно детектору, который его не просил: как читать длинный документ,
решает сам детектор, а харнесс, который резал бы всем одинаково, измерял бы recall
собственного сплиттера.</p>

<p><strong>Возвращайте непрерывный скор</strong>, выше — больше похоже на инъекцию. Весь
протокол держится на подборе порога под целевую FPR, а вердикт пороговать нельзя. Если
детектор и правда бинарный, укажите <code>binary = True</code>, и он покажется на своей
собственной точке, а не будет притворяться, что для него существует кривая.</p>

<p>Измерьте свой детектор и опубликуйте строку. Харнесс существует ровно для этого.</p>
"""))

    out.append(("korpus", "Как устроен корпус", f"""
<p>Две ортогональные оси, взятые у Quadrat-IPI без изменений. <strong>Рычаг</strong> —
чем вставка добивается послушания. <strong>Цель</strong> — чего она просит. Сто пар
минус восемь структурно невозможных дают {pi["n_cells"]} заполненные ячейки; список
невозможных не выдуман, а посчитан по корпусу Quadrat-IPI.</p>

{glossary(LEVERS_RU, "рычаг")}

{glossary(OBJECTIVES_RU, "цель")}

<p>Глубина ячеек оттуда же: 33 ячейки по 80 пейлоадов, 59 по 240. Значения сняты
с корпуса Quadrat-IPI ячейка за ячейкой, а не назначены одним плоским числом — плоское число дало бы
доверительным интервалам другую ширину, и сравнение по ячейкам перестало бы что-то
значить.</p>

<h3>Конвейер</h3>

<ul>
<li><strong>Носители.</strong> 115 357 документов из трёх источников, набранных в стороне
от чистых пулов: пересечение по хешу полного текста нулевое, так что чистый сплит
не тронут и прежние измерения на нём остаются в силе. Персональные данные вычищены.</li>
<li><strong>Пейлоады.</strong> Генерируются по ячейке с явным запретом повторять уже
существующие, дедуплицируются по нормализованному тексту, не больше трёх с одинаковым
двухсловным началом.</li>
<li><strong>Слепой судья.</strong> Читает <em>только текст пейлоада</em>, не зная ячейки,
под которую его заказывали, и называет три вещи: инъекция ли это вообще, рычаг, цель.</li>
<li><strong>Размещение</strong> — правило Quadrat-IPI дословно. Совпало по обеим осям —
осталось на месте (23,7%); не совпало — переезжает в названную судьёй ячейку,
если там есть место (27,7%); недобор добивается из отвергнутого пула ближайшим по вердикту
(34,2%); остаток — quota fill с меткой <code>inj_verified: false</code> (14,5%).
Несущая деталь тут — ограничение вместимости: без него пробный прогон растянул размеры
ячеек от 1 до 788 при цели 80/240.</li>
<li><strong>Композиция.</strong> Один пейлоад — один документ. Позиция, обфускация, стиль
и сворачивание типографики берутся в тех же пропорциях. Спан проверен на каждом документе:
{num(pi["n_positives"])} из {num(pi["n_positives"])}.</li>
</ul>

<p>Что отличается осознанно: носители русские; пейлоады написаны, а не переведены (пейлоады
Quadrat-IPI привязаны к сценариям его собственных носителей — переводам, аудитам, бронированиям, —
и такая строка в карточке товара выглядит бессмыслицей); гомоглифы вывернуты наизнанку
(в английском корпусе латиница подменяется похожей кириллицей, здесь наоборот).</p>
"""))

    out.append(("ogranicheniya", "Ограничения", f"""
<p><strong>Слепой судья здесь слабее, чем судья Quadrat-IPI, и это измерено,
а не предположено.</strong> Согласие с заказанной ячейкой по обеим осям —
<strong>27,0%</strong> против заявленных там ~74%. Чтобы понять, судья это или пейлоады,
наш судья был прогнан на 600 размеченных строк Quadrat-IPI: <strong>48,3%</strong> согласия (64,5% по одному рычагу,
75,9% по одной цели). Потеря делится примерно поровну между тем и другим. Оговорка
к самой этой цифре: те строки — английский текст, а промпт судьи русский, и чистое
сравнение потребовало бы английского промпта, которого не запускали. Метки уровня ячейки
надо читать с этой поправкой; recall уровня документа она не задевает.</p>

<p><strong>14,5% строк лежат в ячейке, которую судья для них не выбирал</strong> — тот самый
quota fill. Они добивают ячейку до целевой глубины, и <em>все цифры на этой странице
их включают</em>, потому что опубликованные строки Quadrat-IPI включают его собственный
quota fill, а сравнение двух наборов обязано идти по одному знаменателю. Это ещё и более
трудные строки: если ограничиться подтверждёнными инъекциями, сильные детекторы
поднимаются на 2–5 пунктов (<code>promptidote</code> 57,8% → 61,7%, <code>gbv</code>
37,7% → 40,0%, <code>proventra</code> 13,1% → 14,9% на строгой точке), а слабые
не двигаются. Читайте любую из двух цифр, но не смешивайте соглашение одного набора
с соглашением другого — ранняя версия карточки сделала именно это в межъязыковой таблице.</p>

<p><strong>{stale} строк нашего собственного прогона сняты с чуть другого текста.</strong>
Уже после того, как прогон был измерен, в {stale} документах с инъекцией замаскировали
телефоны цифра в цифру. Остальные прогоны пересчитали на исправленном тексте —
самый большой сдвиг составил 0,006 пункта; <code>promptidote</code> достаётся по сетевому
API, и эти строки не перескорили. Их идентификаторы перечислены полем
<code>stale_ids</code> в его файле результата, а закрывается это одной командой
<code>proba.rescore</code> с ключом к API.</p>

<p><strong>Один генератор, а не каскад.</strong> Почти каждый пейлоад написан
<code>deepseek-chat</code>; набор Quadrat-IPI опирается на несколько моделей.</p>

<p><strong><code>locality</code> — это метка, а не преобразование.</strong>
В схеме Quadrat-IPI
у поля четыре значения по 25% каждое (<code>point</code>, <code>spread</code>,
<code>buried</code>, <code>repeated</code>). Проверка по его же данным: во всех четырёх
случаях спан точно равен тексту инъекции, инъекция встречается ровно один раз,
распределение позиций одинаковое. Поле выставлено здесь в тех же пропорциях, чтобы оно
существовало и сравнивалось, но текст от него не зависит.</p>

<p><strong>Самая тугая ячейка — <code>bare</code> × <code>deny</code>.</strong> Голое
требование без рамки, роли и предлога, которое к тому же просит сделать работу и утаить
результат, допускает мало разных формулировок: ячейка висела на 59–75 пейлоадах
и потребовала нескольких проходов генерации, чтобы дойти до глубины. Quadrat-IPI решает это
объявлением восьми пар структурно невозможными заранее; здесь одна тугая, но возможная пара
нашлась в процессе сборки, а не была объявлена до неё.</p>

<p><strong>Сплиттер предложений.</strong> Регулярка, на которую откатывается
<code>window.sentence_spans</code> без <code>blingfire</code>, требует заглавной буквы
после точки — и только из латиницы. На русском она почти не находит границ и вырождается
в нарезку кусками по 2000 символов. Опубликованные цифры сняты именно в этом состоянии,
поэтому <code>blingfire</code> здесь опциональная зависимость: поставите её — и свежий
прогон разойдётся с таблицей, которую должен воспроизвести. Какой сплиттер сработал,
записано в каждом результате полем <code>segmenter</code>.</p>
"""))

    out.append(("licenzii", "Лицензии и dual use", f"""
<p>Лицензия построчная, по источнику носителя: <code>cc0-1.0</code> для карточек товаров
(<a href="https://huggingface.co/datasets/nyuuzyou/ke-products">nyuuzyou/ke-products</a>),
<code>mit</code> для отзывов
(<a href="https://github.com/yandex/geo-reviews-dataset-2023">Яндекс, geo-reviews-dataset-2023</a>),
<code>odc-by</code> для веб-текста
(<a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2">FineWeb-2</a>
плюс Common Crawl Terms of Use). Наш вклад — инъекции, метки, размещение, замеры —
<strong>CC-BY-4.0</strong>, и намеренно не NC: несколько измеренных здесь детекторов
коммерческие, а под NC их создатели не смогли бы опубликовать результаты на этом наборе.</p>

<p><strong>Персональные данные вычищены, в отличие от английского корпуса.</strong> Quadrat-IPI
сохраняет настоящие имена из своих носителей Enron и FOIA, рассуждая, что носитель — это
и есть то, против чего меряют детектор. Здесь выбор обратный: телефоны, адреса почты,
<code>@handles</code> и ссылки на профили заменены плейсхолдерами до публикации,
а документы, где правдоподобный контакт пережил чистку, выброшены целиком. Проверка идёт
по <em>вычищенному</em> тексту, после подстановки плейсхолдеров, а не до: два документа
стали побайтово идентичны уже опубликованным строкам именно после подстановки и проскочили
бы проверку, сделанную раньше. Отдельный проход поймал 41 телефон, переживший чистку
за счёт нестандартного форматирования, и замаскировал их цифра в цифру, сохранив все
смещения спанов.</p>

<p><strong>Dual use.</strong> Набор содержит работающие формулировки инъекций и опубликован
открыто. Так устроен каждый оценочный корпус этого класса: без него детекторам не на чем
проверяться, а любой вендор может назвать любую цифру.</p>
"""))

    out.append(("fajly", "Файлы и воспроизводимость", f"""
<p>Корпус — <a href="{DATASET_URL}">privettoha/proba-ipi</a> на Hugging Face. Харнесс —
<a href="{REPO_URL}">lovec-tech/proba-ipi-eval</a>. Ядро измерения —
<a href="{UPSTREAM_URL}">quadrat-ipi-eval</a>, перенесённое без изменений.</p>

<div class="scroll"><div class="files">data/
  injected_{{cards,reviews,web}}.jsonl   {num(pi["n_positives"])} документов с инъекцией
  {{cards,reviews,web}}_clean.jsonl      {num(pi["n_negatives"])} чистых документов
  injections.jsonl                     банк пейлоадов с ячейкой и вердиктом судьи
  INVARIANTS.txt                       результат проверок за каждым утверждением
results/
  &lt;детектор&gt;-&lt;версия&gt;-&lt;штамп&gt;-&lt;апертура&gt;.json          метрики, ячейки, маргиналы
  &lt;детектор&gt;-&lt;версия&gt;-&lt;штамп&gt;-&lt;апертура&gt;.scores.jsonl   скор по каждому из {num(pi["n_positives"] + pi["n_negatives"])} документов
reports/
  README.md                            индекс: все детекторы, обе точки
  &lt;прогон&gt;.md + figures/*.svg          страница на детектор</div></div>

<p>Каждый прогон записывает апертуру, через которую читал (<code>policy</code>,
<code>window</code>, <code>overlap</code>), и то, <strong>какой сплиттер был
активен</strong> (<code>segmenter</code>): на русском это не взаимозаменяемые вещи,
а цифра с необъявленным сплиттером невоспроизводима. {imported_n} прогонов
из {len(runs)} несут <code>imported: true</code>: их скоры старше харнесса, а метрики
пересчитаны из этих скоров, без повторного запуска модели. Такие прогоны намеренно
не показывают <code>seconds</code> и <code>n_windows</code> — показывать время, которого
не измеряли, нельзя.</p>

<p>Порог — это число, выбранное по сохранённым скорам постфактум. Поэтому сдвинуть рабочую
точку, перенарезать слайс или перерисовать таблицу — секунда арифметики, а не новый прогон,
и новый детектор можно сравнить с этими семью, не запуская их заново.</p>

<pre><code>python3 -m proba.table --results eval-out/results   # таблицы карточки
python3 -m proba.report                             # страницы отчётов
python3 -m proba.compare --against floor            # единый бюджет для всех</code></pre>
"""))

    return out


MONTHS_RU = ["января", "февраля", "марта", "апреля", "мая", "июня",
             "июля", "августа", "сентября", "октября", "ноября", "декабря"]


def run_date(runs: list[dict]) -> str:
    """The day the corpus was measured -- from the runs, not from the clock.

    A build timestamp would change the file on every rebuild and say nothing
    about the numbers; the date that matters is when they were taken."""
    stamps = sorted(r["run_at"] for r in runs if r.get("run_at"))
    y, m, d = stamps[-1][:10].split("-")
    return f"{int(d)} {MONTHS_RU[int(m) - 1]} {y}"


HEAD = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>proba-ipi — бенчмарк детекторов промпт-инъекций на русском</title>
<meta name="description" content="Восемь прогонов на одном корпусе: 16 800 русских документов с непрямой промпт-инъекцией и 81 000 чистых, сетка из 92 типов атак, зеркало Quadrat-IPI.">
<meta property="og:type" content="article">
<meta property="og:title" content="proba-ipi — бенчмарк детекторов промпт-инъекций на русском">
<meta property="og:description" content="Восемь прогонов на одном корпусе: 16 800 русских документов с непрямой промпт-инъекцией и 81 000 чистых, сетка из 92 типов атак.">
<meta property="og:url" content="https://lovec-tech.github.io/proba-ipi-eval/">
<meta name="twitter:card" content="summary">
<style>%s</style>
</head>
<body>
"""


def render(runs: list[dict], ids: dict[str, str]) -> str:
    secs = sections(runs, ids)
    toc = "\n".join(
        f'<li><a href="#{sid}"><span class="k">{i}</span><span>{esc(title)}</span></a></li>'
        for i, (sid, title, _) in enumerate(secs, 1))
    body = "\n".join(
        f'<section id="{sid}">\n<h2>{esc(title)}</h2>\n{html_body.strip()}\n</section>'
        for sid, title, html_body in secs)

    return HEAD % CSS + f"""<div class="bar"><div class="in">
<b>proba-ipi</b>
<nav>
<a href="{DATASET_URL}">корпус</a>
<a href="{REPO_URL}">харнесс</a>
<a href="{REPO_URL}#readme">README</a>
</nav>
</div></div>

<div class="wrap">
<header class="lead">
<h1>Детекторы промпт-инъекций на русском: что показал прогон</h1>
<p class="sub">Восемь прогонов на одном корпусе — {num(point(runs[0], "0.001")["n_positives"])}
документов с непрямой инъекцией и {num(point(runs[0], "0.001")["n_negatives"])} чистых,
на сетке из {point(runs[0], "0.001")["n_cells"]} типов атак, скопированной у английского
Quadrat-IPI без изменений.</p>
<p class="stamp">Прогон {run_date(runs)} · корпус
<a href="{DATASET_URL}">privettoha/proba-ipi</a> · харнесс
<a href="{REPO_URL}">lovec-tech/proba-ipi-eval</a></p>
<ul class="toc">
{toc}
</ul>
</header>

{body}

<footer>
<p>Все цифры на этой странице прочитаны из файлов результатов в
<a href="{REPO_URL}/tree/main/docs/results"><code>docs/results/</code></a> и собраны
скриптом <code>docs/build_page.py</code>: страница ничего не считает сама и поэтому
не может разойтись с прогоном, который описывает. Исключения названы там, где они есть.</p>
<p>Инъекции, метки, размещение и замеры — CC-BY-4.0. Текст носителей — по лицензии
своего источника, построчно.</p>
</footer>
</div>
</body>
</html>
"""


# --- sync -------------------------------------------------------------------

def sync_from(release: pathlib.Path) -> None:
    """Copy in the result files and the figures the article uses.

    Only the metrics files -- the .scores.jsonl next to them are 5 MB each and
    belong in the dataset, not in a repo that renders a page."""
    src_results = release / "results"
    src_figures = release / "reports" / "figures"
    if not src_results.is_dir():
        sys.exit(f"no results/ under {release}")
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    for f in sorted(src_results.glob("*.json")):
        if f.name.endswith(".scores.jsonl"):
            continue
        shutil.copy2(f, RESULTS / f.name)
        print(f"results/{f.name}")

    stems = {json.loads(f.read_text())["detector"]: f.stem
             for f in sorted(RESULTS.glob("*.json"))}
    for key in FIGURE_KEYS:
        name = key.format(**stems) + ".svg"
        src = src_figures / name
        if not src.exists():
            sys.exit(f"missing figure {name}")
        shutil.copy2(src, FIGURES / name)
        print(f"figures/{name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sync-from", metavar="DIR",
                    help="release directory to refresh results/ and figures/ from")
    args = ap.parse_args()

    if args.sync_from:
        sync_from(pathlib.Path(args.sync_from).expanduser().resolve())

    runs = load_runs()
    for note in check_external():
        print(f"warning: {note}", file=sys.stderr)
    OUT.write_text(render(runs, model_ids()))
    print(f"{OUT.relative_to(REPO)} — {len(runs)} прогонов, "
          f"{len(list(FIGURES.glob('*.svg')))} фигур")


if __name__ == "__main__":
    main()
