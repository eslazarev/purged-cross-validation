"""Local, no-API heuristic gate for AI-tell prose. A proxy, not a detector.

Scans the project's prose (README, why-this-exists, notebook markdown, CHANGELOG,
examples/README), strips code/tables/URLs/metadata, and reports:

  - burstiness  = stdev / mean of sentence word-counts
                  (uniform rhythm -> low -> AI-like; human prose ~0.6-1.0+)
  - Flesch reading ease (textstat)
  - AI-tell phrase hits (curated GPT-ism list), with the offending snippet
  - em-dash density per 1000 words
  - triad ", x, and " density per 1000 words

Exit code is nonzero if any file FAILs, so it works as a pre-commit/CI gate.
Thresholds are advisory: this catches regressions, it does not certify text.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

import textstat

REPO = Path(__file__).resolve().parent.parent

# Published artifacts only. docstmp/ is git-ignored (not pushed).
# CODE_OF_CONDUCT.md is the verbatim Contributor Covenant text -- it would
# trip the heuristic spuriously, so it is excluded on purpose.
# CITATION.cff / .zenodo.json are machine metadata, not prose.
TARGETS = [
    REPO / "README.md",
    REPO / "CONTRIBUTING.md",
    REPO / "examples" / "README.md",
    REPO / "examples" / "clinical_mortality_physionet.ipynb",
    REPO / "examples" / "predictive_maintenance_nasa.ipynb",
    REPO / "examples" / "precipitation_noaa.ipynb",
    REPO / "examples" / "energy_demand_pjm.ipynb",
    REPO / "examples" / "synthetic_leakage_proof.ipynb",
    REPO / "examples" / "ohlc_trading_signal.ipynb",
    REPO / "examples" / "model_comparison_honest_cv.ipynb",
    REPO / "examples" / "uk_smart_meter_lcl.ipynb",
    REPO / "examples" / "earthquake_magnitude_leakage.ipynb",
    REPO / "examples" / "air_quality_clock_leakage.ipynb",
    REPO / "examples" / "backtest_overfitting_audit.ipynb",
    REPO / "paper" / "paper.md",  # tracked JOSS paper
    REPO / "docs" / "index.md",
    REPO / "docs" / "quickstart.md",
    REPO / "docs" / "examples.md",
    REPO / "docs" / "methodology.md",
    REPO / "docs" / "architecture.md",
]

# Curated, evidence-backed GPT-isms. (regex, severity) — severity: 2=strong, 1=soft.
AI_TELLS: list[tuple[str, int]] = [
    (r"\bleverage[sd]?\b", 2),
    (r"\brobust(ly|ness)?\b", 1),
    (r"\bcomprehensive(ly)?\b", 2),
    (r"\bseamless(ly)?\b", 2),
    (r"\bunderscore[sd]?\b", 2),
    (r"\btestament\b", 2),
    (r"\bdelve[sd]?\b", 2),
    (r"\bmoreover\b", 2),
    (r"\bfurthermore\b", 2),
    (r"\bnotably\b", 1),
    (r"\bcrucial(ly)?\b", 1),
    (r"\bvital\b(?![\s-]signs?\b)", 1),  # not the medical term "vital signs"
    (r"\bpivotal\b", 2),
    (r"\bplethora\b", 2),
    (r"\bmyriad\b", 2),
    (
        r"\bharness(es|ing|ed)?\s+(the|your|its|their|a)\b",
        2,
    ),  # the buzzword verb, not a test harness
    (r"\bunlock(s|ing|ed)?\b", 2),
    (r"\belevate[sd]?\b", 2),
    (r"\bcutting[- ]edge\b", 2),
    (r"\bever[- ]evolving\b", 2),
    (r"\bgame[- ]chang(er|ing)\b", 2),
    (r"\btapestry\b", 2),
    (r"\bin the realm of\b", 2),
    (r"\bwhen it comes to\b", 2),
    (r"\bfirst and foremost\b", 2),
    (r"\blast but not least\b", 2),
    (r"\bneedless to say\b", 2),
    (r"\bit is worth noting\b", 2),
    (r"\bin (?:summary|conclusion)\b", 1),
    (r"\bplays? an? (?:crucial|vital|key|pivotal) role\b", 2),
    (r"\bnot just\b[^.]{1,60}?\bbut\b", 2),  # "it's not just X, but Y"
    (r"\bNet:\s", 2),
    (r"\bin today's\b", 2),
    (r"\bdiving (?:in|into)\b", 2),
]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z])")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
TRIAD = re.compile(r",\s+\w[\w-]*\s*,?\s+and\s+\w")


def strip_md(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # fenced code
    text = re.sub(r"`[^`]+`", " ", text)  # inline code
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("#", "|", ">", "![", "[!", "---", "***")):
            continue
        if re.match(r"^(URL|License|Source|DOI):", s):
            continue
        if re.match(r"^\*\*Data (source|attribution)", s):
            continue
        if re.match(r"^[-*]\s.*\s—\s", s):
            continue  # list items using em-dash as a definition separator
        if re.match(r"^-\s", s) and ("`" in line or "—" in s) and len(s) < 90:
            continue  # short data-dictionary bullets
        if re.match(r"^https?://\S+$", s):
            continue
        out.append(line)
    joined = "\n".join(out)
    joined = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", joined)  # links -> anchor
    joined = re.sub(r"https?://\S+", " ", joined)
    joined = re.sub(r"[*_>#|]", " ", joined)
    return re.sub(r"\s+", " ", joined).strip()


def extract(path: Path) -> str:
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        md = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "markdown"]
        return strip_md("\n\n".join(md))
    return strip_md(path.read_text(encoding="utf-8"))


def analyse(name: str, prose: str) -> bool:
    sents = [s for s in SENT_SPLIT.split(prose) if WORD.search(s)]
    lens = [len(WORD.findall(s)) for s in sents]
    words = sum(lens)
    if words < 40:
        print(f"\n{name}\n  (too little prose: {words} words — skipped)")
        return True

    mean = statistics.mean(lens)
    sd = statistics.pstdev(lens)
    burst = sd / mean if mean else 0.0
    flesch = textstat.flesch_reading_ease(prose)
    emdash = prose.count("—") / words * 1000
    triad = len(TRIAD.findall(prose)) / words * 1000

    hits: list[tuple[int, str]] = []
    for pat, sev in AI_TELLS:
        for m in re.finditer(pat, prose, flags=re.I):
            a, b = max(0, m.start() - 25), m.end() + 25
            hits.append((sev, f"...{prose[a:b].strip()}..."))
    strong = sum(1 for s, _ in hits if s == 2)

    burst_v = "FAIL" if burst < 0.45 else "WARN" if burst < 0.60 else "ok"
    tell_v = "FAIL" if strong >= 4 else "WARN" if strong >= 1 else "ok"
    dash_v = "WARN" if emdash > 8 else "ok"
    verdict = (
        "FAIL"
        if "FAIL" in (burst_v, tell_v)
        else ("WARN" if "WARN" in (burst_v, tell_v, dash_v) else "PASS")
    )

    print(f"\n{name}   ->  {verdict}")
    print(f"  sentences={len(sents)}  words={words}  mean_len={mean:.1f}")
    print(
        f"  burstiness={burst:.2f} [{burst_v}]  flesch={flesch:.0f}"
        f"  em-dash/1k={emdash:.1f} [{dash_v}]  triad/1k={triad:.1f}"
    )
    print(f"  AI-tell hits: {len(hits)} ({strong} strong) [{tell_v}]")
    for sev, snip in sorted(hits, reverse=True)[:12]:
        print(f"    {'!!' if sev == 2 else ' .'} {snip}")
    if len(hits) > 12:
        print(f"    ... +{len(hits) - 12} more")
    return verdict != "FAIL"


def main() -> int:
    args = [Path(a) for a in sys.argv[1:]] or TARGETS
    ok = True
    print("=" * 70)
    print("PROSE GATE — heuristic AI-tell proxy (not a detector). Advisory.")
    print("=" * 70)
    for p in args:
        if not p.exists():
            print(f"\n{p}  -> MISSING")
            ok = False
            continue
        ok &= analyse(str(p.relative_to(REPO)), extract(p))
    print("\n" + "=" * 70)
    print("RESULT:", "all files clear of FAIL" if ok else "FAIL present")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
