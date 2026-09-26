"""User stories: API readers see working links, not raw Sphinx markup.

Build the real site in a subprocess, then inspect its rendered API page.
Run with the project's ``docs`` extra installed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.e2e


class _APIPage(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_article = False
        self.text: list[str] = []
        self.ids: set[str] = set()
        self.references: set[str] = set()
        self.has_note = False
        self.sections: dict[str, list[str]] = {}
        self.current_section: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "article":
            self.in_article = True
        if not self.in_article:
            return
        if identifier := attributes.get("id"):
            self.ids.add(identifier)
            if tag == "h2":
                self.current_section = identifier
                self.sections[identifier] = []
        classes = (attributes.get("class") or "").split()
        if tag == "a" and "autorefs" in classes and (href := attributes.get("href")):
            self.references.add(href)
        if tag in {"div", "details"} and "note" in classes:
            self.has_note = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "article":
            self.in_article = False

    def handle_data(self, data: str) -> None:
        if self.in_article:
            self.text.append(data)
            if self.current_section is not None:
                self.sections[self.current_section].append(data)

    def section_text(self, identifier: str) -> str:
        return " ".join("".join(self.sections[identifier]).split())


@pytest.fixture(scope="module")
def api_page(tmp_path_factory: pytest.TempPathFactory) -> _APIPage:
    pytest.importorskip("mkdocs", reason="API rendering tests require the docs extra")
    site_dir = tmp_path_factory.mktemp("api-site")
    result = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(site_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, f"MkDocs failed:\n{result.stdout}\n{result.stderr}"
    page = _APIPage()
    page.feed((site_dir / "api" / "index.html").read_text(encoding="utf-8"))
    page.close()
    assert "purgedcv.BaseTemporalSplitter.split" in page.ids
    return page


def test_api_has_no_visible_sphinx_markup(api_page: _APIPage) -> None:
    visible_text = "".join(api_page.text)
    assert not re.search(
        r":(?:class|meth|func|attr|mod|ref|obj|data|exc|type):|\.\.\s+note::",
        visible_text,
    )
    assert api_page.has_note
    assert "The subclassing interface" in visible_text


def test_api_cross_references_resolve(api_page: _APIPage) -> None:
    # These must be docstring links, not just navigation or heading anchors.
    assert {
        "#purgedcv.diagnostics.assert_groups_disjoint",
        "#purgedcv.GroupLeakageError",
        "#purgedcv.probabilistic_sharpe_ratio",
        "#purgedcv.DSRDiagnostics",
        "#purgedcv.CombinatorialPurgedCV.backtest_paths",
    } <= api_page.references
    missing = {
        href
        for href in api_page.references
        if href.startswith("#") and unquote(href[1:]) not in api_page.ids
    }
    assert not missing, f"Broken API cross-references: {sorted(missing)}"


@pytest.mark.parametrize(
    ("symbol", "description"),
    [
        ("ArrayLike1D", "a Python sequence"),
        ("TimesLike", "datetime64 or timedelta64 dtype"),
        ("HorizonLike", "duration accepted as text or a timedelta scalar"),
        ("PathMetricFn", "maps one path's 1-D return series to a name -> value mapping"),
        ("PerformanceMetric", "maps a 1-D return slice to a scalar where larger is better"),
    ],
)
def test_type_alias_has_rendered_description(
    api_page: _APIPage, symbol: str, description: str
) -> None:
    # A signature alone is not enough; explanations must render under each alias.
    assert description in api_page.section_text(f"purgedcv.{symbol}")


def test_base_splitter_documents_shared_parameters(api_page: _APIPage) -> None:
    section = api_page.section_text("purgedcv.BaseTemporalSplitter")
    assert "Parameters:" in section
    descriptions = {
        "prediction_times": "Prediction times for all samples in positional row order",
        "evaluation_times": "End of each sample's label horizon",
        "purge_horizon": "Zero padding does not disable label-overlap purging",
        "embargo": "Post-test wall-clock duration",
        "embargo_observations": "Number of row positions immediately after each contiguous test block",
        "embargo_fraction": "Fraction of the full dataset length",
        "groups": "Optional group labels in positional row order",
    }
    for name, description in descriptions.items():
        assert name in section
        assert description in section, f"Missing shared parameter description: {name}"
    assert "at most one embargo mode" in section
