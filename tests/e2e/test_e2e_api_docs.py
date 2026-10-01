"""User stories: API readers see working links, not raw Sphinx markup.

Build the real site, then inspect topic pages and legacy bookmark routes.
Run with the project's ``docs`` extra installed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import pytest

import purgedcv

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.e2e
API_ROUTES = {
    "api/",
    "api/splitters/",
    "api/primitives/",
    "api/paths/",
    "api/metrics/",
    "api/overfitting/",
    "api/diagnostics/",
    "api/time/",
}


class _APIPage(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_article = False
        self.text: list[str] = []
        self.ids: set[str] = set()
        self.references: set[str] = set()
        self.links: set[str] = set()
        self.legacy_links: list[str] = []
        self.has_note = False
        self.sections: dict[str, list[str]] = {}
        self.current_section: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a" and (href := attributes.get("href")):
            self.links.add(href)
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
        if tag == "a" and "api-legacy-link" in classes and (href := attributes.get("href")):
            self.legacy_links.append(href)
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
def api_site(tmp_path_factory: pytest.TempPathFactory) -> dict[str, _APIPage]:
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
    pages = {}
    for path in site_dir.rglob("index.html"):
        page = _APIPage()
        page.feed(path.read_text(encoding="utf-8"))
        page.close()
        pages[path.relative_to(site_dir).as_posix().removesuffix("index.html")] = page
    assert pages.keys() >= API_ROUTES
    assert "purgedcv.BaseTemporalSplitter.split" in pages["api/splitters/"].ids
    return pages


def _target(route: str, href: str) -> tuple[str, str] | None:
    url = urlsplit(urljoin(f"https://docs.invalid/{route}", href))
    if url.netloc != "docs.invalid":
        return None
    return unquote(url.path.lstrip("/").removesuffix("index.html")), unquote(url.fragment)


def test_api_has_no_visible_sphinx_markup(api_site: dict[str, _APIPage]) -> None:
    for route in API_ROUTES:
        visible_text = "".join(api_site[route].text)
        assert not re.search(
            r":(?:class|meth|func|attr|mod|ref|obj|data|exc|type):|\.\.\s+note::",
            visible_text,
        ), route
    assert api_site["api/splitters/"].has_note
    assert "The subclassing interface" in "".join(api_site["api/splitters/"].text)


def test_api_cross_references_resolve(api_site: dict[str, _APIPage]) -> None:
    references = {
        target
        for route in API_ROUTES
        for href in api_site[route].references
        if (target := _target(route, href)) is not None
    }
    # These must be docstring links, not just navigation or heading anchors.
    assert {
        ("api/diagnostics/", "purgedcv.diagnostics.assert_groups_disjoint"),
        ("api/diagnostics/", "purgedcv.GroupLeakageError"),
        ("api/metrics/", "purgedcv.probabilistic_sharpe_ratio"),
        ("api/metrics/", "purgedcv.DSRDiagnostics"),
        ("api/splitters/", "purgedcv.CombinatorialPurgedCV.backtest_paths"),
    } <= references
    for route in API_ROUTES:
        for href in api_site[route].links:
            target = _target(route, href)
            if target is None:
                continue
            target_route, fragment = target
            assert target_route in api_site, f"Broken page link: {route} -> {href}"
            if fragment:
                assert fragment in api_site[target_route].ids, f"Broken anchor: {route} -> {href}"


@pytest.mark.parametrize(
    ("route", "symbol", "description"),
    [
        ("api/time/", "ArrayLike1D", "a Python sequence"),
        ("api/time/", "TimesLike", "datetime64 or timedelta64 dtype"),
        ("api/time/", "HorizonLike", "duration accepted as text or a timedelta scalar"),
        (
            "api/paths/",
            "PathMetricFn",
            "maps one path's 1-D return series to a name -> value mapping",
        ),
        (
            "api/overfitting/",
            "PerformanceMetric",
            "maps a 1-D return slice to a scalar where larger is better",
        ),
    ],
)
def test_type_alias_has_rendered_description(
    api_site: dict[str, _APIPage], route: str, symbol: str, description: str
) -> None:
    # A signature alone is not enough; explanations must render under each alias.
    assert description in api_site[route].section_text(f"purgedcv.{symbol}")


def test_with_times_example_is_rendered(api_site: dict[str, _APIPage]) -> None:
    page = api_site["api/splitters/"]
    assert "purgedcv.BaseTemporalSplitter.with_times" in page.ids
    section = page.section_text("purgedcv.BaseTemporalSplitter")
    assert "Longer labels remove more training" in section
    assert "rebound" in section
    assert "[6, 4, 6]" in section


def test_base_splitter_documents_shared_parameters(api_site: dict[str, _APIPage]) -> None:
    section = api_site["api/splitters/"].section_text("purgedcv.BaseTemporalSplitter")
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


def test_every_public_symbol_has_one_topic_and_an_overview_link(
    api_site: dict[str, _APIPage],
) -> None:
    symbols = {
        f"purgedcv.{name}"
        for name in purgedcv.__all__
        if name not in {"__version__", "diagnostics"}
    }
    symbols |= {
        "purgedcv.optuna_integration.TrialSharpeRecorder",
        "purgedcv.diagnostics.compute_overlap_fraction",
        "purgedcv.diagnostics.assert_no_temporal_leakage",
        "purgedcv.diagnostics.assert_groups_disjoint",
        "purgedcv.diagnostics.assert_embargo_respected",
    }
    targets = {_target("api/", href) for href in api_site["api/"].legacy_links}
    for symbol in symbols:
        routes = [route for route in API_ROUTES if symbol in api_site[route].ids]
        assert len(routes) == 1, f"Missing or duplicate documentation for {symbol}: {routes}"
        assert (routes[0], symbol) in targets
    # Topic pages must be visible from the sidebar, not just from the index.
    for route in API_ROUTES:
        destinations = {_target(route, href) for href in api_site[route].links}
        assert {(topic, "") for topic in API_ROUTES} <= destinations


def test_legacy_api_bookmarks_redirect_to_existing_anchors(api_site: dict[str, _APIPage]) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Legacy JavaScript routing test requires Node.js")
    script = REPO_ROOT / "docs" / "javascripts" / "api-redirects.js"
    anchors = {
        anchor: route
        for route in API_ROUTES
        for anchor in api_site[route].ids
        if anchor.startswith("purgedcv.")
    }
    runner = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
const {links, anchors} = JSON.parse(fs.readFileSync(0, 'utf8'));
function navigate(url, mode, activeLinks = links) {
  const location = new URL(url);
  let replaced = null;
  location.replace = value => { replaced = value; };
  const handlers = {};
  const context = {
    URL, decodeURIComponent,
    window: {location, addEventListener: (name, fn) => {handlers[name] = fn;}},
    document: {
      readyState: mode === 'loading' ? 'loading' : 'complete',
      addEventListener: (name, fn) => {handlers[name] = fn;},
      querySelectorAll: () => activeLinks.map(href => ({href: new URL(href, url).href}))
    }
  };
  if (mode === 'instant') context.document$ = {subscribe: fn => {handlers.instant = fn;}};
  vm.runInNewContext(source, context);
  if (mode === 'instant') handlers.instant();
  if (mode === 'loading') handlers.DOMContentLoaded();
  if (mode === 'hashchange') handlers.hashchange();
  return replaced;
}
for (const prefix of ['/', '/purged-cross-validation/']) {
  for (const mode of ['ready', 'loading', 'instant', 'hashchange']) {
    for (const [anchor, route] of Object.entries(anchors)) {
      const url = 'https://docs.invalid' + prefix + 'api/?source=bookmark#' + encodeURIComponent(anchor);
      const target = new URL(navigate(url, mode));
      assert.equal(target.pathname, prefix + route);
      assert.equal(decodeURIComponent(target.hash.slice(1)), anchor);
      assert.equal(target.search, '?source=bookmark');
    }
    for (const hash of ['', '#splitters', '#purgedcv.Unknown', '#%E0%A4%A']) {
      assert.equal(navigate('https://docs.invalid' + prefix + 'api/' + hash, mode), null);
    }
  }
}
assert.equal(navigate('https://docs.invalid/api/splitters/#purgedcv.PurgedKFold', 'instant', []), null);
assert.equal(navigate('https://docs.invalid/api/#purgedcv.purge', 'ready', ['https://other.invalid/api/#purgedcv.purge']), null);
assert.equal(navigate('https://docs.invalid/api/#purgedcv.purge', 'ready', ['#purgedcv.purge']), null);
console.log('Legacy API bookmarks OK');
"""
    result = subprocess.run(
        [node, "-e", runner, str(script)],
        input=json.dumps({"links": api_site["api/"].legacy_links, "anchors": anchors}),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Legacy API bookmarks OK" in result.stdout
