// Keep bookmarks to the former single-page API usable. The overview's links
// are the routing table, so page moves do not need a second list of symbols.
(() => {
  function redirectLegacyAnchor() {
    let fragment;
    try {
      fragment = decodeURIComponent(window.location.hash.slice(1));
    } catch {
      return; // A malformed URL must not break the rest of the documentation.
    }
    if (!fragment.startsWith("purgedcv.")) return;

    for (const link of document.querySelectorAll("article a.api-legacy-link")) {
      const target = new URL(link.href, window.location.href);
      const symbol = decodeURIComponent(target.hash.slice(1));
      if (fragment !== symbol && !fragment.startsWith(symbol + ".")) continue;
      if (target.origin !== window.location.origin) continue;
      if (target.pathname === window.location.pathname) continue;
      target.hash = fragment; // Preserve method, attribute, and parameter anchors.
      target.search = window.location.search;
      window.location.replace(target.href);
      return;
    }
  }

  window.addEventListener("hashchange", redirectLegacyAnchor);
  // Material's instant navigation replaces the article without a full reload.
  if (typeof document$ !== "undefined") {
    document$.subscribe(redirectLegacyAnchor);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", redirectLegacyAnchor);
  } else {
    redirectLegacyAnchor();
  }
})();
