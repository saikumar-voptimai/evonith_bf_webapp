"""Defensive orientation and fullscreen behavior."""

from __future__ import annotations

import streamlit as st

ORIENTATION_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;padding:0;background:transparent;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.row{display:flex;align-items:center;gap:8px;min-height:44px}
button{min-height:40px;padding:0 14px;border:1px solid #3f6a96;background:#eaf2fb;color:#17324d;
border-radius:6px;font-weight:600;font-size:14px;font-family:inherit;cursor:pointer;white-space:nowrap}
button:hover{background:#dbeaf8}
button:focus-visible{outline:2px solid #1f6fb2;outline-offset:2px}
#note{font-size:12px;color:#5b6b7c;line-height:1.25}
.short{display:none}
@media (max-width:230px){.long{display:none}.short{display:inline}button{padding:0 10px}}
</style></head><body>
<div class="row"><button id="go" type="button" aria-label="Open landscape fullscreen" title="Open landscape / fullscreen"><span aria-hidden="true">&#x26F6;</span> <span class="long" id="lbl-long"></span><span class="short" id="lbl-short"></span></button><span id="note" role="status"></span></div>
<script>
(function () {
  "use strict";
  var host = window;
  try { if (window.parent && window.parent.document) { host = window.parent; } } catch (e) { host = window; }
  var btn = document.getElementById("go");
  var note = document.getElementById("note");
  function safe(fn, fallback) { try { return fn(); } catch (e) { return fallback; } }
  function isSmallScreen() { return safe(function () { return host.matchMedia("(pointer: coarse), (max-width: 900px)").matches; }, false); }
  function orientation() { return safe(function () { return host.screen.orientation; }, null) || safe(function () { return window.screen.orientation; }, null); }
  function isFullscreen() { return safe(function () { return !!(host.document.fullscreenElement || host.document.webkitFullscreenElement); }, false); }
  function lockLandscape() {
    return new Promise(function (resolve, reject) {
      try {
        var o = orientation();
        if (!o || typeof o.lock !== "function") { reject(new Error("unsupported")); return; }
        Promise.resolve(o.lock("landscape")).then(resolve, reject);
      } catch (e) { reject(e); }
    });
  }
  function enterFullscreen() {
    return new Promise(function (resolve, reject) {
      try {
        var el = host.document.documentElement;
        var req = el.requestFullscreen || el.webkitRequestFullscreen;
        if (!req) { reject(new Error("unsupported")); return; }
        Promise.resolve(req.call(el)).then(resolve, reject);
      } catch (e) { reject(e); }
    });
  }
  function exitFullscreen() {
    safe(function () { var d = host.document; (d.exitFullscreen || d.webkitExitFullscreen).call(d); });
    safe(function () { orientation().unlock(); });
  }
  function refreshLabel() {
    var on = isFullscreen();
    document.getElementById("lbl-long").textContent = on ? "Exit fullscreen" : "Open landscape / fullscreen";
    document.getElementById("lbl-short").textContent = on ? "Exit" : "Fullscreen";
  }
  btn.addEventListener("click", function () {
    note.textContent = "";
    if (isFullscreen()) { exitFullscreen(); return; }
    enterFullscreen().catch(function () {}).then(lockLandscape).catch(function () {
      note.textContent = "Your browser did not allow this. Rotate your device manually.";
    });
  });
  safe(function () { host.document.addEventListener("fullscreenchange", refreshLabel); });
  safe(function () { host.document.addEventListener("webkitfullscreenchange", refreshLabel); });
  refreshLabel();
  if (isSmallScreen()) { lockLandscape().catch(function () {}); }
})();
</script></body></html>"""


def _embed_html(markup: str, height: int) -> None:
    iframe = getattr(st, "iframe", None)
    if callable(iframe):
        iframe(markup, height=height)
        return
    import streamlit.components.v1 as components

    components.html(markup, height=height)


def render_rotate_hint() -> None:
    """Rotate your device for the best view."""
    with st.container(key="fs-rotate"):
        st.html(
            '<div class="fs-rotate-hint" role="status">'
            '<span aria-hidden="true">⟳</span>'
            "<span>Rotate your device for the best view.</span></div>"
        )


def render_fullscreen_control() -> None:
    with st.container(key="fs-fullscreen"):
        _embed_html(ORIENTATION_HTML, height=48)
