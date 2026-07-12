"""Read-only local web monitor for jlens-mlx band-fit progress.

Usage:
    uv run python scripts/fit_monitor.py --out out/band-n14-fixed [--port 8765]

Serves a small dashboard (polled every 3s) over whatever a LIVE fit run is writing to disk:
`ckpt/progress.json` (heartbeat), `ckpt/ckpt.json` (banked items), `ckpt/corpus.json` (fitting
corpus, summarized -- never shipped raw), `ckpt/corpus_decoded.md` (human-readable prompts +
completions), and the run's `.log` file one level up from `--out`.

Hard invariants:
  - stdlib ONLY -- no pip deps, no bun, no build step, no framework. `import mlx` is FORBIDDEN
    (this tool never touches the GPU).
  - READ-ONLY -- never writes to any `out/` directory, never writes anywhere at all.
  - `--out <dir>` is the ONLY readable root (plus one fixed, server-computed sibling `.log` file
    one level up -- never a client-supplied path). All filenames served are hardcoded server-side;
    no client input is ever joined into a filesystem path, so traversal isn't reachable through
    any route -- `safe_path()` is still applied as defense in depth.

Endpoints (all GET, all read-only):
    /                    -- the HTML dashboard
    /api/progress        -- ckpt/progress.json, re-read fresh every request ({} if absent)
    /api/ckpt            -- ckpt/ckpt.json, re-read fresh every request ({} if absent)
    /api/corpus           -- small derived summary of ckpt/corpus.json (never raw token ids/items)
    /api/corpus_items     -- per-item summary (index/stratum/on_policy/n_tokens/n_fitted_positions)
                             joined with the decoded prompt/completion text, plus a provenance block
    /api/log?n=200        -- last n lines of the run log as JSON (default 200, hard cap 2000), plus
                             a parsed chunk-progress tail for stall detection
    /decoded              -- raw ckpt/corpus_decoded.md (text/markdown), 404 if absent
    anything else          -- 404 (never a raw traceback / 500)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

# --- log line format ------------------------------------------------------------------------
# "[14:04:44]   item 1/12 chunk 1/40 (2%) item_elapsed=66s"
CHUNK_RE = re.compile(
    r"\[(\d{2}:\d{2}:\d{2})\]\s*item (\d+)/(\d+) chunk (\d+)/(\d+) \((\d+)%\) item_elapsed=(\d+)s")

# --- corpus_decoded.md format ----------------------------------------------------------------
# "## item 0  [stratum=safety  on_policy=True  tokens=78  masked_positions=47]"
DECODED_HEADER_RE = re.compile(
    r"^## item (\d+)\s+\[stratum=([^\s\]]+)\s+on_policy=(True|False)\s+"
    r"tokens=(\d+)\s+masked_positions=(\d+)\]\s*$", re.MULTILINE)
PROMPT_MARKER = "--- prompt (through the generation prompt) ---"
RESPONSE_MARKER = "--- model's on-policy response (the span J reads) ---"


# --- pure, importable, unit-testable functions ------------------------------------------------

def safe_path(root: Path, relative: str) -> Path | None:
    """Resolve `relative` under `root`, refusing anything that escapes it -- `../` traversal,
    an absolute path (which `Path.__truediv__` would otherwise silently replace `root` with),
    or a symlink escape. Returns None if the resolved path would land outside `root`."""
    try:
        root_resolved = root.resolve()
        candidate = (root_resolved / relative).resolve()
        candidate.relative_to(root_resolved)
    except (ValueError, OSError):
        return None
    return candidate


def parse_chunk_progress(lines: list[str]) -> list[dict]:
    """Parse chunk-progress heartbeat lines out of the run log. Returns one dict per matching
    line: ts, item, n_items, chunk, n_chunks, pct, item_elapsed, delta, stall.

    `delta` is the change in `item_elapsed` since the previous matched line for the SAME item
    number (None for the first chunk seen for an item -- item_elapsed resets at an item boundary,
    which is expected, not a stall). `stall` is True iff delta == 0 (a frozen heartbeat)."""
    results: list[dict] = []
    last_elapsed: dict[int, int] = {}
    for line in lines:
        m = CHUNK_RE.search(line)
        if not m:
            continue
        ts, item_s, n_items_s, chunk_s, n_chunks_s, pct_s, elapsed_s = m.groups()
        item, n_items, chunk, n_chunks = int(item_s), int(n_items_s), int(chunk_s), int(n_chunks_s)
        pct, elapsed = int(pct_s), int(elapsed_s)
        prev = last_elapsed.get(item)
        delta = elapsed - prev if prev is not None else None
        last_elapsed[item] = elapsed
        results.append({
            "ts": ts, "item": item, "n_items": n_items, "chunk": chunk, "n_chunks": n_chunks,
            "pct": pct, "item_elapsed": elapsed, "delta": delta, "stall": delta == 0,
        })
    return results


def summarize_corpus(corpus: dict) -> dict:
    """Derive the small, privacy-safe summary served at /api/corpus -- NEVER the raw `items`
    list (token ids). Reads keys defensively; missing fields degrade to None rather than raising."""
    if not corpus:
        return {}
    provenance = corpus.get("provenance") or {}
    diversity = provenance.get("diversity") or {}
    n_items = provenance.get("n_prompts")
    if n_items is None:
        n_items = diversity.get("n_items")
    on_policy_div = diversity.get("on_policy") or {}
    return {
        "n_items": n_items,
        "diversity": {
            "shared_fraction": diversity.get("shared_fraction"),
            "on_policy": {"shared_fraction": on_policy_div.get("shared_fraction")},
        },
        "strata": provenance.get("strata") or {},
        "dropped_over_len": provenance.get("dropped_over_len"),
        "enable_thinking": provenance.get("enable_thinking"),
    }


def corpus_items_summary(corpus: dict) -> list[dict]:
    """Per-item metadata only -- index/stratum/on_policy/n_tokens/n_fitted_positions. NEVER the
    raw `ids`/`positions` token-id lists (privacy + payload size)."""
    items = corpus.get("items") or []
    out = []
    for i, it in enumerate(items):
        out.append({
            "index": i,
            "stratum": it.get("stratum"),
            "on_policy": bool(it.get("on_policy", False)),
            "n_tokens": len(it.get("ids") or []),
            "n_fitted_positions": len(it.get("positions") or []),
        })
    return out


def parse_decoded_md(text: str) -> list[dict]:
    """Parse `corpus_decoded.md` into per-item {index, stratum, on_policy, tokens,
    masked_positions, prompt, completion}. On-policy items have a prompt/completion split (the
    two `---` markers); off-policy (human-text) items have no completion -- the whole block is
    the prompt/content text and `completion` is None. Item `index` here matches the `## item N`
    header, which must line up with corpus_items_summary()'s `index` (both are corpus.json
    item-list order)."""
    matches = list(DECODED_HEADER_RE.finditer(text))
    results = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        prompt: str | None
        completion: str | None
        if PROMPT_MARKER in block and RESPONSE_MARKER in block:
            p_start = block.index(PROMPT_MARKER) + len(PROMPT_MARKER)
            r_start = block.index(RESPONSE_MARKER)
            prompt = block[p_start:r_start].strip("\n")
            completion = block[r_start + len(RESPONSE_MARKER):].strip("\n")
        else:
            prompt = block.strip("\n")
            completion = None
        results.append({
            "index": int(m.group(1)),
            "stratum": m.group(2),
            "on_policy": m.group(3) == "True",
            "tokens": int(m.group(4)),
            "masked_positions": int(m.group(5)),
            "prompt": prompt,
            "completion": completion,
        })
    return results


# --- file I/O helpers (never raise -- callers get a graceful empty value) ---------------------

def _read_json_safe(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        # Transient: progress.json is rewritten atomically (tmp+rename) so this should be rare,
        # but never crash the monitor over a race with the live writer.
        return {}


def _tail_log(path: Path, n: int) -> list[str]:
    if n <= 0 or not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            tail = deque(f, maxlen=n)
        return [line.rstrip("\n") for line in tail]
    except OSError:
        return []


# --- HTTP handler ------------------------------------------------------------------------------

def build_handler(out_root: Path) -> type[BaseHTTPRequestHandler]:
    """Bind a request handler class to a specific sandboxed `--out` root (and its one fixed
    sibling log path) via closure, so nothing depends on mutable module-level globals."""
    ckpt_dir = out_root / "ckpt"
    progress_path = safe_path(ckpt_dir, "progress.json")
    ckpt_path = safe_path(ckpt_dir, "ckpt.json")
    corpus_path = safe_path(ckpt_dir, "corpus.json")
    decoded_path = safe_path(ckpt_dir, "corpus_decoded.md")
    # The run log is a SIBLING of the run dir (one level up), not inside --out. This single path
    # is computed once, here, from --out -- it is never derived from client/request input, so the
    # "readable root" traversal guard doesn't apply to it (there is nothing for a client to steer).
    log_path = out_root.parent / f"{out_root.name}.log"

    class Handler(BaseHTTPRequestHandler):
        server_version = "fit-monitor/1.0"

        def log_message(self, fmt, *args):  # noqa: A003 -- stdlib signature
            pass  # keep the terminal quiet; this is a local dev tool

        def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj, status: int = 200) -> None:
            self._send_bytes(json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8",
                              status)

        def _send_text(self, text: str, content_type: str, status: int = 200) -> None:
            self._send_bytes(text.encode("utf-8"), content_type, status)

        def _send_404(self) -> None:
            self._send_bytes(b"not found", "text/plain; charset=utf-8", 404)

        def do_GET(self) -> None:  # noqa: N802 -- stdlib signature
            try:
                self._route()
            except Exception:
                # Never leak a traceback to the client -- a clean 404, not a 500.
                self._send_404()

        def _route(self) -> None:
            parsed = urlsplit(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path == "/":
                self._send_text(INDEX_HTML, "text/html; charset=utf-8")
                return

            if path == "/api/progress":
                self._send_json(_read_json_safe(progress_path) if progress_path else {})
                return

            if path == "/api/ckpt":
                self._send_json(_read_json_safe(ckpt_path) if ckpt_path else {})
                return

            if path == "/api/corpus":
                corpus = _read_json_safe(corpus_path) if corpus_path else {}
                self._send_json(summarize_corpus(corpus))
                return

            if path == "/api/corpus_items":
                corpus = _read_json_safe(corpus_path) if corpus_path else {}
                if not corpus:
                    self._send_json({"items": [], "provenance": {}})
                    return
                items = corpus_items_summary(corpus)
                decoded_by_index: dict[int, dict] = {}
                if decoded_path and decoded_path.exists():
                    try:
                        text = decoded_path.read_text(encoding="utf-8", errors="replace")
                        decoded_by_index = {d["index"]: d for d in parse_decoded_md(text)}
                    except OSError:
                        decoded_by_index = {}
                for it in items:
                    d = decoded_by_index.get(it["index"])
                    it["prompt"] = d["prompt"] if d else None
                    it["completion"] = d["completion"] if d else None
                self._send_json({"items": items, "provenance": summarize_corpus(corpus)})
                return

            if path == "/api/log":
                n_raw = query.get("n", ["200"])[0]
                try:
                    n = int(n_raw)
                except (TypeError, ValueError):
                    n = 200
                if n < 0:
                    n = 200
                n = min(n, 2000)
                lines = _tail_log(log_path, n)
                self._send_json({"lines": lines, "chunk_progress": parse_chunk_progress(lines)[-20:]})
                return

            if path == "/decoded":
                if not decoded_path or not decoded_path.exists():
                    self._send_404()
                    return
                text = decoded_path.read_text(encoding="utf-8", errors="replace")
                self._send_text(text, "text/markdown; charset=utf-8")
                return

            self._send_404()

    return Handler


# --- inline dashboard (static -- all data comes from fetch(), never templated) ----------------

INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>jlens-mlx fit monitor</title>
<style>
:root { color-scheme: dark; --bg:#0b0d10; --panel:#12151a; --text:#e6e8eb; --muted:#8a919b;
  --accent:#4f9dff; --ok:#3ecf8e; --warn:#e0b03e; --bad:#e0554e; --border:#232830; }
@media (prefers-color-scheme: light) {
  :root { color-scheme: light; --bg:#f6f7f9; --panel:#ffffff; --text:#1b1f24; --muted:#5b6470;
    --border:#e1e4e8; }
}
* { box-sizing: border-box; }
body { margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background:var(--bg); color:var(--text); }
header { display:flex; align-items:center; justify-content:space-between; padding:12px 20px;
  border-bottom:1px solid var(--border); }
h1 { font-size:16px; margin:0; font-weight:600; }
nav .tab { background:none; border:1px solid var(--border); color:var(--text); padding:6px 14px;
  margin-left:8px; border-radius:6px; cursor:pointer; font-size:13px; }
nav .tab.active { background:var(--accent); color:#fff; border-color:var(--accent); }
section { padding:20px; max-width:900px; margin:0 auto; }
.hidden { display:none !important; }
.empty { color:var(--muted); padding:40px 0; text-align:center; font-size:14px; }
.status-line { display:flex; gap:16px; font-size:20px; font-weight:600; margin-bottom:12px;
  flex-wrap:wrap; }
.bar-label { font-size:12px; color:var(--muted); margin-top:10px; }
.bar { background:var(--panel); border:1px solid var(--border); border-radius:6px; height:18px;
  overflow:hidden; margin-top:4px; }
.bar-small { height:10px; }
.bar-fill { background:var(--accent); height:100%; width:0%; transition:width .3s ease; }
.stat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:12px;
  margin-top:20px; }
.stat { background:var(--panel); border:1px solid var(--border); border-radius:8px;
  padding:10px 12px; }
.stat-label { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }
.stat-value { font-size:18px; font-weight:600; margin-top:2px; }
.last-update { font-size:12px; color:var(--muted); margin-top:16px; }
.last-update.stale { color:var(--bad); font-weight:600; }
.corpus-health-mini, .corpus-health { background:var(--panel); border:1px solid var(--border);
  border-radius:8px; padding:10px 12px; margin-top:16px; font-size:13px; }
.health-row { display:flex; justify-content:space-between; padding:3px 0; gap:12px; }
.ok { color:var(--ok); font-weight:600; }
.warn { color:var(--warn); font-weight:600; }
.bad { color:var(--bad); font-weight:600; }
.stall-strip { margin-top:16px; font-size:12px; display:flex; align-items:center; gap:4px;
  flex-wrap:wrap; }
.stall-strip .tick { display:inline-block; width:8px; height:8px; border-radius:2px;
  background:var(--ok); }
.stall-strip .tick.bad { background:var(--bad); }
.muted { color:var(--muted); }
.small { font-size:11px; }
h3 { font-size:13px; color:var(--muted); margin:20px 0 6px; text-transform:uppercase;
  letter-spacing:.03em; }
.log-tail { background:#000; color:#8fe0a8; font-family: ui-monospace, SFMono-Regular, Menlo,
  monospace; font-size:11px; padding:10px; border-radius:8px; max-height:260px; overflow-y:auto;
  white-space:pre-wrap; word-break:break-word; border:1px solid var(--border); }
.provenance { background:var(--panel); border:1px solid var(--border); border-radius:8px;
  padding:10px 12px; margin-top:12px; font-size:13px; }
.provenance > div { padding:2px 0; }
.strata-list { margin-top:6px; display:flex; flex-wrap:wrap; gap:6px; }
.strata-item { background:var(--bg); border:1px solid var(--border); border-radius:4px;
  padding:2px 6px; font-size:11px; }
.bucket-h { margin:20px 0 8px; font-size:13px; text-transform:capitalize; }
.bucket-h.harmful { color:var(--bad); }
.bucket-h.benign { color:var(--ok); }
.bucket-h.other { color:var(--accent); }
.card { background:var(--panel); border:1px solid var(--border); border-left-width:4px;
  border-radius:8px; padding:12px; margin-bottom:10px; }
.card.harmful { border-left-color:var(--bad); }
.card.benign { border-left-color:var(--ok); }
.card.other { border-left-color:var(--accent); }
.card-head { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
.badge { font-size:11px; padding:2px 6px; border-radius:4px; font-weight:600; }
.badge.harmful { background:rgba(224,85,78,.15); color:var(--bad); }
.badge.benign { background:rgba(62,207,142,.15); color:var(--ok); }
.badge.other { background:rgba(79,157,255,.15); color:var(--accent); }
.chip { font-size:11px; padding:2px 6px; border-radius:10px; border:1px solid var(--border);
  color:var(--muted); }
.card-label { font-size:10px; color:var(--muted); text-transform:uppercase; margin:8px 0 2px; }
.card-text { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px;
  white-space:pre-wrap; word-break:break-word; margin:0; background:var(--bg);
  border:1px solid var(--border); border-radius:6px; padding:8px; }
a { color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>jlens-mlx fit monitor</h1>
  <nav>
    <button id="tab-run" class="tab active" onclick="showTab('run')">Run</button>
    <button id="tab-corpus" class="tab" onclick="showTab('corpus')">Corpus</button>
  </nav>
</header>

<section id="panel-run">
  <div id="run-empty" class="empty">Waiting for fit to start&hellip;</div>
  <div id="run-content" class="hidden">
    <div class="status-line">
      <span id="status-item">item &mdash; / &mdash;</span>
      <span id="status-chunk">chunk &mdash; / &mdash;</span>
    </div>
    <div class="bar-label">Overall (positions)</div>
    <div class="bar"><div id="bar-overall" class="bar-fill"></div></div>
    <div class="bar-label">This item (chunks)</div>
    <div class="bar bar-small"><div id="bar-chunk" class="bar-fill"></div></div>

    <div class="stat-grid">
      <div class="stat"><div class="stat-label">ETA</div><div class="stat-value" id="stat-eta">&mdash;</div></div>
      <div class="stat"><div class="stat-label">sec/pos</div><div class="stat-value" id="stat-secpos">&mdash;</div></div>
      <div class="stat"><div class="stat-label">peak GB</div><div class="stat-value" id="stat-peak">&mdash;</div></div>
      <div class="stat"><div class="stat-label">items banked</div><div class="stat-value" id="stat-banked">&mdash;</div></div>
    </div>

    <div id="last-update" class="last-update">last update &mdash;</div>

    <div class="corpus-health-mini" id="corpus-health-mini"></div>

    <div class="stall-strip" id="stall-strip"></div>

    <h3>Log tail</h3>
    <pre id="log-tail" class="log-tail"></pre>
  </div>
</section>

<section id="panel-corpus" class="hidden">
  <div id="corpus-empty" class="empty">No corpus.json yet.</div>
  <div id="corpus-content" class="hidden">
    <div class="corpus-health" id="corpus-health-full"></div>
    <div class="provenance" id="provenance"></div>
    <div id="corpus-cards"></div>
  </div>
</section>

<script>
const POLL_MS = 3000;
let currentTab = 'run';

function showTab(tab) {
  currentTab = tab;
  document.getElementById('tab-run').classList.toggle('active', tab === 'run');
  document.getElementById('tab-corpus').classList.toggle('active', tab === 'corpus');
  document.getElementById('panel-run').classList.toggle('hidden', tab !== 'run');
  document.getElementById('panel-corpus').classList.toggle('hidden', tab !== 'corpus');
  tick();
}

function fmtEta(s) {
  if (s === null || s === undefined) return 'pending (first item)';
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const parts = [];
  if (h) parts.push(h + 'h');
  if (h || m) parts.push(m + 'm');
  parts.push(sec + 's');
  return parts.join(' ');
}

function fmtAgo(tsStr) {
  if (!tsStr) return null;
  // "2026-07-12T14:11:19" -- no timezone in the source; treat as local wall-clock time.
  const t = new Date(tsStr);
  if (isNaN(t.getTime())) return null;
  return Math.round((Date.now() - t.getTime()) / 1000);
}

function sharedFractionClass(v) {
  if (v === null || v === undefined) return '';
  if (v < 0.35) return 'ok';
  if (v <= 0.5) return 'warn';
  return 'bad';
}

function bucketFor(stratum) {
  const s = (stratum || '').toLowerCase();
  if (s.includes('harm') || s.includes('safety')) return 'harmful';
  if (s.includes('benign')) return 'benign';
  return 'other';
}

function esc(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

async function fetchJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(url + ': ' + r.status);
  return r.json();
}

async function refreshCorpusHealth(targetId) {
  let corpus = {};
  try { corpus = await fetchJSON('/api/corpus'); } catch (e) { /* keep polling */ }
  const el = document.getElementById(targetId);
  if (!corpus || !corpus.diversity) {
    el.innerHTML = '<span class="muted">no corpus.json yet</span>';
    return corpus;
  }
  const overall = corpus.diversity.shared_fraction;
  const onp = corpus.diversity.on_policy ? corpus.diversity.on_policy.shared_fraction : null;
  el.innerHTML =
    '<div class="health-row"><span>overall shared_fraction</span><span class="' +
      sharedFractionClass(overall) + '">' + (overall != null ? overall.toFixed(3) : '—') + '</span></div>' +
    '<div class="health-row"><span>on-policy shared_fraction</span><span class="' +
      sharedFractionClass(onp) + '">' + (onp != null ? onp.toFixed(3) : '—') + '</span></div>' +
    '<div style="margin-top:6px"><a href="/decoded" target="_blank">read the completions</a></div>';
  return corpus;
}

async function refreshRun() {
  let progress = {}, ckpt = {}, logData = { lines: [], chunk_progress: [] };
  try { progress = await fetchJSON('/api/progress'); } catch (e) { /* keep polling */ }
  try { ckpt = await fetchJSON('/api/ckpt'); } catch (e) { /* keep polling */ }
  try { logData = await fetchJSON('/api/log?n=200'); } catch (e) { /* keep polling */ }

  const empty = !progress || Object.keys(progress).length === 0;
  document.getElementById('run-empty').classList.toggle('hidden', !empty);
  document.getElementById('run-content').classList.toggle('hidden', empty);
  if (empty) return;

  document.getElementById('status-item').textContent =
    'item ' + (progress.item ?? '—') + ' / ' + (progress.n_items ?? '—');
  document.getElementById('status-chunk').textContent =
    'chunk ' + (progress.chunk ?? '—') + ' / ' + (progress.n_chunks ?? '—');

  const posDone = progress.positions_done ?? 0, posTotal = progress.positions_total ?? 0;
  document.getElementById('bar-overall').style.width =
    posTotal ? Math.min(100, 100 * posDone / posTotal) + '%' : '0%';
  const chunk = progress.chunk ?? 0, nChunks = progress.n_chunks ?? 0;
  document.getElementById('bar-chunk').style.width =
    nChunks ? Math.min(100, 100 * chunk / nChunks) + '%' : '0%';

  document.getElementById('stat-eta').textContent = fmtEta(progress.eta_s);
  document.getElementById('stat-secpos').textContent =
    progress.sec_per_pos != null ? Number(progress.sec_per_pos).toFixed(2) : '—';
  document.getElementById('stat-peak').textContent =
    progress.peak_gb != null ? Number(progress.peak_gb).toFixed(1) + ' GB' : '—';

  const banked = ckpt.n_done;
  const total = ckpt.n_total ?? progress.n_items;
  document.getElementById('stat-banked').textContent =
    (banked ?? '—') + ' / ' + (total ?? '—');

  const ago = fmtAgo(progress.ts);
  const lu = document.getElementById('last-update');
  if (ago === null) {
    lu.textContent = 'last update —';
    lu.classList.remove('stale');
  } else {
    lu.textContent = 'last update ' + ago + 's ago';
    lu.classList.toggle('stale', ago > 180);
  }

  await refreshCorpusHealth('corpus-health-mini');

  const cp = logData.chunk_progress || [];
  const strip = document.getElementById('stall-strip');
  if (cp.length) {
    const last = cp[cp.length - 1];
    const frozen = last.delta === 0;
    strip.innerHTML = '<span class="' + (frozen ? 'bad' : 'ok') + '">' +
      (frozen ? 'STALL: item_elapsed frozen' : 'progressing') + '</span>' +
      cp.map(function (c) {
        return '<span class="tick' + (c.delta === 0 ? ' bad' : '') + '" title="item ' + c.item +
          ' chunk ' + c.chunk + '/' + c.n_chunks + ' +' + (c.delta ?? '—') + 's"></span>';
      }).join('');
  } else {
    strip.innerHTML = '<span class="muted">no chunk-progress lines yet</span>';
  }

  const tail = (logData.lines || []).slice(-40);
  const pre = document.getElementById('log-tail');
  pre.textContent = tail.join('\\n');
  pre.scrollTop = pre.scrollHeight;
}

function renderCard(it) {
  const bucket = bucketFor(it.stratum);
  const completionHtml = (it.completion !== null && it.completion !== undefined)
    ? '<div class="card-label">completion</div><pre class="card-text">' + esc(it.completion) + '</pre>'
    : '<div class="muted small" style="margin-top:8px">off-policy item &mdash; no separate completion, positions are read from the prompt text itself</div>';
  return '<div class="card ' + bucket + '">' +
    '<div class="card-head">' +
      '<span class="badge ' + bucket + '">' + esc(it.stratum) + '</span>' +
      '<span class="chip">' + (it.on_policy ? 'on-policy' : 'off-policy') + '</span>' +
      '<span class="muted">#' + it.index + ' · ' + it.n_tokens + ' tok · ' +
        it.n_fitted_positions + ' fitted</span>' +
    '</div>' +
    '<div class="card-label">prompt</div>' +
    '<pre class="card-text">' + esc(it.prompt) + '</pre>' +
    completionHtml +
    '</div>';
}

async function refreshCorpus() {
  const corpus = await refreshCorpusHealth('corpus-health-full');
  const empty = !corpus || !corpus.n_items;
  document.getElementById('corpus-empty').classList.toggle('hidden', !empty);
  document.getElementById('corpus-content').classList.toggle('hidden', empty);
  if (empty) return;

  const prov = document.getElementById('provenance');
  const strata = corpus.strata || {};
  prov.innerHTML =
    '<div>n_items: ' + corpus.n_items + '</div>' +
    '<div>dropped_over_len: ' + (corpus.dropped_over_len ?? '—') + '</div>' +
    '<div>enable_thinking: ' + (corpus.enable_thinking ?? '—') + '</div>' +
    '<div class="strata-list">' + Object.entries(strata).map(function (kv) {
      return '<span class="strata-item">' + esc(kv[0]) + ': ' + kv[1] + '</span>';
    }).join('') + '</div>';

  let items = [];
  try { items = (await fetchJSON('/api/corpus_items')).items || []; } catch (e) { /* keep polling */ }

  const buckets = { harmful: [], benign: [], other: [] };
  for (const it of items) buckets[bucketFor(it.stratum)].push(it);

  const cardsEl = document.getElementById('corpus-cards');
  cardsEl.innerHTML = ['harmful', 'benign', 'other'].map(function (b) {
    if (!buckets[b].length) return '';
    return '<h4 class="bucket-h ' + b + '">' + b + ' (' + buckets[b].length + ')</h4>' +
      buckets[b].map(renderCard).join('');
  }).join('');
}

async function tick() {
  try {
    if (currentTab === 'run') await refreshRun();
    else await refreshCorpus();
  } catch (e) { /* keep polling even if one cycle fails */ }
}

tick();
setInterval(tick, POLL_MS);
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only local web monitor for jlens-mlx band-fit progress.")
    parser.add_argument("--out", required=True,
                        help="Run directory to monitor (contains ckpt/); resolved to an absolute "
                             "path and used as the sandboxed readable root.")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to bind (default 8765; 0 picks an ephemeral port).")
    args = parser.parse_args(argv)

    out_root = Path(args.out).resolve()
    handler_cls = build_handler(out_root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_cls)
    print(f"http://127.0.0.1:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
