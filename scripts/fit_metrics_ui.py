"""Read-only web dashboard for the jlens-mlx fit-metrics DuckDB store.

Usage:
    uv run python scripts/fit_metrics_ui.py [--db out/fit_metrics.duckdb] [--port 8766]

This is the ANALYTICAL sibling of `fit_monitor.py`: that tool watches ONE live run's
`ckpt/*` on disk; this one reads the cross-run history that `scripts/fit_metrics.py`
accumulates into a DuckDB store (`dim_run`, `fact_item_fit`, and the `v_peak_vs_seq`
/ `v_peak_vs_positions` / `v_throughput` views). It never touches a live fit, never
touches the GPU, and never writes to the store.

Hard invariants:
  - stdlib + `duckdb` ONLY -- no other pip deps, no bun, no build step, no framework.
    `import mlx` is FORBIDDEN (this tool never touches the GPU).
  - READ-ONLY, always. Every DuckDB connection is opened with `read_only=True`; this
    tool never issues a write and never creates the database file if it's missing
    (a missing/empty store degrades to an empty-state page, not an error). A fit run
    may be writing to the same file concurrently elsewhere -- opening read-only is
    what makes that safe to coexist with, and each request opens a fresh connection
    so it always sees the latest committed rows rather than a stale snapshot.
  - No client input ever becomes a filesystem path. The only client-supplied value
    that reaches a query is `?run_id=`, which is always passed as a bound DuckDB
    parameter (`?`), never string-interpolated into SQL. Route matching is an exact
    allowlist (`/`, `/api/runs`, `/api/peak_vs_seq`, `/api/peak_vs_positions`,
    `/api/throughput`); anything else, including a traversal-shaped path, is a
    clean 404 -- there is no file-serving route here for a traversal to reach in
    the first place.

Endpoints (all GET, all read-only):
    /                       -- the HTML dashboard
    /api/runs               -- {"runs": [...]}            dim_run rows (run list + config)
    /api/peak_vs_seq         -- {"points": [...]}          v_peak_vs_seq rows, optional ?run_id=
    /api/peak_vs_positions   -- {"points": [...]}          v_peak_vs_positions rows, optional ?run_id=
    /api/throughput          -- {"rows": [...]}            v_throughput rows
    anything else            -- 404 (never a raw traceback / 500)
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "out" / "fit_metrics.duckdb"


# --- pure, importable, unit-testable functions -------------------------------------------------

def open_ro(
    db_path: Path, *, attempts: int = 4, delay: float = 0.25
) -> duckdb.DuckDBPyConnection | None:
    """Open a FRESH read-only connection to `db_path`, or None if the file doesn't exist yet (the
    fit hasn't written a store; `out/fit_metrics.duckdb` fills in overnight). Never creates the
    file -- checking existence first avoids duckdb's own "file not found" noise, and read_only=True
    means this call can never write even if the check raced a create.

    Retries a short, bounded number of times on lock-shaped `duckdb.Error`s -- an ingest holds a
    brief exclusive write lock, and a poll landing in that window should retry rather than flicker
    to an empty page. A non-lock error, or exhausting all retries, still degrades to None -- this
    tool must never raise."""
    if not db_path.exists():
        return None
    for attempt in range(attempts):
        try:
            return duckdb.connect(str(db_path), read_only=True)
        except duckdb.Error as e:
            msg = str(e).lower()
            if "lock" not in msg and "being used by another" not in msg:
                return None
            if attempt == attempts - 1:
                return None
            time.sleep(delay)
    return None


def rows_as_dicts(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> list[dict]:
    """Run `sql` and return rows as column-name-keyed dicts. Column names come from the cursor
    description rather than a hardcoded list, so this tolerates schema growth in fit_metrics.py
    without a code change. Any failure (missing table/view -- an empty or half-built store) degrades
    to an empty list; this tool must never 500 on partial or absent data."""
    try:
        cur = con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        return []


def fetch_runs(con: duckdb.DuckDBPyConnection | None) -> list[dict]:
    if con is None:
        return []
    return rows_as_dicts(con, "SELECT * FROM dim_run ORDER BY inserted_at")


def fetch_peak_vs_seq(con: duckdb.DuckDBPyConnection | None, run_id: str | None = None) -> list[dict]:
    if con is None:
        return []
    if run_id:
        return rows_as_dicts(
            con, "SELECT * FROM v_peak_vs_seq WHERE run_id = ? ORDER BY seq_len", [run_id])
    return rows_as_dicts(con, "SELECT * FROM v_peak_vs_seq ORDER BY seq_len")


def fetch_peak_vs_positions(
    con: duckdb.DuckDBPyConnection | None, run_id: str | None = None
) -> list[dict]:
    if con is None:
        return []
    if run_id:
        return rows_as_dicts(
            con, "SELECT * FROM v_peak_vs_positions WHERE run_id = ? ORDER BY n_positions",
            [run_id])
    return rows_as_dicts(con, "SELECT * FROM v_peak_vs_positions ORDER BY n_positions")


def fetch_throughput(con: duckdb.DuckDBPyConnection | None) -> list[dict]:
    if con is None:
        return []
    return rows_as_dicts(con, "SELECT * FROM v_throughput ORDER BY chunk_size")


# --- HTTP handler --------------------------------------------------------------------------------

def build_handler(db_path: Path) -> type[BaseHTTPRequestHandler]:
    """Bind a request handler class to a specific DuckDB path via closure, so nothing depends on
    mutable module-level globals (mirrors fit_monitor.py's build_handler)."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "fit-metrics-ui/1.0"

        def log_message(self, fmt, *args):  # noqa: A003 -- stdlib signature
            pass  # keep the terminal quiet; this is a local dev tool

        def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj, status: int = 200) -> None:
            # default=str: TIMESTAMP columns come back as datetime objects; str() is enough for a
            # read-only display tool (no round-tripping back into a query).
            self._send_bytes(
                json.dumps(obj, default=str).encode("utf-8"),
                "application/json; charset=utf-8", status)

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

            if path == "/api/runs":
                con = open_ro(db_path)
                try:
                    self._send_json({"runs": fetch_runs(con)})
                finally:
                    if con is not None:
                        con.close()
                return

            if path == "/api/peak_vs_seq":
                run_id = (query.get("run_id") or [None])[0]
                con = open_ro(db_path)
                try:
                    self._send_json({"points": fetch_peak_vs_seq(con, run_id)})
                finally:
                    if con is not None:
                        con.close()
                return

            if path == "/api/peak_vs_positions":
                run_id = (query.get("run_id") or [None])[0]
                con = open_ro(db_path)
                try:
                    self._send_json({"points": fetch_peak_vs_positions(con, run_id)})
                finally:
                    if con is not None:
                        con.close()
                return

            if path == "/api/throughput":
                con = open_ro(db_path)
                try:
                    self._send_json({"rows": fetch_throughput(con)})
                finally:
                    if con is not None:
                        con.close()
                return

            # Exact allowlist above; everything else (including a traversal-shaped path -- there
            # is no filesystem-serving route here for one to reach) is a clean 404.
            self._send_404()

    return Handler


# --- inline dashboard (static -- all data comes from fetch(), never templated) ------------------
# Visual language matches apps/heylook-frontend-v3/DESIGN.md + css/app.css: same OKLCH token
# values (warm-white writing surface, honey-bronze accent, single theme -- v3 has no dark mode,
# so neither does this), same data-strength chip formula (fixed L/C, hue carries data), same
# --brand-tint "chosen item" fill for the runs-table selection grammar, same tabular-nums-for-
# telemetry rule. Chart series color (chunk 128 vs 64) is a SEPARATE categorical pairing -- the
# strength ramp is reserved for scalar/ordered data per DESIGN.md #2, not appropriate for two
# unordered category buckets -- chosen for colorblind-safety (Okabe-Ito-style blue/bronze split)
# and reinforced with distinct marker shapes so color is never the only channel.

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>jlens-mlx fit metrics</title>
<style>
:root {
  --bg: oklch(1 0 0);
  --surface: oklch(0.974 0.007 91);
  --surface-2: oklch(0.948 0.010 91);
  --ink: oklch(0.28 0.02 78);
  --ink-muted: oklch(0.49 0.02 82);
  --ink-faint: oklch(0.60 0.018 85);
  --line: oklch(0.905 0.012 88);
  --line-strong: oklch(0.83 0.015 88);

  --brand: oklch(0.842 0.165 91.3);
  --brand-tint: oklch(0.962 0.05 95);
  --accent: oklch(0.47 0.10 78);
  --accent-hover: oklch(0.40 0.10 78);
  --on-accent: oklch(1 0 0);
  --danger: oklch(0.50 0.18 29);
  --danger-tint: oklch(0.962 0.03 29);
  --ok: oklch(0.55 0.15 145);
  --warn: oklch(0.65 0.16 70);

  /* chart-only categorical pair: chunk 128 (bronze, reuses --accent) vs chunk 64 (blue) --
     Okabe-Ito-derived, colorblind-safe, distinct from the scalar strength ramp on purpose. */
  --series-a: var(--accent);
  --series-b: oklch(0.55 0.14 250);
  --series-fallback: oklch(0.55 0.02 78);

  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, monospace;

  --text-sm: 0.8125rem;
  --text-ui: 0.875rem;
  --text-body: 1rem;
  --text-lg: 1.1875rem;

  --r-ctl: 6px;
  --r-card: 10px;
  --r-big: 14px;

  --ease: cubic-bezier(0.25, 0.1, 0.25, 1);
  --t-fast: 140ms;
}

* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  font-family: var(--font); font-size: var(--text-body); line-height: 1.55;
  color: var(--ink); background: var(--bg); -webkit-font-smoothing: antialiased;
}
h1, h2, h3 { font-weight: 600; margin: 0; }
::selection { background: var(--brand-tint); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.tabular { font-variant-numeric: tabular-nums; font-family: var(--mono); }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important; animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

header {
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  padding: 1rem 1.5rem; border-bottom: 1px solid var(--line); background: var(--surface);
}
header h1 { font-size: var(--text-lg); letter-spacing: -0.01em; }
header h1 .accent { color: var(--accent); }
header .sub { font-size: var(--text-sm); color: var(--ink-muted); margin-top: 2px; }
#refresh {
  font: inherit; font-size: var(--text-ui); background: var(--accent); color: var(--on-accent);
  border: none; border-radius: var(--r-ctl); padding: 0.5rem 0.9rem; cursor: pointer;
  transition: background var(--t-fast) var(--ease);
}
#refresh:hover { background: var(--accent-hover); }
#last-updated { font-size: var(--text-sm); color: var(--ink-muted); margin-top: 0.35rem; }

main { max-width: 1100px; margin: 0 auto; padding: 1.5rem; }

.empty {
  text-align: center; color: var(--ink-muted); padding: 3rem 1rem; font-size: var(--text-ui);
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-card);
}
.empty strong { color: var(--ink); }
.hidden { display: none !important; }

.stat-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem;
  margin-bottom: 1.5rem;
}
.stat {
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-card);
  padding: 0.85rem 1rem;
}
.stat-label {
  font-size: var(--text-sm); color: var(--ink-muted); text-transform: uppercase;
  letter-spacing: 0.03em;
}
.stat-value { font-size: 1.375rem; font-weight: 650; margin-top: 0.2rem; }
.stat-value.tabular { font-weight: 600; }

section.panel {
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-card);
  padding: 1.1rem 1.25rem; margin-bottom: 1.5rem;
}
.panel h2 {
  font-size: var(--text-ui); text-transform: uppercase; letter-spacing: 0.03em;
  color: var(--ink-muted); margin-bottom: 0.75rem;
}
.panel .caption { font-size: var(--text-sm); color: var(--ink-muted); margin-top: 0.6rem; }

.filter-note {
  font-size: var(--text-sm); color: var(--ink-muted); display: flex; align-items: center;
  gap: 0.5rem; margin-bottom: 0.6rem; flex-wrap: wrap;
}
#clear-filter {
  font: inherit; font-size: var(--text-sm); background: none; color: var(--accent);
  border: 1px solid var(--line-strong); border-radius: var(--r-ctl); padding: 0.15rem 0.5rem;
  cursor: pointer;
}
#clear-filter:hover { background: var(--surface-2); }

/* ---- chart ---- */
.chart-wrap { overflow-x: auto; }
svg.chart { width: 100%; min-width: 480px; height: auto; display: block; }
.chart .grid line { stroke: var(--line); stroke-width: 1; }
.chart .axis text {
  fill: var(--ink-muted); font-family: var(--mono); font-size: 10.5px;
  font-variant-numeric: tabular-nums;
}
.chart .axis-title { fill: var(--ink-muted); font-size: 11px; font-family: var(--font); }
.chart .ceiling-line { stroke: var(--danger); stroke-width: 1.5; stroke-dasharray: 5 4; }
.chart .ceiling-label { fill: var(--danger); font-size: 10.5px; font-family: var(--mono); font-weight: 600; }
.chart .danger-band { fill: color-mix(in oklch, var(--danger) 10%, transparent); }
.chart .pt { cursor: pointer; }
.chart .pt:hover, .chart .pt:focus-visible { outline: none; }
.chart .pt:hover .mark, .chart .pt:focus-visible .mark {
  transform: scale(1.35); transition: transform var(--t-fast) var(--ease);
}
.chart .mark { transition: transform var(--t-fast) var(--ease); }
.chart .pt:focus-visible .mark { stroke: var(--ink); stroke-width: 2; }

.legend { display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 0.6rem; font-size: var(--text-sm); }
.legend-item { display: flex; align-items: center; gap: 0.4rem; color: var(--ink-muted); }
.legend-swatch { width: 12px; height: 12px; flex: none; }
.legend-line { width: 18px; height: 0; border-top: 2px dashed var(--danger); flex: none; }

#tooltip {
  position: fixed; pointer-events: none; z-index: 30; background: var(--ink); color: var(--bg);
  font-family: var(--mono); font-size: 11.5px; padding: 0.4rem 0.55rem; border-radius: 6px;
  line-height: 1.4; max-width: 240px; opacity: 0; transition: opacity var(--t-fast) var(--ease);
}
#tooltip.show { opacity: 1; }
#tooltip .row { white-space: nowrap; }

/* ---- throughput table ---- */
table.throughput { width: 100%; border-collapse: collapse; font-size: var(--text-ui); }
table.throughput th {
  text-align: left; font-size: var(--text-sm); color: var(--ink-muted); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.02em; padding: 0.3rem 0.6rem;
  border-bottom: 1px solid var(--line);
}
table.throughput th.num, table.throughput td.num { text-align: right; }
table.throughput td { padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--line); }
table.throughput tr:last-child td { border-bottom: none; }
.bar-cell { display: flex; align-items: center; gap: 0.5rem; justify-content: flex-end; }
.bar-track {
  flex: 1 1 auto; max-width: 140px; height: 8px; border-radius: 4px; background: var(--surface-2);
  overflow: hidden;
}
.bar-fill { height: 100%; background: color-mix(in oklch, var(--brand) 65%, var(--accent) 35%); }

/* ---- chip system (DESIGN.md #2: fixed L/C, hue carries data) ---- */
.chip {
  display: inline-flex; align-items: center; font-family: var(--mono); font-size: var(--text-sm);
  padding: 0.1rem 0.45rem; border-radius: var(--r-ctl); border: 1px solid var(--line-strong);
  color: var(--ink-muted); white-space: nowrap;
}
.chip.diversity { border: none; color: #1a1a1a; font-weight: 600; }
.chip.diversity.ok { background: oklch(0.86 0.11 145); }
.chip.diversity.warn { background: oklch(0.86 0.11 85); }
.chip.diversity.bad { background: oklch(0.86 0.11 25); }
.chip.series { border: none; color: #1a1a1a; font-weight: 600; }

/* ---- runs table ---- */
.runs-list { display: flex; flex-direction: column; gap: 0.5rem; }
.run-row {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 0.6rem;
  border: 1px solid var(--line); border-radius: var(--r-ctl); padding: 0.6rem 0.75rem;
  cursor: pointer; transition: background var(--t-fast) var(--ease);
}
.run-row:hover { background: var(--surface-2); }
.run-row[aria-pressed="true"] { background: var(--brand-tint); border-color: var(--brand); }
.run-row .run-id { font-family: var(--mono); font-size: var(--text-sm); color: var(--ink); font-weight: 600; }
.run-row .run-meta { font-size: var(--text-sm); color: var(--ink-muted); }
.run-row .chips { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.run-row .metrics {
  margin-left: auto; display: flex; gap: 1rem; font-size: var(--text-sm); color: var(--ink-muted);
}
.run-row .metrics .num { color: var(--ink); font-weight: 600; }
.run-row .run-head { display: flex; flex-direction: column; min-width: 140px; }

footer { text-align: center; font-size: var(--text-sm); color: var(--ink-faint); padding: 1.5rem; }
</style>
</head>
<body>
<header>
  <div>
    <h1>jlens-mlx <span class="accent">fit metrics</span></h1>
    <div class="sub">read-only cross-run analytics over the fit-metrics DuckDB store</div>
    <div id="last-updated" role="status"></div>
  </div>
  <button id="refresh">Refresh</button>
</header>

<main>
  <div id="global-empty" class="empty hidden">
    <strong>Waiting for completed items.</strong><br>
    The fit writes one row per item as it finishes -- this page will fill in as runs complete.
  </div>

  <div id="content" class="hidden">
    <div class="stat-grid" id="stat-grid"></div>

    <section class="panel">
      <h2>Peak memory vs. fitted positions</h2>
      <div id="peak-empty" class="empty hidden">No item-fit data yet for this selection.</div>
      <div id="peak-content">
        <div class="chart-wrap"><svg id="peak-chart" class="chart" viewBox="0 0 780 420" role="img" aria-label="Scatter plot of peak GPU memory versus fitted positions, split by chunk size, with a 192 GB unified-memory ceiling line. Sequence length is also shown per point on hover."></svg></div>
        <div class="legend" id="peak-legend"></div>
      </div>
    </section>

    <section class="panel">
      <h2>Throughput by chunk size</h2>
      <div id="throughput-empty" class="empty hidden">No throughput data yet.</div>
      <table class="throughput hidden" id="throughput-table">
        <thead>
          <tr>
            <th>Chunk</th><th class="num">Items</th><th class="num">Sec / position</th>
            <th class="num">Sec / chunk</th><th class="num">Avg peak GB</th>
          </tr>
        </thead>
        <tbody id="throughput-body"></tbody>
      </table>
      <div class="caption" id="throughput-caption"></div>
    </section>

    <section class="panel">
      <h2>Runs</h2>
      <div class="filter-note">
        <span>Click a run to filter the chart above.</span>
        <button id="clear-filter" class="hidden">Clear filter &times;</button>
      </div>
      <div class="runs-list" id="runs-list"></div>
    </section>
  </div>
</main>

<footer>jlens-mlx &middot; read-only &middot; never writes to the store</footer>
<div id="tooltip" role="presentation"></div>

<script>
'use strict';

const state = { runs: [], points: [], throughput: [], selectedRun: null };

function esc(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

async function fetchJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(url + ': ' + r.status);
  return r.json();
}

function sharedFractionClass(v) {
  // Mirrors fit_monitor.py's sharedFractionClass AND fit_band_corpus.py's gate thresholds:
  // <=0.35 ok, (0.35, 0.5] warn, >0.5 bad.
  if (v === null || v === undefined) return '';
  if (v <= 0.35) return 'ok';
  if (v <= 0.5) return 'warn';
  return 'bad';
}

// chunk_size -> {color, shape, label}; 128/64 are the expected pair, anything else falls back
// to a neutral color + diamond so an unexpected chunk size still renders (never breaks).
const SERIES = {
  128: { color: 'var(--series-a)', shape: 'circle', label: 'chunk 128' },
  64: { color: 'var(--series-b)', shape: 'triangle', label: 'chunk 64' },
};
function seriesFor(chunkSize) {
  return SERIES[chunkSize] || {
    color: 'var(--series-fallback)', shape: 'diamond', label: 'chunk ' + chunkSize,
  };
}

function fmtNum(v, digits) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}

// ---- stat tiles --------------------------------------------------------------------------

function renderStats() {
  const grid = document.getElementById('stat-grid');
  const totalRuns = state.runs.length;
  const totalItems = state.points.length;
  const maxPeak = state.points.reduce((m, p) => Math.max(m, p.peak_gb ?? 0), 0);
  let bestCfg = '—';
  if (state.throughput.length) {
    const best = state.throughput.reduce((a, b) =>
      (a.avg_sec_per_pos ?? Infinity) <= (b.avg_sec_per_pos ?? Infinity) ? a : b);
    bestCfg = 'chunk ' + best.chunk_size + ' · ' + fmtNum(best.avg_sec_per_pos, 2) + ' s/pos';
  }
  const tiles = [
    ['Runs', totalRuns],
    ['Items fit', totalItems],
    ['Max peak observed', maxPeak ? fmtNum(maxPeak, 1) + ' GB' : '—'],
    ['Best throughput config', bestCfg],
  ];
  grid.innerHTML = tiles.map(function (t) {
    return '<div class="stat"><div class="stat-label">' + esc(t[0]) + '</div>' +
      '<div class="stat-value tabular">' + esc(t[1]) + '</div></div>';
  }).join('');
}

// ---- hero chart: peak_gb vs n_positions ----------------------------------------------------
// Measured data shows peak memory clusters by FITTED POSITIONS, not sequence length (flat
// peak_gb across seq_len 72->78 at a fixed position count) -- positions is the x-axis here;
// seq_len rides along per point (still visible in the tooltip) so a residual can be eyeballed.

function niceMax(v, floor) {
  return Math.max(floor, v);
}

function markerPath(shape, size) {
  if (shape === 'triangle') {
    const h = size * 1.1;
    return 'M 0 ' + (-h * 0.62) + ' L ' + (h * 0.62) + ' ' + (h * 0.5) + ' L ' + (-h * 0.62) + ' ' + (h * 0.5) + ' Z';
  }
  if (shape === 'diamond') {
    return 'M 0 ' + (-size) + ' L ' + size + ' 0 L 0 ' + size + ' L ' + (-size) + ' 0 Z';
  }
  return null; // circle drawn as <circle>, not a path
}

function renderPeakChart() {
  const svg = document.getElementById('peak-chart');
  const emptyEl = document.getElementById('peak-empty');
  const contentEl = document.getElementById('peak-content');
  const legendEl = document.getElementById('peak-legend');

  const points = state.selectedRun
    ? state.points.filter(function (p) { return p.run_id === state.selectedRun; })
    : state.points;

  if (!points.length) {
    emptyEl.classList.remove('hidden');
    contentEl.classList.add('hidden');
    return;
  }
  emptyEl.classList.add('hidden');
  contentEl.classList.remove('hidden');

  const W = 780, H = 420, padL = 58, padR = 20, padT = 18, padB = 46;
  const innerW = W - padL - padR, innerH = H - padT - padB;

  const posCounts = points.map(function (p) { return p.n_positions; });
  const peaks = points.map(function (p) { return p.peak_gb; });
  const posMax = Math.max.apply(null, posCounts);
  const posMin = Math.min.apply(null, posCounts);
  const peakMax = Math.max.apply(null, peaks.concat([192])); // ceiling always in view

  // Domain padding so 1-2 points (or all-identical n_positions) never sit flush on an axis or
  // collapse to a zero-width domain.
  const xSpan = Math.max(posMax - posMin, Math.max(posMax * 0.15, 32));
  const xMin = Math.max(0, posMin - xSpan * 0.15);
  const xMax = posMax + xSpan * 0.25;
  const yMax = niceMax(peakMax * 1.12, 40);
  const yMin = 0;

  function x(v) { return padL + (v - xMin) / (xMax - xMin) * innerW; }
  function y(v) { return padT + innerH - (v - yMin) / (yMax - yMin) * innerH; }

  const xTicks = 6, yTicks = 6;
  let grid = '';
  const xTickVals = [];
  for (let i = 0; i <= xTicks; i++) xTickVals.push(xMin + (xMax - xMin) * i / xTicks);
  const yTickVals = [];
  for (let i = 0; i <= yTicks; i++) yTickVals.push(yMin + (yMax - yMin) * i / yTicks);

  for (const v of yTickVals) {
    grid += '<line class="grid" x1="' + padL + '" x2="' + (padL + innerW) + '" y1="' + y(v) + '" y2="' + y(v) + '" />';
  }
  for (const v of xTickVals) {
    grid += '<line class="grid" x1="' + x(v) + '" x2="' + x(v) + '" y1="' + padT + '" y2="' + (padT + innerH) + '" />';
  }

  let axis = '<g class="axis">';
  for (const v of yTickVals) {
    axis += '<text x="' + (padL - 8) + '" y="' + (y(v) + 3) + '" text-anchor="end">' + Math.round(v) + '</text>';
  }
  for (const v of xTickVals) {
    axis += '<text x="' + x(v) + '" y="' + (padT + innerH + 16) + '" text-anchor="middle">' + Math.round(v) + '</text>';
  }
  axis += '</g>';
  axis += '<text class="axis-title" x="' + (padL + innerW / 2) + '" y="' + (H - 6) + '" text-anchor="middle">fitted positions</text>';
  axis += '<text class="axis-title" x="14" y="' + (padT + innerH / 2) + '" text-anchor="middle" transform="rotate(-90 14 ' + (padT + innerH / 2) + ')">peak memory (GB)</text>';

  // Danger band: from 90% of the ceiling up to the top of the chart -- the "approaching the
  // wall" zone -- plus the ceiling line itself, always the most emphasized mark on the chart.
  let ceiling = '';
  const bandTop = 192 * 0.9;
  if (bandTop < yMax) {
    ceiling += '<rect class="danger-band" x="' + padL + '" y="' + y(yMax) + '" width="' + innerW + '" height="' + (y(bandTop) - y(yMax)) + '" />';
  }
  if (192 <= yMax) {
    ceiling += '<line class="ceiling-line" x1="' + padL + '" x2="' + (padL + innerW) + '" y1="' + y(192) + '" y2="' + y(192) + '" />';
    ceiling += '<text class="ceiling-label" x="' + (padL + innerW - 6) + '" y="' + (y(192) - 6) + '" text-anchor="end">192 GB unified-memory ceiling</text>';
  }

  let marks = '';
  const seriesSeen = {};
  for (const p of points) {
    const s = seriesFor(p.chunk_size);
    seriesSeen[p.chunk_size] = s;
    const cx = x(p.n_positions), cy = y(p.peak_gb);
    const title = 'item #' + p.item_index + ' · ' + p.n_positions + ' pos (seq ' + p.seq_len +
      ' tok) · peak ' + fmtNum(p.peak_gb, 1) + ' GB · chunk ' + p.chunk_size + ' · ' +
      esc(p.stratum) + ' · ' + (p.on_policy ? 'on-policy' : 'off-policy');
    const tipData = JSON.stringify({
      item_index: p.item_index, n_positions: p.n_positions, seq_len: p.seq_len,
      peak_gb: p.peak_gb, chunk_size: p.chunk_size, stratum: p.stratum, on_policy: p.on_policy,
    }).replace(/"/g, '&quot;');
    marks += '<g class="pt" tabindex="0" role="img" aria-label="' + esc(title) +
      '" data-tip="' + tipData + '" transform="translate(' + cx + ',' + cy + ')">' +
      '<title>' + esc(title) + '</title>';
    if (s.shape === 'circle') {
      marks += '<circle class="mark" r="5" fill="' + s.color + '" fill-opacity="0.85" stroke="' + s.color + '" />';
    } else {
      marks += '<path class="mark" d="' + markerPath(s.shape, 6.5) + '" fill="' + s.color + '" fill-opacity="0.85" stroke="' + s.color + '" />';
    }
    marks += '</g>';
  }

  svg.innerHTML = '<g class="grid">' + grid + '</g>' + ceiling + marks + axis;

  // legend: one entry per series actually present, plus the ceiling line.
  const legendItems = Object.keys(seriesSeen).sort().map(function (k) {
    const s = seriesSeen[k];
    const swatch = s.shape === 'circle'
      ? '<span class="legend-swatch" style="background:' + s.color + ';border-radius:50%"></span>'
      : '<span class="legend-swatch" style="background:' + s.color + '"></span>';
    return '<span class="legend-item">' + swatch + esc(s.label) + '</span>';
  });
  legendItems.push('<span class="legend-item"><span class="legend-line"></span>192 GB ceiling</span>');
  legendEl.innerHTML = legendItems.join('');

  // tooltip wiring
  const tooltip = document.getElementById('tooltip');
  function showTip(el, evt) {
    const d = JSON.parse(el.getAttribute('data-tip').replace(/&quot;/g, '"'));
    tooltip.innerHTML =
      '<div class="row">item #' + d.item_index + ' · ' + esc(d.stratum) + '</div>' +
      '<div class="row">' + d.n_positions + ' pos · seq ' + d.seq_len + ' tok · chunk ' + d.chunk_size + '</div>' +
      '<div class="row">peak ' + fmtNum(d.peak_gb, 1) + ' GB · ' + (d.on_policy ? 'on-policy' : 'off-policy') + '</div>';
    tooltip.classList.add('show');
    positionTip(evt);
  }
  function positionTip(evt) {
    const x0 = (evt && evt.clientX) || 0, y0 = (evt && evt.clientY) || 0;
    tooltip.style.left = (x0 + 14) + 'px';
    tooltip.style.top = (y0 + 14) + 'px';
  }
  function hideTip() { tooltip.classList.remove('show'); }

  svg.querySelectorAll('.pt').forEach(function (el) {
    el.addEventListener('mouseenter', function (e) { showTip(el, e); });
    el.addEventListener('mousemove', positionTip);
    el.addEventListener('mouseleave', hideTip);
    el.addEventListener('focus', function () {
      const r = el.getBoundingClientRect();
      showTip(el, { clientX: r.left, clientY: r.top });
    });
    el.addEventListener('blur', hideTip);
  });
}

// ---- throughput panel ---------------------------------------------------------------------

function renderThroughput() {
  const emptyEl = document.getElementById('throughput-empty');
  const table = document.getElementById('throughput-table');
  const body = document.getElementById('throughput-body');
  const caption = document.getElementById('throughput-caption');
  const rows = state.throughput;

  if (!rows.length) {
    emptyEl.classList.remove('hidden');
    table.classList.add('hidden');
    caption.textContent = '';
    return;
  }
  emptyEl.classList.add('hidden');
  table.classList.remove('hidden');

  const maxSecPos = Math.max.apply(null, rows.map(function (r) { return r.avg_sec_per_pos || 0; }));
  body.innerHTML = rows.map(function (r) {
    const s = seriesFor(r.chunk_size);
    const pct = maxSecPos ? Math.max(4, 100 * (r.avg_sec_per_pos || 0) / maxSecPos) : 0;
    return '<tr>' +
      '<td><span class="chip series" style="background:' + s.color + '">' + esc(s.label) + '</span></td>' +
      '<td class="num tabular">' + (r.n ?? '—') + '</td>' +
      '<td class="num"><div class="bar-cell"><span class="tabular">' + fmtNum(r.avg_sec_per_pos, 3) + '</span>' +
        '<span class="bar-track"><span class="bar-fill" style="width:' + pct + '%"></span></span></div></td>' +
      '<td class="num tabular">' + fmtNum(r.avg_sec_per_chunk, 2) + '</td>' +
      '<td class="num tabular">' + fmtNum(r.avg_peak_gb, 1) + '</td>' +
      '</tr>';
  }).join('');

  if (rows.length === 2) {
    const [a, b] = [...rows].sort(function (r1, r2) { return r2.chunk_size - r1.chunk_size; });
    const speedDelta = a.avg_sec_per_pos ? 100 * (b.avg_sec_per_pos - a.avg_sec_per_pos) / a.avg_sec_per_pos : null;
    const peakDelta = a.avg_peak_gb ? 100 * (a.avg_peak_gb - b.avg_peak_gb) / a.avg_peak_gb : null;
    if (speedDelta !== null && peakDelta !== null) {
      caption.textContent = 'chunk ' + b.chunk_size + ' vs. chunk ' + a.chunk_size + ': ' +
        (Math.abs(speedDelta) < 8 ? 'about the same speed per position' :
          (speedDelta < 0 ? Math.abs(speedDelta).toFixed(0) + '% faster per position' : speedDelta.toFixed(0) + '% slower per position')) +
        ', ' + (peakDelta > 0 ? peakDelta.toFixed(0) + '% less peak memory' : Math.abs(peakDelta).toFixed(0) + '% more peak memory') + '.';
    } else {
      caption.textContent = '';
    }
  } else {
    caption.textContent = '';
  }
}

// ---- runs table -----------------------------------------------------------------------------

function chunkSizesForRun(runId) {
  const seen = new Set();
  for (const p of state.points) if (p.run_id === runId) seen.add(p.chunk_size);
  return [...seen].sort(function (a, b) { return a - b; });
}

function statsForRun(runId) {
  let n = 0, peak = 0;
  for (const p of state.points) {
    if (p.run_id !== runId) continue;
    n += 1;
    if ((p.peak_gb || 0) > peak) peak = p.peak_gb;
  }
  return { n, peak };
}

function renderRuns() {
  const list = document.getElementById('runs-list');
  if (!state.runs.length) {
    list.innerHTML = '<div class="empty">No runs recorded yet.</div>';
    return;
  }
  list.innerHTML = state.runs.map(function (run) {
    const chunks = chunkSizesForRun(run.run_id);
    const rs = statsForRun(run.run_id);
    const selected = state.selectedRun === run.run_id;
    const chips = [];
    if (run.band_start !== undefined && run.band_end !== undefined) {
      chips.push('<span class="chip">band [' + run.band_start + ', ' + run.band_end + ')</span>');
    }
    if (run.target !== undefined && run.target !== null) chips.push('<span class="chip">target ' + run.target + '</span>');
    if (chunks.length) chips.push('<span class="chip">' + chunks.map(function (c) { return 'chunk ' + c; }).join(', ') + '</span>');
    chips.push('<span class="chip">thinking ' + (run.enable_thinking ? 'on' : 'off') + '</span>');
    if (run.use_chain !== undefined && run.use_chain !== null) chips.push('<span class="chip">' + (run.use_chain ? 'chained' : 'unchained') + '</span>');
    if (run.shared_fraction_overall !== undefined && run.shared_fraction_overall !== null) {
      chips.push('<span class="chip diversity ' + sharedFractionClass(run.shared_fraction_overall) + '" title="corpus diversity gate: overall shared_fraction">sf ' + fmtNum(run.shared_fraction_overall, 2) + '</span>');
    }
    if (run.shared_fraction_onpolicy !== undefined && run.shared_fraction_onpolicy !== null) {
      chips.push('<span class="chip diversity ' + sharedFractionClass(run.shared_fraction_onpolicy) + '" title="corpus diversity gate: on-policy shared_fraction">sf(on-policy) ' + fmtNum(run.shared_fraction_onpolicy, 2) + '</span>');
    }
    if (run.dropped_over_len) chips.push('<span class="chip">dropped_over_len ' + run.dropped_over_len + '</span>');

    return '<div class="run-row" role="button" tabindex="0" aria-pressed="' + selected + '" data-run="' + esc(run.run_id) + '">' +
      '<div class="run-head"><span class="run-id">' + esc(run.run_id) + '</span>' +
        '<span class="run-meta">' + esc(run.model || '—') + (run.git_sha ? ' · ' + esc(String(run.git_sha).slice(0, 8)) : '') + '</span></div>' +
      '<div class="chips">' + chips.join('') + '</div>' +
      '<div class="metrics">' +
        '<span>items <span class="num tabular">' + rs.n + '</span></span>' +
        '<span>peak <span class="num tabular">' + (rs.peak ? fmtNum(rs.peak, 1) + ' GB' : '—') + '</span></span>' +
      '</div>' +
    '</div>';
  }).join('');

  list.querySelectorAll('.run-row').forEach(function (el) {
    function toggle() {
      const id = el.getAttribute('data-run');
      state.selectedRun = state.selectedRun === id ? null : id;
      renderAll();
    }
    el.addEventListener('click', toggle);
    el.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
    });
  });

  const clearBtn = document.getElementById('clear-filter');
  clearBtn.classList.toggle('hidden', !state.selectedRun);
}

document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape' && state.selectedRun) {
    state.selectedRun = null;
    renderAll();
  }
});
document.getElementById('clear-filter').addEventListener('click', function () {
  state.selectedRun = null;
  renderAll();
});

// ---- orchestration --------------------------------------------------------------------------

function renderAll() {
  const globalEmpty = document.getElementById('global-empty');
  const content = document.getElementById('content');
  if (!state.runs.length && !state.points.length) {
    globalEmpty.classList.remove('hidden');
    content.classList.add('hidden');
    return;
  }
  globalEmpty.classList.add('hidden');
  content.classList.remove('hidden');
  renderStats();
  renderPeakChart();
  renderThroughput();
  renderRuns();
}

async function loadAll() {
  // Hero chart is charted against /api/peak_vs_positions (fitted positions drive peak memory,
  // not sequence length) -- that view is a superset of /api/peak_vs_seq's columns plus
  // n_positions, so state.points works unchanged for the runs table / stat tiles below.
  // /api/peak_vs_seq stays available server-side for direct callers; this page just doesn't
  // chart it as the default anymore.
  const [runsRes, pointsRes, throughputRes] = await Promise.all([
    fetchJSON('/api/runs').catch(function () { return { runs: [] }; }),
    fetchJSON('/api/peak_vs_positions').catch(function () { return { points: [] }; }),
    fetchJSON('/api/throughput').catch(function () { return { rows: [] }; }),
  ]);
  state.runs = runsRes.runs || [];
  state.points = pointsRes.points || [];
  state.throughput = throughputRes.rows || [];
  renderAll();
  document.getElementById('last-updated').textContent = 'updated ' + new Date().toLocaleTimeString();
}

document.getElementById('refresh').addEventListener('click', loadAll);
loadAll();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only web dashboard for the jlens-mlx fit-metrics DuckDB store.")
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help=f"Path to the fit-metrics DuckDB file (default: {DEFAULT_DB}). "
                             "Opened read-only; never created or written to.")
    parser.add_argument("--port", type=int, default=8766,
                        help="Port to bind (default 8766; 0 picks an ephemeral port).")
    args = parser.parse_args(argv)

    db_path = Path(args.db).resolve()
    handler_cls = build_handler(db_path)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_cls)
    print(f"http://127.0.0.1:{server.server_port}/  (db: {db_path})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
