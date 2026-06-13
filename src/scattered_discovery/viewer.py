from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from scattered_discovery.config import WorldConfig
from scattered_discovery.envs.scattered_causal import ScatteredDiscoveryEnv
from scattered_discovery.envs.scattered_dsl import (
    Edge,
    Expr,
    PathExpr,
    canonical_key,
    edge_keys_for_path,
    format_expr,
    variables_in_expr,
)


@dataclass
class ViewerSession:
    env: ScatteredDiscoveryEnv
    world_seed: int
    episode_seed: int
    dispersion: float
    transcript: list[dict[str, Any]]
    last_step: dict[str, Any] | None = None


SESSIONS: dict[str, ViewerSession] = {}


def _world_config_from_payload(payload: dict[str, Any]) -> WorldConfig:
    allowed = {field.name for field in fields(WorldConfig)}
    values: dict[str, Any] = {}
    for key, value in payload.items():
        if key in allowed and value not in (None, ""):
            values[key] = value
    int_fields = {
        "num_branches",
        "branch_depth",
        "distractors_per_node",
        "base_budget",
        "test_cost",
        "intervene_cost",
        "invalid_action_cost",
    }
    for key in int_fields & values.keys():
        values[key] = int(values[key])
    float_fields = set(allowed) - int_fields
    for key in float_fields & values.keys():
        values[key] = float(values[key])
    return WorldConfig(**values)


def _new_session(payload: dict[str, Any]) -> tuple[str, ViewerSession]:
    config = _world_config_from_payload(payload)
    world_seed = int(payload.get("world_seed", 1))
    episode_seed = int(payload.get("episode_seed", world_seed * 1009 + 19))
    dispersion = float(payload.get("dispersion", 0.5))
    env = ScatteredDiscoveryEnv(
        config,
        world_seed=world_seed,
        episode_seed=episode_seed,
        dispersion=dispersion,
        protocol=str(payload.get("protocol", "single")),
        max_commit=int(payload.get("max_commit", 1)),
    )
    session = ViewerSession(
        env=env,
        world_seed=world_seed,
        episode_seed=episode_seed,
        dispersion=dispersion,
        transcript=[
            {
                "role": "environment",
                "content": "Reset environment.",
            }
        ],
    )
    session_id = uuid4().hex
    SESSIONS[session_id] = session
    return session_id, session


def _snapshot(session_id: str, session: ViewerSession) -> dict[str, Any]:
    env = session.env
    return {
        "session_id": session_id,
        "done": env.done,
        "world_seed": session.world_seed,
        "episode_seed": session.episode_seed,
        "dispersion": session.dispersion,
        "config": asdict(env.config),
        "budget_remaining": env.budget,
        "budget_initial": env.initial_budget,
        "known_variables": sorted(env.known_variables),
        "accepted_claims": sorted(env.accepted_claims),
        "rejected_claims": sorted(env.rejected_claims),
        "public_state": env.public_state_text(max_evidence_items=40),
        "world": _world_payload(env),
        "candidates": _candidate_payload(env),
        "last_step": session.last_step,
        "transcript": session.transcript,
    }


def _world_payload(env: ScatteredDiscoveryEnv) -> dict[str, Any]:
    return {
        "initial_variables": sorted(env.world.initial_variables),
        "target_count": env.target_count,
        "branches": [
            {
                "branch_id": branch.branch_id,
                "path": list(branch.path),
                "key": branch.terminal_key,
            }
            for branch in env.world.branches
        ],
        "true_edges": [
            {"src": src, "dst": dst} for src, dst in sorted(env.world.true_edges)
        ],
        "outgoing_candidates": [
            {
                "src": src,
                "dst": dst,
                "true_edge": (src, dst) in env.world.true_edges,
            }
            for src, targets in sorted(env.world.outgoing_candidates.items())
            for dst in targets
        ],
    }


def _candidate_payload(env: ScatteredDiscoveryEnv) -> list[dict[str, Any]]:
    candidates: dict[str, Expr] = {}
    for src, targets in env.world.outgoing_candidates.items():
        for dst in targets:
            edge = Edge(src, dst)
            candidates[canonical_key(edge)] = edge
            path = PathExpr((src, dst))
            candidates.setdefault(canonical_key(path), path)

    for branch in env.world.branches:
        for length in range(2, len(branch.path) + 1):
            path = PathExpr(branch.path[:length])
            candidates.setdefault(canonical_key(path), path)

    return [
        _candidate_entry(env, expr)
        for _, expr in sorted(candidates.items(), key=lambda item: item[0])
    ]


def _candidate_entry(env: ScatteredDiscoveryEnv, expr: Expr) -> dict[str, Any]:
    key = canonical_key(expr)
    info = env.world.classify(expr)
    summary = env.evidence.summary(key)
    supported = env._evidence_backed(expr, info)
    if not info.true:
        label = "false"
    elif not info.terminal:
        label = "true_non_final"
    elif supported:
        label = "valid_final_supported"
    else:
        label = "final_unsupported"

    edge_evidence: list[dict[str, Any]] = []
    if isinstance(expr, PathExpr):
        edge_evidence = [
            asdict(env.evidence.summary(edge_key))
            for edge_key in edge_keys_for_path(expr)
        ]

    variables = sorted(variables_in_expr(expr))
    return {
        "key": key,
        "expr": format_expr(expr),
        "kind": "path" if isinstance(expr, PathExpr) else "edge",
        "true": info.true,
        "final": info.terminal,
        "role": info.role,
        "label": label,
        "branch_ids": sorted(info.branch_ids),
        "variables": variables,
        "admissible_now": set(variables).issubset(env.known_variables),
        "supported_now": supported,
        "evidence": asdict(summary),
        "edge_evidence": edge_evidence,
    }


def _step(session: ViewerSession, action: str) -> None:
    result = session.env.step(action)
    score = asdict(result.score) if result.score is not None else None
    session.last_step = {
        "observation": result.observation,
        "done": result.done,
        "parse_ok": result.parse_ok,
        "action_text": result.action_text,
        "reward": result.score.reward if result.score is not None else None,
        "score": score,
        "debug": result.debug,
    }
    session.transcript.append({"role": "assistant", "content": action})
    session.transcript.append(
        {
            "role": "environment",
            "content": result.observation,
            "score": score,
        }
    )


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True).encode("utf-8")


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "ScatteredDiscoveryViewer/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_html(VIEWER_HTML)
            return
        self._send_json({"error": f"Unknown path: {path}"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/reset":
                session_id, session = _new_session(payload)
                self._send_json(_snapshot(session_id, session))
                return
            if path == "/api/step":
                session_id = str(payload.get("session_id", ""))
                action = str(payload.get("action", "")).strip()
                if not action:
                    self._send_json({"error": "Missing action."}, status=400)
                    return
                session = SESSIONS.get(session_id)
                if session is None:
                    self._send_json({"error": "Unknown session."}, status=404)
                    return
                _step(session, action)
                self._send_json(_snapshot(session_id, session))
                return
            self._send_json({"error": f"Unknown path: {path}"}, status=404)
        except Exception as exc:  # pragma: no cover - returned to browser.
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Scattered Discovery Viewer</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #15202b;
      --muted: #5d6978;
      --line: #d8dde6;
      --green: #157a5c;
      --green-soft: #e7f5ef;
      --blue: #265fbd;
      --blue-soft: #e9f0ff;
      --amber: #9a5b00;
      --amber-soft: #fff2d8;
      --red: #b42318;
      --red-soft: #fde8e5;
      --violet: #6750a4;
      --violet-soft: #eee9ff;
      --shadow: 0 1px 4px rgba(20, 27, 36, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.4 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, select {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
      min-height: 34px;
    }
    button:hover { border-color: #aeb7c5; }
    button.primary {
      background: #183a59;
      color: white;
      border-color: #183a59;
    }
    button.small {
      min-height: 28px;
      padding: 4px 7px;
      font-size: 12px;
    }
    input, select {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      padding: 7px 8px;
      min-width: 0;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .topbar {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 10px 14px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
    }
    .title {
      font-weight: 700;
      letter-spacing: 0;
      white-space: nowrap;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(10, minmax(76px, 1fr));
      gap: 8px;
      align-items: end;
    }
    .control label {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 3px;
    }
    .control input {
      width: 100%;
    }
    .main {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(460px, 1fr) minmax(360px, 480px);
      gap: 12px;
      padding: 12px;
      min-height: 0;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-height: 0;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 10px 12px;
      font-size: 13px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    .panel-body {
      padding: 10px 12px;
      overflow: auto;
      max-height: calc(100vh - 118px);
    }
    .stack { display: grid; gap: 10px; }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 8px;
      min-height: 52px;
    }
    .stat .label {
      color: var(--muted);
      font-size: 11px;
    }
    .stat .value {
      font-weight: 700;
      margin-top: 4px;
      overflow-wrap: anywhere;
    }
    .action-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
    }
    .button-row {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      font-size: 12px;
      max-height: 220px;
      overflow: auto;
    }
    .graph-wrap {
      position: relative;
      height: 500px;
      border-bottom: 1px solid var(--line);
      background:
        linear-gradient(#f0f3f7 1px, transparent 1px),
        linear-gradient(90deg, #f0f3f7 1px, transparent 1px),
        #fbfcfd;
      background-size: 28px 28px;
      overflow: hidden;
    }
    svg {
      display: block;
      width: 100%;
      height: 100%;
      user-select: none;
      touch-action: none;
    }
    .graph-toolbar {
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 2;
      display: flex;
      gap: 5px;
      padding: 5px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: var(--shadow);
    }
    .graph-toolbar button {
      min-height: 28px;
      min-width: 30px;
      padding: 3px 8px;
      font-size: 12px;
    }
    .graph-hint {
      position: absolute;
      left: 10px;
      bottom: 10px;
      z-index: 2;
      max-width: min(520px, calc(100% - 20px));
      padding: 5px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.9);
      color: var(--muted);
      font-size: 11px;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 9px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    .legend span {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .swatch {
      width: 14px;
      height: 3px;
      display: inline-block;
      background: var(--green);
    }
    .swatch.false {
      background: var(--red);
      border-top: 1px dashed var(--red);
    }
    .swatch.unsupported { background: var(--amber); }
    .swatch.accepted { background: var(--green); height: 10px; border-radius: 50%; }
    .swatch.rejected { background: var(--red); height: 10px; border-radius: 50%; }
    .swatch.known { height: 10px; border-radius: 50%; background: var(--blue); }
    .graph-edge-hit {
      cursor: pointer;
    }
    .graph-edge-hit:hover + .graph-edge {
      stroke-width: 5;
      opacity: 1;
    }
    .graph-node {
      cursor: pointer;
    }
    .graph-node:hover circle {
      stroke-width: 3;
    }
    .candidate-tools {
      display: grid;
      grid-template-columns: 1fr 140px;
      gap: 8px;
      margin-bottom: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 12px;
    }
    th, td {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th {
      color: var(--muted);
      font-weight: 600;
      background: #fbfcfd;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    .tag {
      display: inline-flex;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 650;
      white-space: nowrap;
    }
    .tag.valid { background: var(--green-soft); color: var(--green); }
    .tag.unsupported { background: var(--amber-soft); color: var(--amber); }
    .tag.nonfinal { background: var(--blue-soft); color: var(--blue); }
    .tag.false { background: var(--red-soft); color: var(--red); }
    .tag.meta { background: var(--violet-soft); color: var(--violet); }
    .muted { color: var(--muted); }
    .transcript {
      display: grid;
      gap: 8px;
    }
    .turn {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 8px;
      background: #fbfcfd;
    }
    .turn strong {
      display: block;
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 3px;
      text-transform: uppercase;
    }
    .error {
      color: var(--red);
      font-weight: 650;
    }
    @media (max-width: 1180px) {
      .controls { grid-template-columns: repeat(5, minmax(76px, 1fr)); }
      .main { grid-template-columns: 1fr; }
      .panel-body { max-height: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="title">Scattered Discovery Viewer</div>
      <div class="button-row">
        <button id="reset" class="primary">Reset world</button>
      </div>
      <div class="controls">
        <div class="control"><label for="world_seed">world seed</label><input id="world_seed" type="number" value="1"></div>
        <div class="control"><label for="episode_seed">episode seed</label><input id="episode_seed" type="number" value="1028"></div>
        <div class="control"><label for="dispersion">dispersion</label><input id="dispersion" type="number" min="0" max="1" step="0.05" value="0.5"></div>
        <div class="control"><label for="num_branches">branches</label><input id="num_branches" type="number" min="1" max="8" value="4"></div>
        <div class="control"><label for="branch_depth">depth</label><input id="branch_depth" type="number" min="1" max="6" value="3"></div>
        <div class="control"><label for="distractors_per_node">distractors</label><input id="distractors_per_node" type="number" min="0" max="6" value="2"></div>
        <div class="control"><label for="base_budget">budget</label><input id="base_budget" type="number" min="1" max="40" value="10"></div>
        <div class="control"><label for="noise_sigma">noise</label><input id="noise_sigma" type="number" min="0.01" max="2" step="0.05" value="0.35"></div>
        <div class="control"><label for="accept_threshold">accept</label><input id="accept_threshold" type="number" min="0.5" max="0.99" step="0.01" value="0.82"></div>
        <div class="control"><label for="reject_threshold">reject</label><input id="reject_threshold" type="number" min="0.01" max="0.5" step="0.01" value="0.18"></div>
      </div>
    </header>

    <main class="main">
      <section class="panel">
        <h2>Episode</h2>
        <div class="panel-body stack">
          <div class="stats" id="stats"></div>
          <div>
            <div class="muted" style="margin-bottom: 5px;">known variables</div>
            <div class="button-row" id="knownVariables"></div>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 5px;">action</div>
            <div class="action-row">
              <input id="actionInput" value="ACTION: INTERVENE x00">
              <button id="runAction">Run</button>
            </div>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 5px;">public state</div>
            <pre id="publicState"></pre>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 5px;">last result</div>
            <pre id="lastResult"></pre>
          </div>
        </div>
      </section>

      <section class="panel">
        <h2>World</h2>
        <div class="graph-wrap">
          <div class="graph-toolbar">
            <button id="zoomIn" title="Zoom in">+</button>
            <button id="zoomOut" title="Zoom out">-</button>
            <button id="fitGraph" title="Fit graph">Fit</button>
          </div>
          <svg id="graph"></svg>
          <div class="graph-hint">Drag to pan. Scroll to zoom. Click a node to intervene, an edge to test, or a final path label to commit.</div>
        </div>
        <div class="legend">
          <span><i class="swatch"></i>true edge</span>
          <span><i class="swatch false"></i>false edge</span>
          <span><i class="swatch unsupported"></i>final unsupported</span>
          <span><i class="swatch accepted"></i>accepted evidence</span>
          <span><i class="swatch rejected"></i>rejected evidence</span>
          <span><i class="swatch known"></i>known variable</span>
        </div>
        <div class="panel-body stack">
          <div>
            <div class="muted" style="margin-bottom: 5px;">target final paths</div>
            <div id="targetPaths"></div>
          </div>
          <div>
            <div class="muted" style="margin-bottom: 5px;">transcript</div>
            <div class="transcript" id="transcript"></div>
          </div>
        </div>
      </section>

      <section class="panel">
        <h2>Candidate Claims</h2>
        <div class="panel-body">
          <div class="candidate-tools">
            <input id="candidateSearch" placeholder="filter expression">
            <select id="candidateFilter">
              <option value="all">all labels</option>
              <option value="valid_final_supported">valid final now</option>
              <option value="final_unsupported">final unsupported</option>
              <option value="true_non_final">true non-final</option>
              <option value="false">false</option>
            </select>
          </div>
          <table>
            <thead>
              <tr>
                <th style="width: 39%;">claim</th>
                <th style="width: 23%;">label</th>
                <th style="width: 18%;">evidence</th>
                <th style="width: 20%;">actions</th>
              </tr>
            </thead>
            <tbody id="candidateRows"></tbody>
          </table>
        </div>
      </section>
    </main>
  </div>

  <script>
    let state = null;
    const graphView = {
      x: 0,
      y: 0,
      w: 900,
      h: 520,
      bounds: null,
      dragging: false,
      moved: false,
      dragStart: null
    };

    const $ = (id) => document.getElementById(id);

    function payloadFromControls() {
      const numeric = [
        "world_seed", "episode_seed", "dispersion", "num_branches", "branch_depth",
        "distractors_per_node", "base_budget", "noise_sigma", "accept_threshold",
        "reject_threshold"
      ];
      const payload = {};
      for (const key of numeric) payload[key] = Number($(key).value);
      payload.protocol = "single";
      payload.max_commit = 1;
      return payload;
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }

    async function resetWorld() {
      try {
        state = await postJson("/api/reset", payloadFromControls());
        graphView.bounds = null;
        render();
      } catch (err) {
        $("lastResult").textContent = String(err);
        $("lastResult").classList.add("error");
      }
    }

    async function runAction(action) {
      if (!state) return;
      try {
        state = await postJson("/api/step", {session_id: state.session_id, action});
        $("actionInput").value = action;
        render();
      } catch (err) {
        $("lastResult").textContent = String(err);
        $("lastResult").classList.add("error");
      }
    }

    function render() {
      renderStats();
      renderKnownVariables();
      renderPublicState();
      renderGraph();
      renderTargets();
      renderCandidates();
      renderTranscript();
      renderLastResult();
    }

    function renderStats() {
      const stats = [
        ["budget", `${state.budget_remaining} / ${state.budget_initial}`],
        ["targets", state.world.target_count],
        ["known", state.known_variables.length],
        ["accepted", state.accepted_claims.length],
        ["rejected", state.rejected_claims.length],
        ["done", state.done ? "yes" : "no"]
      ];
      $("stats").innerHTML = stats.map(([label, value]) =>
        `<div class="stat"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(String(value))}</div></div>`
      ).join("");
    }

    function renderKnownVariables() {
      $("knownVariables").innerHTML = state.known_variables.map((variable) =>
        `<button class="small" data-intervene="${escapeAttr(variable)}">INTERVENE ${escapeHtml(variable)}</button>`
      ).join("");
      document.querySelectorAll("[data-intervene]").forEach((button) => {
        button.onclick = () => runAction(`ACTION: INTERVENE ${button.dataset.intervene}`);
      });
    }

    function renderPublicState() {
      $("publicState").textContent = state.public_state;
    }

    function renderLastResult() {
      $("lastResult").classList.remove("error");
      if (!state.last_step) {
        $("lastResult").textContent = "No action yet.";
        return;
      }
      const lines = [
        state.last_step.action_text || "",
        "",
        state.last_step.observation || ""
      ];
      if (state.last_step.score) {
        const score = state.last_step.score;
        lines.push("");
        lines.push(`score: reward=${score.reward}, valid=${score.valid_unique_count}, false=${score.false_count}, non_final=${score.non_final_count}, unsupported=${score.unsupported_count}`);
      }
      $("lastResult").textContent = lines.join("\n");
    }

    function renderTargets() {
      $("targetPaths").innerHTML = state.world.branches.map((branch) =>
        `<div class="turn"><strong>branch ${branch.branch_id}</strong>${escapeHtml(branch.key)}</div>`
      ).join("");
    }

    function renderTranscript() {
      $("transcript").innerHTML = state.transcript.slice(-8).map((turn) =>
        `<div class="turn"><strong>${escapeHtml(turn.role)}</strong>${escapeHtml(turn.content)}</div>`
      ).join("");
    }

    function renderCandidates() {
      const filter = $("candidateFilter").value;
      const search = $("candidateSearch").value.trim().toLowerCase();
      const rows = state.candidates.filter((candidate) => {
        if (filter !== "all" && candidate.label !== filter) return false;
        if (search && !candidate.expr.toLowerCase().includes(search)) return false;
        return true;
      });
      $("candidateRows").innerHTML = rows.map((candidate) => {
        const testDisabled = candidate.admissible_now ? "" : "disabled";
        const evidence = candidate.evidence.samples > 0
          ? `${candidate.evidence.status} (${candidate.evidence.posterior.toFixed(2)})`
          : "untested";
        return `<tr>
          <td><code>${escapeHtml(candidate.expr)}</code><br><span class="muted">${escapeHtml(candidate.role)}</span></td>
          <td>${labelTag(candidate)}</td>
          <td>${escapeHtml(evidence)}</td>
          <td>
            <div class="button-row">
              <button class="small" data-test="${escapeAttr(candidate.expr)}" ${testDisabled}>TEST</button>
              <button class="small" data-commit="${escapeAttr(candidate.expr)}">COMMIT</button>
            </div>
          </td>
        </tr>`;
      }).join("");
      document.querySelectorAll("[data-test]").forEach((button) => {
        button.onclick = () => runAction(`ACTION: TEST ${button.dataset.test}`);
      });
      document.querySelectorAll("[data-commit]").forEach((button) => {
        button.onclick = () => runAction(`ACTION: COMMIT ${button.dataset.commit}`);
      });
    }

    function labelTag(candidate) {
      const labels = {
        valid_final_supported: ["valid", "valid final now"],
        final_unsupported: ["unsupported", "final unsupported"],
        true_non_final: ["nonfinal", "true non-final"],
        false: ["false", "false"]
      };
      const [klass, text] = labels[candidate.label] || ["meta", candidate.label];
      return `<span class="tag ${klass}">${escapeHtml(text)}</span>`;
    }

    function renderGraph() {
      const svg = $("graph");
      const world = state.world;
      const layout = layoutGraph(world);
      const positions = layout.positions;
      const candidatesByKey = Object.fromEntries(state.candidates.map((candidate) => [candidate.key, candidate]));
      const defs = `<defs>
        <filter id="nodeShadow" x="-40%" y="-40%" width="180%" height="180%">
          <feDropShadow dx="0" dy="1" stdDeviation="1.2" flood-color="#15202b" flood-opacity="0.16"></feDropShadow>
        </filter>
        <marker id="arrowTrue" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L8,3 z" fill="#157a5c"></path>
        </marker>
        <marker id="arrowFalse" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L8,3 z" fill="#b42318"></path>
        </marker>
        <marker id="arrowAmber" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L8,3 z" fill="#9a5b00"></path>
        </marker>
      </defs>`;

      const grid = renderDepthGrid(layout);
      const branchRibbons = world.branches.map((branch, index) => {
        const points = branch.path.map((node) => positions[node]);
        const path = smoothPath(points, 0);
        const color = branchColor(index);
        return `<path d="${path}" fill="none" stroke="${color}" stroke-width="14" stroke-linecap="round" stroke-linejoin="round" opacity="0.13"></path>
          <path d="${path}" fill="none" stroke="${color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.46"></path>`;
      }).join("");

      const edges = world.outgoing_candidates.map((edge) => {
        const a = positions[edge.src];
        const b = positions[edge.dst];
        const key = `edge:${edge.src}->${edge.dst}`;
        const candidate = candidatesByKey[key];
        const status = candidate ? candidate.evidence.status : "unresolved";
        const color = edgeColor(edge.true_edge, status);
        const marker = edgeMarker(edge.true_edge, status);
        const width = status === "accepted" || status === "rejected" ? 3.8 : 2.4;
        const dash = edge.true_edge ? "" : "stroke-dasharray=\"7 5\"";
        const path = edgePath(a, b, edge.true_edge ? 0 : 18);
        const expr = `edge(${edge.src},${edge.dst})`;
        const title = `${expr}: ${edge.true_edge ? "true edge" : "false distractor"}; evidence=${status}`;
        return `<g data-edge-expr="${escapeAttr(expr)}">
          <path class="graph-edge-hit" d="${path}" fill="none" stroke="transparent" stroke-width="14"></path>
          <path class="graph-edge" d="${path}" fill="none" stroke="${color}" stroke-width="${width}" ${dash} marker-end="${marker}" opacity="0.9">
            <title>${escapeHtml(title)}</title>
          </path>
        </g>`;
      }).join("");

      const targetBadges = world.branches.map((branch, index) => {
        const end = positions[branch.path[branch.path.length - 1]];
        const expr = `path(${branch.path.join(",")})`;
        const x = end.x + 36;
        const y = end.y - 15;
        const text = `commit b${branch.branch_id}`;
        const width = 78;
        return `<g data-path-expr="${escapeAttr(expr)}" style="cursor:pointer">
          <rect x="${x}" y="${y}" width="${width}" height="28" rx="6" fill="#ffffff" stroke="${branchColor(index)}" stroke-width="1.5"></rect>
          <text x="${x + width / 2}" y="${y + 18}" text-anchor="middle" font-size="11" fill="#15202b" font-weight="700">${escapeHtml(text)}</text>
          <title>${escapeHtml(`COMMIT ${expr}`)}</title>
        </g>`;
      }).join("");

      const nodes = Object.entries(positions).map(([name, p]) => {
        const known = state.known_variables.includes(name);
        const initial = world.initial_variables.includes(name);
        const trueNode = p.trueNode;
        const fill = known ? "#265fbd" : initial ? "#6750a4" : trueNode ? "#ffffff" : "#fff7ed";
        const stroke = known || initial ? fill : trueNode ? "#157a5c" : "#b42318";
        const textFill = known || initial ? "#ffffff" : "#15202b";
        const dash = trueNode ? "" : "stroke-dasharray=\"4 3\"";
        const title = `${name}: ${known ? "known" : "hidden"} ${trueNode ? "world node" : "distractor node"}`;
        return `<g class="graph-node" data-node="${escapeAttr(name)}">
          <circle cx="${p.x}" cy="${p.y}" r="20" fill="${fill}" stroke="${stroke}" stroke-width="2" ${dash} filter="url(#nodeShadow)"></circle>
          <text x="${p.x}" y="${p.y + 4}" text-anchor="middle" font-size="11" fill="${textFill}" font-weight="700">${escapeHtml(name)}</text>
          <title>${escapeHtml(title)}</title>
        </g>`;
      }).join("");

      svg.innerHTML = defs + grid + branchRibbons + edges + targetBadges + nodes;
      installGraphClickHandlers(svg);
      if (!graphView.bounds) {
        fitGraphView(layout.bounds);
      } else {
        applyGraphView();
      }
    }

    function layoutGraph(world) {
      const left = 100;
      const top = 78;
      const layerGap = 165;
      const laneGap = 90;
      const acc = {};
      for (const branch of world.branches) {
        branch.path.forEach((node, depth) => {
          if (!acc[node]) acc[node] = {x: 0, y: 0, n: 0, branches: new Set(), trueNode: true};
          acc[node].x += left + depth * layerGap;
          acc[node].y += top + branch.branch_id * laneGap;
          acc[node].n += 1;
          acc[node].branches.add(branch.branch_id);
        });
      }
      const positions = {};
      for (const [node, value] of Object.entries(acc)) {
        positions[node] = {
          x: value.x / value.n,
          y: value.y / value.n,
          branches: Array.from(value.branches),
          trueNode: value.trueNode
        };
      }
      const falseOffsets = {};
      for (const edge of world.outgoing_candidates) {
        if (positions[edge.dst]) continue;
        const parent = positions[edge.src] || {x: left, y: top};
        falseOffsets[edge.src] = (falseOffsets[edge.src] || 0) + 1;
        const offset = falseOffsets[edge.src];
        const direction = offset % 2 === 0 ? -1 : 1;
        const magnitude = Math.ceil(offset / 2);
        positions[edge.dst] = {
          x: parent.x + layerGap * 0.74,
          y: parent.y + direction * (34 + magnitude * 18),
          branches: [],
          trueNode: false
        };
      }
      const xs = Object.values(positions).map((p) => p.x);
      const ys = Object.values(positions).map((p) => p.y);
      const maxDepth = Math.max(...world.branches.map((branch) => branch.path.length - 1));
      return {
        positions,
        maxDepth,
        layerGap,
        left,
        top,
        laneGap,
        branchCount: world.branches.length,
        bounds: {
          x: Math.min(...xs) - 90,
          y: Math.min(...ys) - 80,
          width: Math.max(...xs) - Math.min(...xs) + 250,
          height: Math.max(...ys) - Math.min(...ys) + 170
        }
      };
    }

    function renderDepthGrid(layout) {
      const lines = [];
      const y1 = layout.top - 48;
      const y2 = layout.top + (layout.branchCount - 1) * layout.laneGap + 54;
      for (let depth = 0; depth <= layout.maxDepth; depth += 1) {
        const x = layout.left + depth * layout.layerGap;
        lines.push(`<line x1="${x}" y1="${y1}" x2="${x}" y2="${y2}" stroke="#dfe5ee" stroke-width="1"></line>`);
        lines.push(`<text x="${x}" y="${y1 - 12}" text-anchor="middle" fill="#5d6978" font-size="11">depth ${depth}</text>`);
      }
      for (let branch = 0; branch < layout.branchCount; branch += 1) {
        const y = layout.top + branch * layout.laneGap;
        lines.push(`<line x1="${layout.left - 54}" y1="${y}" x2="${layout.left + layout.maxDepth * layout.layerGap + 130}" y2="${y}" stroke="#e8edf4" stroke-width="1"></line>`);
        lines.push(`<text x="${layout.left - 62}" y="${y + 4}" text-anchor="end" fill="#5d6978" font-size="11">branch ${branch}</text>`);
      }
      return `<g>${lines.join("")}</g>`;
    }

    function edgePath(a, b, offset) {
      const startX = a.x + 22;
      const endX = b.x - 22;
      const dx = Math.max(52, Math.abs(endX - startX) * 0.5);
      return `M ${startX} ${a.y} C ${startX + dx} ${a.y + offset}, ${endX - dx} ${b.y + offset}, ${endX} ${b.y}`;
    }

    function smoothPath(points, offset) {
      if (!points.length) return "";
      let path = `M ${points[0].x} ${points[0].y + offset}`;
      for (let i = 1; i < points.length; i += 1) {
        const prev = points[i - 1];
        const curr = points[i];
        const mid = (prev.x + curr.x) / 2;
        path += ` C ${mid} ${prev.y + offset}, ${mid} ${curr.y + offset}, ${curr.x} ${curr.y + offset}`;
      }
      return path;
    }

    function branchColor(index) {
      const colors = ["#157a5c", "#265fbd", "#9a5b00", "#6750a4", "#b42318", "#087990", "#6f42c1", "#0f766e"];
      return colors[index % colors.length];
    }

    function edgeColor(trueEdge, status) {
      if (status === "accepted") return "#157a5c";
      if (status === "rejected") return "#b42318";
      return trueEdge ? "#157a5c" : "#b42318";
    }

    function edgeMarker(trueEdge, status) {
      if (status === "accepted") return "url(#arrowTrue)";
      if (status === "rejected") return "url(#arrowFalse)";
      return trueEdge ? "url(#arrowTrue)" : "url(#arrowFalse)";
    }

    function installGraphClickHandlers(svg) {
      svg.querySelectorAll("[data-node]").forEach((node) => {
        node.onclick = () => runAction(`ACTION: INTERVENE ${node.dataset.node}`);
      });
      svg.querySelectorAll("[data-edge-expr]").forEach((edge) => {
        edge.onclick = () => runAction(`ACTION: TEST ${edge.dataset.edgeExpr}`);
      });
      svg.querySelectorAll("[data-path-expr]").forEach((path) => {
        path.onclick = () => runAction(`ACTION: COMMIT ${path.dataset.pathExpr}`);
      });
    }

    function fitGraphView(bounds) {
      graphView.bounds = bounds;
      graphView.x = bounds.x;
      graphView.y = bounds.y;
      graphView.w = bounds.width;
      graphView.h = bounds.height;
      applyGraphView();
    }

    function applyGraphView() {
      $("graph").setAttribute("viewBox", `${graphView.x} ${graphView.y} ${graphView.w} ${graphView.h}`);
    }

    function zoomGraph(factor, center) {
      if (!graphView.bounds) return;
      const svgCenter = center || {
        x: graphView.x + graphView.w / 2,
        y: graphView.y + graphView.h / 2
      };
      const newW = graphView.w * factor;
      const newH = graphView.h * factor;
      graphView.x = svgCenter.x - (svgCenter.x - graphView.x) * factor;
      graphView.y = svgCenter.y - (svgCenter.y - graphView.y) * factor;
      graphView.w = newW;
      graphView.h = newH;
      applyGraphView();
    }

    function svgPointFromEvent(evt) {
      const svg = $("graph");
      const rect = svg.getBoundingClientRect();
      return {
        x: graphView.x + ((evt.clientX - rect.left) / rect.width) * graphView.w,
        y: graphView.y + ((evt.clientY - rect.top) / rect.height) * graphView.h
      };
    }

    function setupGraphInteractions() {
      const svg = $("graph");
      svg.addEventListener("wheel", (evt) => {
        evt.preventDefault();
        const factor = evt.deltaY < 0 ? 0.86 : 1.16;
        zoomGraph(factor, svgPointFromEvent(evt));
      }, {passive: false});
      svg.addEventListener("pointerdown", (evt) => {
        if (evt.target.closest && evt.target.closest("[data-node], [data-edge-expr], [data-path-expr]")) return;
        graphView.dragging = true;
        graphView.moved = false;
        graphView.dragStart = {clientX: evt.clientX, clientY: evt.clientY, x: graphView.x, y: graphView.y};
        svg.setPointerCapture(evt.pointerId);
      });
      svg.addEventListener("pointermove", (evt) => {
        if (!graphView.dragging || !graphView.dragStart) return;
        const rect = svg.getBoundingClientRect();
        const dx = ((evt.clientX - graphView.dragStart.clientX) / rect.width) * graphView.w;
        const dy = ((evt.clientY - graphView.dragStart.clientY) / rect.height) * graphView.h;
        graphView.x = graphView.dragStart.x - dx;
        graphView.y = graphView.dragStart.y - dy;
        graphView.moved = Math.abs(dx) + Math.abs(dy) > 2;
        applyGraphView();
      });
      svg.addEventListener("pointerup", (evt) => {
        graphView.dragging = false;
        graphView.dragStart = null;
        if (svg.hasPointerCapture(evt.pointerId)) svg.releasePointerCapture(evt.pointerId);
      });
      $("zoomIn").onclick = () => zoomGraph(0.82);
      $("zoomOut").onclick = () => zoomGraph(1.22);
      $("fitGraph").onclick = () => graphView.bounds && fitGraphView(graphView.bounds);
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function escapeAttr(value) {
      return escapeHtml(value);
    }

    $("reset").onclick = resetWorld;
    $("runAction").onclick = () => runAction($("actionInput").value);
    $("candidateSearch").oninput = () => state && renderCandidates();
    $("candidateFilter").onchange = () => state && renderCandidates();
    setupGraphInteractions();
    resetWorld();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the scattered env viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    url = f"http://{args.host}:{server.server_port}"
    print(f"Scattered Discovery Viewer: {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
