from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from scattered_discovery.envs.causal_micro_lab.interventions import Experiment
from scattered_discovery.envs.causal_micro_lab.planner import (
    outcome_entropy,
    run_oracle_closed_loop,
)
from scattered_discovery.envs.causal_micro_lab.prompt_builder import build_prompt
from scattered_discovery.envs.causal_micro_lab.signatures import ModeTable, build_mode_table
from scattered_discovery.envs.causal_micro_lab.state_generator import EvidenceState
from scattered_discovery.envs.causal_micro_lab.tables import generate_states
from scattered_discovery.envs.causal_micro_lab.verifier import verify_output


@dataclass(frozen=True)
class MicroLabBank:
    states: tuple[EvidenceState, ...]
    seed: int
    target_counts: tuple[int, ...]
    states_per_count: int
    max_evidence: int
    beam_width: int
    separation_bucket: str


BANKS: dict[str, MicroLabBank] = {}
TABLE = build_mode_table()


def _parse_target_counts(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        parts = [str(item) for item in value]
    else:
        parts = ["2", "4", "8", "16"]
    counts = tuple(int(item) for item in parts)
    if not counts:
        raise ValueError("target_counts must not be empty")
    return counts


def create_bank(payload: dict[str, Any], *, mode_table: ModeTable | None = None) -> tuple[str, MicroLabBank]:
    table = mode_table or TABLE
    seed = int(payload.get("seed", 1))
    target_counts = _parse_target_counts(payload.get("target_counts", "2,4,8,16"))
    states_per_count = int(payload.get("states_per_count", 1))
    max_evidence = int(payload.get("max_evidence", 8))
    beam_width = int(payload.get("beam_width", 32))
    separation_bucket = str(payload.get("separation_bucket", "any"))
    if states_per_count < 1 or states_per_count > 6:
        raise ValueError("states_per_count must be between 1 and 6 for the live viewer")
    if beam_width < 8 or beam_width > 128:
        raise ValueError("beam_width must be between 8 and 128 for the live viewer")
    if separation_bucket not in {"any", "low", "medium", "high"}:
        raise ValueError("separation_bucket must be any, low, medium, or high")
    states = generate_states(
        target_counts=target_counts,
        states_per_count=states_per_count,
        seed=seed,
        max_evidence=max_evidence,
        beam_width=beam_width,
        separation_bucket=separation_bucket,
        mode_table=table,
    )
    bank = MicroLabBank(
        states=tuple(states),
        seed=seed,
        target_counts=target_counts,
        states_per_count=states_per_count,
        max_evidence=max_evidence,
        beam_width=beam_width,
        separation_bucket=separation_bucket,
    )
    bank_id = uuid4().hex
    BANKS[bank_id] = bank
    return bank_id, bank


def bank_summary(bank_id: str, bank: MicroLabBank, *, mode_table: ModeTable | None = None) -> dict[str, Any]:
    table = mode_table or TABLE
    counts = Counter(state.valid_mode_count for state in bank.states)
    separation = Counter(state.separation_bucket for state in bank.states)
    families = Counter(state.family_bucket for state in bank.states)
    return {
        "bank_id": bank_id,
        "seed": bank.seed,
        "target_counts": list(bank.target_counts),
        "states_per_count": bank.states_per_count,
        "max_evidence": bank.max_evidence,
        "beam_width": bank.beam_width,
        "separation_bucket": bank.separation_bucket,
        "mode_count": len(table.modes),
        "syntactic_hypothesis_count": table.all_hypotheses_count,
        "experiment_count": len(table.experiments),
        "state_count": len(bank.states),
        "state_counts": {str(key): value for key, value in sorted(counts.items())},
        "separation_counts": dict(sorted(separation.items())),
        "family_bucket_counts": dict(sorted(families.items())),
        "states": [_state_list_item(state) for state in bank.states],
    }


def _state_list_item(state: EvidenceState) -> dict[str, Any]:
    return {
        "state_id": state.state_id,
        "hidden_mode_id": state.hidden_mode_id,
        "valid_mode_count": state.valid_mode_count,
        "evidence_size": state.evidence_size,
        "separation_bucket": state.separation_bucket,
        "mean_separation": state.mean_separation,
        "minimum_separation": state.minimum_separation,
        "maximum_separation": state.maximum_separation,
        "family_bucket": state.family_bucket,
    }


def state_snapshot(
    bank_id: str,
    bank: MicroLabBank,
    state_id: str | None = None,
    *,
    mode_table: ModeTable | None = None,
) -> dict[str, Any]:
    table = mode_table or TABLE
    state = _select_state(bank, state_id)
    valid_modes = [table.modes_by_id[mode_id] for mode_id in state.valid_mode_ids]
    family_counts = Counter(mode.family for mode in valid_modes)
    source_counts = Counter(mode.family[0] for mode in valid_modes)
    operator_counts = Counter(mode.family[1] for mode in valid_modes)
    hidden = table.modes_by_id.get(state.hidden_mode_id)
    m_trajectory = _m_trajectory(state, mode_table=table)
    return {
        "summary": bank_summary(bank_id, bank, mode_table=table),
        "selected": {
            **_state_list_item(state),
            "visible_experiments": [
                item.to_json(table.experiments) for item in state.evidence
            ],
            "available_experiments": _available_experiment_rows(
                state,
                valid_modes=valid_modes,
                experiments=table.experiments,
            ),
            "family_counts": {
                f"{source}/{operator}": count
                for (source, operator), count in sorted(family_counts.items())
            },
            "source_counts": dict(sorted(source_counts.items())),
            "operator_counts": dict(sorted(operator_counts.items())),
            "hidden_canonical_json": hidden.canonical.render_json() if hidden else None,
            "valid_mode_examples": [
                {
                    "mode_id": mode.mode_id,
                    "family": list(mode.family),
                    "canonical_json": mode.canonical.render_json(),
                }
                for mode in valid_modes[:24]
            ],
            "m_trajectory": m_trajectory,
            "prompt": build_prompt(state),
        },
    }


def _m_trajectory(
    state: EvidenceState,
    *,
    mode_table: ModeTable,
) -> list[dict[str, Any]]:
    trajectory: list[dict[str, Any]] = [
        {
            "step": 0,
            "experiment_id": None,
            "remaining_modes": state.valid_mode_count,
            "information_gain_modes": 0,
        }
    ]
    try:
        trace = run_oracle_closed_loop(state, max_steps=8, mode_table=mode_table)
    except ValueError:
        return trajectory
    for item in trace.steps:
        trajectory.append(
            {
                "step": item["step"],
                "experiment_id": item["experiment_id"],
                "remaining_modes": item["remaining_version_space_size"],
                "information_gain_modes": item["information_gain_modes"],
            }
        )
    return trajectory


def _select_state(bank: MicroLabBank, state_id: str | None) -> EvidenceState:
    if not bank.states:
        raise ValueError("bank contains no states")
    if not state_id:
        return bank.states[0]
    for state in bank.states:
        if state.state_id == state_id:
            return state
    raise ValueError(f"unknown state_id {state_id!r}")


def _available_experiment_rows(
    state: EvidenceState,
    *,
    valid_modes: list[Any],
    experiments: tuple[Experiment, ...],
) -> list[dict[str, Any]]:
    observed = set(state.observed_experiment_ids())
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        if experiment.experiment_id in observed:
            continue
        outcomes = [mode.signature[experiment.experiment_id] for mode in valid_modes]
        counts = Counter(outcomes)
        rows.append(
            {
                "experiment_id": experiment.experiment_id,
                "inputs": experiment.inputs_dict(),
                "intervention": experiment.intervention,
                "entropy": outcome_entropy(outcomes),
                "outcome_counts": {
                    "".join(str(bit) for bit in outcome): count
                    for outcome, count in sorted(counts.items())
                },
            }
        )
    return sorted(rows, key=lambda item: (-float(item["entropy"]), int(item["experiment_id"])))


def verify_payload(payload: dict[str, Any], *, mode_table: ModeTable | None = None) -> dict[str, Any]:
    table = mode_table or TABLE
    bank_id = str(payload.get("bank_id", ""))
    bank = BANKS.get(bank_id)
    if bank is None:
        raise ValueError("unknown bank_id")
    state = _select_state(bank, payload.get("state_id"))
    text = str(payload.get("text", ""))
    result = verify_output(text, state, mode_table=table)
    return {
        "verification": result.as_dict(),
        "state": _state_list_item(state),
    }


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True).encode("utf-8")


class MicroLabViewerHandler(BaseHTTPRequestHandler):
    server_version = "CausalMicroLabViewer/0.1"

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
            if path == "/api/build":
                bank_id, bank = create_bank(payload)
                self._send_json(state_snapshot(bank_id, bank))
                return
            if path == "/api/select":
                bank_id = str(payload.get("bank_id", ""))
                bank = BANKS.get(bank_id)
                if bank is None:
                    self._send_json({"error": "Unknown bank."}, status=404)
                    return
                self._send_json(state_snapshot(bank_id, bank, payload.get("state_id")))
                return
            if path == "/api/verify":
                self._send_json(verify_payload(payload))
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
        return json.loads(self.rfile.read(length).decode("utf-8"))

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
  <title>Boolean Causal Micro-Lab Viewer</title>
  <style>
    :root {
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #17212b;
      --muted: #667384;
      --line: #d9dee7;
      --accent: #1f6f78;
      --accent-dark: #164d54;
      --green: #18795b;
      --amber: #9c5b00;
      --red: #b3261e;
      --soft: #eef3f5;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font: 14px/1.4 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    h2 { margin: 0 0 10px; font-size: 14px; font-weight: 650; }
    button, input, select, textarea {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      min-height: 34px;
      padding: 7px 10px;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    button:hover { border-color: #9ea8b6; }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 8px;
      background: white;
      color: var(--ink);
    }
    textarea {
      min-height: 210px;
      resize: vertical;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    main {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      gap: 14px;
      padding: 14px;
    }
    aside, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .controls {
      display: grid;
      gap: 10px;
    }
    .field label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 14px;
    }
    .span-3 { grid-column: span 3; }
    .span-4 { grid-column: span 4; }
    .span-5 { grid-column: span 5; }
    .span-6 { grid-column: span 6; }
    .span-7 { grid-column: span 7; }
    .span-8 { grid-column: span 8; }
    .span-12 { grid-column: span 12; }
    .metric-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 8px;
    }
    .metric {
      background: var(--soft);
      border-radius: 6px;
      padding: 8px;
      min-width: 0;
    }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { font-size: 20px; font-weight: 700; margin-top: 2px; }
    .list {
      display: grid;
      gap: 6px;
      max-height: 560px;
      overflow: auto;
    }
    .state-item {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      background: white;
      border-radius: 6px;
      padding: 8px;
    }
    .state-item.active { border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }
    .muted { color: var(--muted); }
    .tag {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      background: var(--soft);
      color: var(--ink);
      font-size: 12px;
      margin: 2px 3px 2px 0;
    }
    .tag.green { color: var(--green); background: #e8f5ef; }
    .tag.amber { color: var(--amber); background: #fff2da; }
    .tag.red { color: var(--red); background: #fde9e7; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 6px;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 650;
      position: sticky;
      top: 0;
      background: white;
    }
    .scroll {
      max-height: 320px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow: auto;
      max-height: 420px;
      background: #101820;
      color: #edf5f7;
      border-radius: 6px;
      padding: 10px;
      font-size: 12px;
    }
    .bars { display: grid; gap: 6px; }
    .bar-row {
      display: grid;
      grid-template-columns: 110px 1fr 36px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
    }
    .bar-track {
      height: 9px;
      background: var(--soft);
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: var(--accent);
      border-radius: 999px;
    }
    .error {
      color: var(--red);
      font-weight: 650;
      margin-top: 8px;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      .span-3, .span-4, .span-5, .span-6, .span-7, .span-8, .span-12 { grid-column: span 1; }
      .metric-row { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Boolean Causal Micro-Lab Viewer</h1>
    <div class="muted" id="status">Ready</div>
  </header>
  <main>
    <aside>
      <h2>State Bank</h2>
      <div class="controls">
        <div class="field">
          <label>Target mode counts</label>
          <input id="targetCounts" value="2,4,8,16">
        </div>
        <div class="field">
          <label>States per count</label>
          <input id="statesPerCount" type="number" min="1" max="6" value="1">
        </div>
        <div class="field">
          <label>Seed</label>
          <input id="seed" type="number" value="1">
        </div>
        <div class="field">
          <label>Beam width</label>
          <input id="beamWidth" type="number" min="8" max="128" value="32">
        </div>
        <div class="field">
          <label>Separation bucket</label>
          <select id="separationBucket">
            <option value="any">Any</option>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>
        <button class="primary" id="buildBtn">Generate</button>
      </div>
      <hr>
      <div id="bankStats" class="muted">No bank loaded.</div>
      <h2 style="margin-top:14px;">States</h2>
      <div class="list" id="stateList"></div>
    </aside>
    <div class="grid">
      <section class="span-12">
        <h2>Summary</h2>
        <div class="metric-row" id="metrics"></div>
      </section>
      <section class="span-5">
        <h2>Valid-Mode Families</h2>
        <div class="bars" id="familyBars"></div>
      </section>
      <section class="span-7">
        <h2>Evidence</h2>
        <div id="evidence"></div>
      </section>
      <section class="span-5">
        <h2>M Trajectory</h2>
        <div class="scroll">
          <table>
            <thead><tr><th>Step</th><th>Experiment</th><th>M</th><th>Gain</th></tr></thead>
            <tbody id="mTrajectory"></tbody>
          </table>
        </div>
      </section>
      <section class="span-7">
        <h2>Experiment Disagreement</h2>
        <div class="scroll">
          <table>
            <thead><tr><th>ID</th><th>Inputs</th><th>Intervention</th><th>Entropy</th><th>Outcomes</th></tr></thead>
            <tbody id="experiments"></tbody>
          </table>
        </div>
      </section>
      <section class="span-5">
        <h2>Hidden Canonical Program</h2>
        <pre id="hiddenProgram"></pre>
      </section>
      <section class="span-6">
        <h2>Prompt</h2>
        <pre id="prompt"></pre>
      </section>
      <section class="span-6">
        <h2>Verifier</h2>
        <textarea id="verifyText" spellcheck="false"></textarea>
        <div style="display:flex; gap:8px; margin-top:8px;">
          <button class="primary" id="verifyBtn">Verify JSON</button>
          <button id="copyHiddenBtn">Use Hidden Program</button>
        </div>
        <pre id="verifyResult" style="margin-top:8px;"></pre>
      </section>
    </div>
  </main>
  <script>
    let current = null;
    let selectedStateId = null;

    const $ = (id) => document.getElementById(id);
    const status = (text) => { $("status").textContent = text; };

    async function post(path, payload) {
      status("Loading...");
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 45000);
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload || {}),
        signal: controller.signal
      });
      clearTimeout(timer);
      const data = await response.json();
      if (!response.ok || data.error) {
        status("Error");
        throw new Error(data.error || response.statusText);
      }
      status("Ready");
      return data;
    }

    function fmtNumber(value, digits = 3) {
      if (value === null || value === undefined) return "n/a";
      if (Number.isInteger(value)) return String(value);
      return Number(value).toFixed(digits);
    }

    function renderBars(target, counts) {
      const entries = Object.entries(counts || {});
      const max = Math.max(1, ...entries.map(([, value]) => value));
      target.innerHTML = entries.length ? entries.map(([key, value]) => `
        <div class="bar-row">
          <div>${key}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${(value / max) * 100}%"></div></div>
          <div>${value}</div>
        </div>
      `).join("") : '<div class="muted">No data.</div>';
    }

    function render(data) {
      current = data;
      const summary = data.summary;
      const selected = data.selected;
      selectedStateId = selected.state_id;
      $("bankStats").innerHTML = `
        <div><strong>${summary.state_count}</strong> states</div>
        <div>${summary.syntactic_hypothesis_count} programs, ${summary.mode_count} semantic modes</div>
        <div>${summary.experiment_count} experiments</div>
        <div>separation: ${summary.separation_bucket}</div>
      `;
      $("stateList").innerHTML = summary.states.map((state) => `
        <button class="state-item ${state.state_id === selected.state_id ? "active" : ""}" data-state="${state.state_id}">
          <div><strong>M=${state.valid_mode_count}</strong> <span class="tag">${state.separation_bucket}</span> <span class="tag">${state.family_bucket}</span></div>
          <div class="muted">evidence=${state.evidence_size}, mean d=${fmtNumber(state.mean_separation)}</div>
        </button>
      `).join("");
      document.querySelectorAll("[data-state]").forEach((button) => {
        button.addEventListener("click", () => selectState(button.dataset.state));
      });
      $("metrics").innerHTML = [
        ["Valid modes", selected.valid_mode_count],
        ["Evidence", selected.evidence_size],
        ["Mean sep.", fmtNumber(selected.mean_separation)],
        ["Min sep.", fmtNumber(selected.minimum_separation)],
        ["Max sep.", fmtNumber(selected.maximum_separation)]
      ].map(([label, value]) => `
        <div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>
      `).join("");
      renderBars($("familyBars"), selected.family_counts);
      $("evidence").innerHTML = selected.visible_experiments.map((item, index) => `
        <div style="margin-bottom:8px;">
          <span class="tag green">Experiment ${index + 1}</span>
          <span class="tag">id ${item.experiment_id}</span>
          <div>Inputs: X1=${item.inputs.X1}, X2=${item.inputs.X2}, X3=${item.inputs.X3}</div>
          <div>Intervention: ${item.intervention}</div>
          <div>Observed: Z1=${item.observation.Z1}, Z2=${item.observation.Z2}, Y=${item.observation.Y}</div>
        </div>
      `).join("");
      $("mTrajectory").innerHTML = selected.m_trajectory.map((item) => `
        <tr>
          <td>${item.step}</td>
          <td>${item.experiment_id === null ? "current" : item.experiment_id}</td>
          <td>${item.remaining_modes}</td>
          <td>${item.information_gain_modes}</td>
        </tr>
      `).join("");
      $("experiments").innerHTML = selected.available_experiments.slice(0, 40).map((item) => `
        <tr>
          <td>${item.experiment_id}</td>
          <td>X1=${item.inputs.X1}, X2=${item.inputs.X2}, X3=${item.inputs.X3}</td>
          <td>${item.intervention}</td>
          <td>${fmtNumber(item.entropy)}</td>
          <td>${Object.entries(item.outcome_counts).map(([k, v]) => `${k}:${v}`).join(", ")}</td>
        </tr>
      `).join("");
      $("hiddenProgram").textContent = selected.hidden_canonical_json || "";
      $("prompt").textContent = selected.prompt || "";
      $("verifyText").value = selected.valid_mode_examples[0]?.canonical_json || selected.hidden_canonical_json || "";
      $("verifyResult").textContent = "";
    }

    async function build() {
      try {
        $("buildBtn").disabled = true;
        $("buildBtn").textContent = "Generating...";
        $("bankStats").innerHTML = '<div><strong>Generating states...</strong></div><div>This can take a few seconds for all four M buckets.</div>';
        await new Promise((resolve) => setTimeout(resolve, 0));
        const data = await post("/api/build", {
          target_counts: $("targetCounts").value,
          states_per_count: Number($("statesPerCount").value),
          seed: Number($("seed").value),
          beam_width: Number($("beamWidth").value),
          separation_bucket: $("separationBucket").value,
          max_evidence: 8
        });
        render(data);
      } catch (error) {
        const message = error.name === "AbortError"
          ? "Generation timed out. Try fewer target counts, states per count = 1, or beam width = 32."
          : error.message;
        $("bankStats").innerHTML = `<div class="error">${message}</div>`;
      } finally {
        $("buildBtn").disabled = false;
        $("buildBtn").textContent = "Generate";
      }
    }

    async function selectState(stateId) {
      try {
        const data = await post("/api/select", {
          bank_id: current.summary.bank_id,
          state_id: stateId
        });
        render(data);
      } catch (error) {
        status(error.message);
      }
    }

    async function verify() {
      if (!current) return;
      try {
        const data = await post("/api/verify", {
          bank_id: current.summary.bank_id,
          state_id: selectedStateId,
          text: $("verifyText").value
        });
        $("verifyResult").textContent = JSON.stringify(data.verification, null, 2);
      } catch (error) {
        $("verifyResult").textContent = error.message;
      }
    }

    $("buildBtn").addEventListener("click", build);
    $("verifyBtn").addEventListener("click", verify);
    $("copyHiddenBtn").addEventListener("click", () => {
      if (current) $("verifyText").value = current.selected.hidden_canonical_json || "";
    });
    build();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Boolean Causal Micro-Lab viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MicroLabViewerHandler)
    url = f"http://{args.host}:{server.server_port}"
    print(f"Boolean Causal Micro-Lab Viewer: {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
