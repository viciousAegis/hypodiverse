from __future__ import annotations

import argparse
import base64
import json
import math
import os
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ENTITY = "akshitsinha3"
DEFAULT_PROJECT = "scattered-discovery"


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def _wandb_gql(api_key: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Basic "
            + base64.b64encode(("api:" + api_key).encode()).decode(),
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], indent=2))
    return payload


def _latest_run(api_key: str, *, entity: str, project: str, pattern: str) -> str:
    query = """
query Runs($entity:String!,$project:String!,$filters:JSONString){
  project(name:$project,entityName:$entity){
    runs(first:3,order:"-created_at",filters:$filters){
      edges{node{name}}
    }
  }
}
"""
    payload = _wandb_gql(
        api_key,
        query,
        {
            "entity": entity,
            "project": project,
            "filters": json.dumps({"display_name": {"$regex": pattern}}),
        },
    )
    edges = payload["data"]["project"]["runs"]["edges"]
    if not edges:
        raise RuntimeError(f"No W&B runs matched pattern {pattern!r}")
    return str(edges[0]["node"]["name"])


def _run_history(
    api_key: str, *, entity: str, project: str, run_name: str
) -> dict[str, Any]:
    query = """
query RunHistory($entity:String!,$project:String!,$run:String!){
  project(name:$project,entityName:$entity){
    run(name:$run){
      name displayName state updatedAt history(samples:1000) summaryMetrics
    }
  }
}
"""
    payload = _wandb_gql(
        api_key,
        query,
        {"entity": entity, "project": project, "run": run_name},
    )
    return dict(payload["data"]["project"]["run"])


def _history_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in run["history"]:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    return None


def _points(rows: list[dict[str, Any]], key: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        x = _num(row.get("training/global_step", row.get("_step")))
        y = _num(row.get(key))
        if x is not None and y is not None:
            points.append((x, y))
    return points


def _write_superplot(rows: list[dict[str, Any]], run: dict[str, Any], path: Path) -> None:
    panels = [
        (
            "Training Reward",
            [
                ("score mean", "critic/score/mean", "#2563eb"),
                ("score max", "critic/score/max", "#16a34a"),
                ("score min", "critic/score/min", "#dc2626"),
            ],
        ),
        (
            "Length",
            [
                ("response mean", "response_length/mean", "#7c3aed"),
                ("clip ratio", "response_length/clip_ratio", "#ea580c"),
            ],
        ),
        (
            "Actor",
            [
                ("loss", "actor/loss", "#2563eb"),
                ("grad norm", "actor/grad_norm", "#16a34a"),
                ("entropy", "actor/entropy", "#dc2626"),
            ],
        ),
        (
            "Timing Seconds",
            [
                ("step", "perf/time_per_step", "#111827"),
                ("gen", "timing_s/gen", "#2563eb"),
                ("update actor", "timing_s/update_actor", "#16a34a"),
            ],
        ),
        (
            "Validation",
            [
                ("val reward", "val-core/causal_micro_lab_val/reward/mean@1", "#dc2626"),
                (
                    "base reward",
                    "val-aux/causal_micro_lab_val/base_terminal_reward/mean@1",
                    "#16a34a",
                ),
                (
                    "parse valid",
                    "val-aux/causal_micro_lab_val/parse_valid/mean@1",
                    "#2563eb",
                ),
                (
                    "cap hit",
                    "val-aux/causal_micro_lab_val/response_length_cap_hit/mean@1",
                    "#ea580c",
                ),
            ],
        ),
    ]
    width, height = 900, 220
    left, right, top, bottom = 60, 20, 26, 42
    html = [
        "<!doctype html><meta charset=\"utf-8\"><title>W&B CML superplot</title>",
        "<style>body{font-family:system-ui,Arial;margin:24px;color:#111827}"
        "svg{width:100%;max-width:980px;border:1px solid #e5e7eb;margin:12px 0;background:#fff}"
        ".label{font-size:12px;fill:#374151}.title{font-size:15px;font-weight:700}"
        ".legend{font-size:12px}.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#9ca3af;stroke-width:1}"
        ".note{color:#4b5563}</style>",
        f"<h2>{run['displayName']}</h2>",
        f"<p class=\"note\">State: {run['state']} | Updated: {run['updatedAt']}</p>",
    ]
    for title, series in panels:
        all_points = [point for _, key, _ in series for point in _points(rows, key)]
        if not all_points:
            continue
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmin == xmax:
            xmax = xmin + 1
        if ymin == ymax:
            ymin -= 0.5
            ymax += 0.5
        pad = (ymax - ymin) * 0.08
        ymin -= pad
        ymax += pad

        def sx(x: float) -> float:
            return left + (x - xmin) / (xmax - xmin) * (width - left - right)

        def sy(y: float) -> float:
            return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

        html.append(f"<svg viewBox=\"0 0 {width} {height}\">")
        html.append(f"<text class=\"title\" x=\"{left}\" y=\"18\">{title}</text>")
        for idx in range(5):
            y = top + idx * (height - top - bottom) / 4
            val = ymax - idx * (ymax - ymin) / 4
            html.append(
                f"<line class=\"grid\" x1=\"{left}\" x2=\"{width-right}\" "
                f"y1=\"{y:.1f}\" y2=\"{y:.1f}\"/>"
                f"<text class=\"label\" x=\"6\" y=\"{y+4:.1f}\">{val:.3g}</text>"
            )
        html.append(
            f"<line class=\"axis\" x1=\"{left}\" x2=\"{width-right}\" "
            f"y1=\"{height-bottom}\" y2=\"{height-bottom}\"/>"
            f"<line class=\"axis\" x1=\"{left}\" x2=\"{left}\" "
            f"y1=\"{top}\" y2=\"{height-bottom}\"/>"
        )
        for tick in range(int(xmin), int(xmax) + 1):
            html.append(
                f"<text class=\"label\" x=\"{sx(tick)-4:.1f}\" "
                f"y=\"{height-18}\">{tick}</text>"
            )
        legend_x, legend_y = left, height - 6
        for label, key, color in series:
            pts = _points(rows, key)
            if not pts:
                continue
            path_data = " ".join(
                ("M" if idx == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}"
                for idx, (x, y) in enumerate(pts)
            )
            html.append(
                f"<path d=\"{path_data}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\"/>"
            )
            for x, y in pts:
                html.append(
                    f"<circle cx=\"{sx(x):.1f}\" cy=\"{sy(y):.1f}\" r=\"3\" fill=\"{color}\">"
                    f"<title>{label} step {x:g}: {y:g}</title></circle>"
                )
            html.append(
                f"<circle cx=\"{legend_x}\" cy=\"{legend_y-4}\" r=\"4\" fill=\"{color}\"/>"
                f"<text class=\"legend\" x=\"{legend_x+8}\" y=\"{legend_y}\">{label}</text>"
            )
            legend_x += 126
        html.append("</svg>")
    path.write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--run-name")
    parser.add_argument("--pattern", default="causal_micro_lab")
    parser.add_argument("--output-dir", default="artifacts/wandb_metrics")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    _load_env(Path(args.env_file))
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise SystemExit("WANDB_API_KEY is not set.")
    run_name = args.run_name or _latest_run(
        api_key, entity=args.entity, project=args.project, pattern=args.pattern
    )
    run = _run_history(api_key, entity=args.entity, project=args.project, run_name=run_name)
    rows = _history_rows(run)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"{run_name}_history.jsonl"
    with history_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    plot_path = output_dir / f"{run_name}_superplot.html"
    _write_superplot(rows, run, plot_path)
    print(f"run={run_name} display={run['displayName']} state={run['state']} updated={run['updatedAt']}")
    print(f"history={history_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
