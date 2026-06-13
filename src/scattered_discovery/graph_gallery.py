from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from scattered_discovery.envs.scattered_world import GeneratedWorld, WorldGenerator


def parse_dispersions(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one dispersion value is required.")
    return values


def generate_gallery(
    *,
    output: Path,
    dispersions: list[float],
    samples_per_dispersion: int,
    seed: int,
    num_branches: int,
    branch_depth: int,
    distractors_per_node: int,
) -> Path:
    generator = WorldGenerator(
        num_branches=num_branches,
        branch_depth=branch_depth,
        distractors_per_node=distractors_per_node,
    )
    cards = []
    for dispersion_index, dispersion in enumerate(dispersions):
        for sample_index in range(samples_per_dispersion):
            world_seed = seed + dispersion_index * 1000 + sample_index
            world = generator.generate(seed=world_seed, dispersion=dispersion)
            cards.append(
                render_card(
                    world=world,
                    world_seed=world_seed,
                    dispersion=dispersion,
                    sample_index=sample_index,
                )
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_page(cards), encoding="utf-8")
    return output


def render_page(cards: list[str]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Scattered Causal Dispersion Gallery</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #15202b;
      --muted: #5d6978;
      --line: #d8dde6;
      --green: #157a5c;
      --red: #b42318;
      --blue: #265fbd;
      --amber: #9a5b00;
      --violet: #6750a4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.4 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 18px 22px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 5px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: var(--muted);
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 14px;
      padding: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 4px rgba(20, 27, 36, 0.08);
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }}
    .title {{
      font-weight: 750;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .svg-wrap {{
      height: 390px;
      background:
        linear-gradient(#f0f3f7 1px, transparent 1px),
        linear-gradient(90deg, #f0f3f7 1px, transparent 1px),
        #fbfcfd;
      background-size: 28px 28px;
    }}
    svg {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .targets {{
      padding: 9px 12px 12px;
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    code {{
      color: var(--ink);
      background: #f1f4f8;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 1px 4px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Scattered Causal Dispersion Gallery</h1>
    <p>Each card uses the same graph shape settings and varies only dispersion seed/value. Green edges are true causal edges; dashed red edges are distractors.</p>
  </header>
  <main>
    {"".join(cards)}
  </main>
</body>
</html>
"""


def render_card(
    *,
    world: GeneratedWorld,
    world_seed: int,
    dispersion: float,
    sample_index: int,
) -> str:
    targets = "\n".join(
        f"<div>branch {branch.branch_id}: <code>{escape(branch.terminal_key)}</code></div>"
        for branch in world.branches
    )
    return f"""<section class="card">
  <div class="card-head">
    <div class="title">dispersion {dispersion:.2f} / sample {sample_index + 1}</div>
    <div class="meta">seed {world_seed}</div>
  </div>
  <div class="svg-wrap">
    {render_svg(world)}
  </div>
  <div class="targets">
    {targets}
  </div>
</section>
"""


def render_svg(world: GeneratedWorld) -> str:
    positions, bounds = layout_world(world)
    true_edge_set = set(world.true_edges)
    edges = []
    for src, targets in sorted(world.outgoing_candidates.items()):
        for dst in targets:
            a = positions[src]
            b = positions[dst]
            true_edge = (src, dst) in true_edge_set
            color = "#157a5c" if true_edge else "#b42318"
            marker = "arrowTrue" if true_edge else "arrowFalse"
            dash = "" if true_edge else ' stroke-dasharray="7 5"'
            curve = edge_path(a, b, 0 if true_edge else 18)
            edges.append(
                f'<path d="{curve}" fill="none" stroke="{color}" stroke-width="2.6"{dash} '
                f'marker-end="url(#{marker})" opacity="0.92"></path>'
            )

    ribbons = []
    for index, branch in enumerate(world.branches):
        path_points = [positions[node] for node in branch.path]
        path = smooth_path(path_points)
        color = branch_color(index)
        ribbons.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="12" '
            'stroke-linecap="round" stroke-linejoin="round" opacity="0.12"></path>'
        )
        ribbons.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.1" '
            'stroke-linecap="round" stroke-linejoin="round" opacity="0.48"></path>'
        )

    nodes = []
    initial = set(world.initial_variables)
    for name, point in sorted(positions.items()):
        true_node = point["true_node"]
        fill = "#265fbd" if name in initial else "#ffffff" if true_node else "#fff7ed"
        stroke = "#265fbd" if name in initial else "#157a5c" if true_node else "#b42318"
        text = "#ffffff" if name in initial else "#15202b"
        dash = "" if true_node else ' stroke-dasharray="4 3"'
        nodes.append(
            f'<g><circle cx="{point["x"]:.1f}" cy="{point["y"]:.1f}" r="18" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}></circle>'
            f'<text x="{point["x"]:.1f}" y="{point["y"] + 4:.1f}" '
            f'text-anchor="middle" font-size="11" fill="{text}" font-weight="700">'
            f"{escape(name)}</text></g>"
        )

    depth_lines = render_depth_lines(world, bounds)
    defs = """<defs>
  <marker id="arrowTrue" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L8,3 z" fill="#157a5c"></path></marker>
  <marker id="arrowFalse" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L8,3 z" fill="#b42318"></path></marker>
</defs>"""
    return (
        f'<svg viewBox="{bounds["x"]:.1f} {bounds["y"]:.1f} {bounds["w"]:.1f} {bounds["h"]:.1f}" '
        'role="img" aria-label="generated scattered causal world">'
        f"{defs}{depth_lines}{''.join(ribbons)}{''.join(edges)}{''.join(nodes)}</svg>"
    )


def layout_world(
    world: GeneratedWorld,
) -> tuple[dict[str, dict[str, float | bool]], dict[str, float]]:
    left = 100.0
    top = 70.0
    layer_gap = 145.0
    lane_gap = 76.0
    acc: dict[str, dict[str, float | int | bool]] = {}
    for branch in world.branches:
        for depth, node in enumerate(branch.path):
            item = acc.setdefault(node, {"x": 0.0, "y": 0.0, "n": 0, "true_node": True})
            item["x"] = float(item["x"]) + left + depth * layer_gap
            item["y"] = float(item["y"]) + top + branch.branch_id * lane_gap
            item["n"] = int(item["n"]) + 1

    positions: dict[str, dict[str, float | bool]] = {}
    for node, value in acc.items():
        count = int(value["n"])
        positions[node] = {
            "x": float(value["x"]) / count,
            "y": float(value["y"]) / count,
            "true_node": True,
        }

    false_offsets: dict[str, int] = {}
    for src, targets in sorted(world.outgoing_candidates.items()):
        for dst in targets:
            if dst in positions:
                continue
            parent = positions[src]
            false_offsets[src] = false_offsets.get(src, 0) + 1
            offset = false_offsets[src]
            direction = -1 if offset % 2 == 0 else 1
            magnitude = (offset + 1) // 2
            positions[dst] = {
                "x": float(parent["x"]) + layer_gap * 0.68,
                "y": float(parent["y"]) + direction * (30 + magnitude * 16),
                "true_node": False,
            }

    xs = [float(point["x"]) for point in positions.values()]
    ys = [float(point["y"]) for point in positions.values()]
    bounds = {
        "x": min(xs) - 80,
        "y": min(ys) - 70,
        "w": max(xs) - min(xs) + 190,
        "h": max(ys) - min(ys) + 140,
        "left": left,
        "top": top,
        "layer_gap": layer_gap,
        "lane_gap": lane_gap,
    }
    return positions, bounds


def render_depth_lines(world: GeneratedWorld, bounds: dict[str, float]) -> str:
    max_depth = max(len(branch.path) for branch in world.branches) - 1
    y1 = bounds["y"] + 18
    y2 = bounds["y"] + bounds["h"] - 20
    lines = []
    for depth in range(max_depth + 1):
        x = bounds["left"] + depth * bounds["layer_gap"]
        lines.append(
            f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" '
            'stroke="#dfe5ee" stroke-width="1"></line>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{y1 - 10:.1f}" text-anchor="middle" '
            'fill="#5d6978" font-size="10">'
            f"depth {depth}</text>"
        )
    return "<g>" + "".join(lines) + "</g>"


def edge_path(
    a: dict[str, float | bool], b: dict[str, float | bool], offset: float
) -> str:
    start_x = float(a["x"]) + 20
    end_x = float(b["x"]) - 20
    start_y = float(a["y"])
    end_y = float(b["y"])
    dx = max(42.0, abs(end_x - start_x) * 0.5)
    return (
        f"M {start_x:.1f} {start_y:.1f} "
        f"C {start_x + dx:.1f} {start_y + offset:.1f}, "
        f"{end_x - dx:.1f} {end_y + offset:.1f}, "
        f"{end_x:.1f} {end_y:.1f}"
    )


def smooth_path(points: list[dict[str, float | bool]]) -> str:
    if not points:
        return ""
    path = f"M {float(points[0]['x']):.1f} {float(points[0]['y']):.1f}"
    for index in range(1, len(points)):
        prev = points[index - 1]
        curr = points[index]
        mid = (float(prev["x"]) + float(curr["x"])) / 2
        path += (
            f" C {mid:.1f} {float(prev['y']):.1f}, "
            f"{mid:.1f} {float(curr['y']):.1f}, "
            f"{float(curr['x']):.1f} {float(curr['y']):.1f}"
        )
    return path


def branch_color(index: int) -> str:
    colors = [
        "#157a5c",
        "#265fbd",
        "#9a5b00",
        "#6750a4",
        "#b42318",
        "#087990",
        "#6f42c1",
        "#0f766e",
    ]
    return colors[index % len(colors)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a scattered graph gallery.")
    parser.add_argument(
        "--output",
        default="results/scattered_graph_gallery.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--dispersions",
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated dispersion values.",
    )
    parser.add_argument("--samples-per-dispersion", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--num-branches", type=int, default=4)
    parser.add_argument("--branch-depth", type=int, default=4)
    parser.add_argument("--distractors-per-node", type=int, default=1)
    args = parser.parse_args()

    output = generate_gallery(
        output=Path(args.output),
        dispersions=parse_dispersions(args.dispersions),
        samples_per_dispersion=args.samples_per_dispersion,
        seed=args.seed,
        num_branches=args.num_branches,
        branch_depth=args.branch_depth,
        distractors_per_node=args.distractors_per_node,
    )
    print(f"wrote graph gallery to {output}")


if __name__ == "__main__":
    main()
