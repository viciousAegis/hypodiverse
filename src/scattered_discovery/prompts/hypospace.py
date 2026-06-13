from __future__ import annotations


def causal_initial_prompt(
    *,
    nodes: tuple[str, ...],
    max_edges: int | None,
    query_budget: int,
) -> str:
    return (
        "You are solving an interactive causal-graph HypoSpace task.\n"
        f"Nodes: {', '.join(nodes)}.\n"
        f"Graph constraint: directed acyclic graph with at most {max_edges} edges.\n"
        f"Query budget: {query_budget} interventions.\n\n"
        "Action schemas:\n"
        "- ACTION: INTERVENE <node>\n"
        "- ACTION: COMMIT graph(<source>-><target>, ...)\n"
        "- ACTION: COMMIT [graph(...); graph(...)]\n\n"
        "An intervention returns which nodes change downstream of the perturbed node. "
        "Replace placeholders with listed node names. "
        "Commit only after gathering evidence. Return exactly one ACTION line."
    )


def boolean_initial_prompt(
    *,
    variables: tuple[str, ...],
    operators: tuple[str, ...],
    max_depth: int,
    query_budget: int,
) -> str:
    return (
        "You are solving an interactive Boolean-function HypoSpace task.\n"
        f"Variables: {', '.join(variables)}.\n"
        f"Allowed operators: {', '.join(sorted(operators))}.\n"
        f"Maximum expression depth: {max_depth}.\n"
        f"Query budget: {query_budget} input-output queries.\n\n"
        "Action schemas:\n"
        "- ACTION: QUERY <var>=0,<var>=1\n"
        "- ACTION: COMMIT expr(<expression>)\n"
        "- ACTION: COMMIT [expr(...); expr(...)]\n\n"
        "A query returns the output bit for that assignment. Commit only expressions "
        "over the listed variables and allowed operators. Return exactly one ACTION line."
    )


def reconstruction_3d_initial_prompt(
    *,
    grid_size: int,
    max_height: int,
    max_blocks: int | None,
    query_budget: int,
) -> str:
    return (
        "You are solving an interactive 3D reconstruction HypoSpace task.\n"
        f"Grid size: {grid_size}x{grid_size}. Max height: {max_height}.\n"
        f"Max occupied cells: {max_blocks}. Query budget: {query_budget}.\n\n"
        "Action schemas:\n"
        "- ACTION: VIEW top\n"
        "- ACTION: VIEW front\n"
        "- ACTION: VIEW side\n"
        "- ACTION: PROBE row=0,col=1\n"
        "- ACTION: COMMIT heights([<row>; <row>])\n"
        "- ACTION: COMMIT [heights(...)| heights(...)]\n\n"
        "Heights encode supported vertical stacks. Return exactly one ACTION line."
    )
