from __future__ import annotations

from scattered_discovery.config import AgentConfig
from scattered_discovery.envs.scattered_causal import ScatteredDiscoveryEnv


SYSTEM_PROMPT = """You are an agent in a synthetic causal discovery environment.
Your job is to run budgeted experiments and then commit valid final causal hypotheses.

You may reason privately, but your final visible response must contain only one action line.
If your model uses <think> tags or a hidden thinking channel, keep that reasoning short, close it,
and put exactly one ACTION line in the final answer content.
Do not put explanations, plans, or multiple actions in the final answer content.

The visible response must be one of:
ACTION: INTERVENE xNN
ACTION: TEST edge(xNN,xMM)
ACTION: COMMIT path(xNN,xMM,xKK,...)
ACTION: COMMIT [path(...); path(...)]

Rules:
- Variables are opaque symbols. Do not invent variables that are not known.
- INTERVENE on a known variable exposes measured downstream effects and may reveal new variables.
- TEST a known edge hypothesis to gather evidence for one adjacent edge.
- Final credit requires a complete path of the required length, backed by evidence for every adjacent edge gathered in this episode.
- Shorter intermediate paths can help you explore, but they are not final answers.
- Prefer exploring multiple promising branches when the budget allows.
- If you have no evidence yet, INTERVENE on a known variable before testing edges.
- To extend a measured edge xA->xB with a strong effect, usually INTERVENE xB next.
- Do not COMMIT until you have a complete path with evidence for each adjacent edge.
- Never COMMIT a path that is shorter than the required final path length.
"""


def _public_state(env: ScatteredDiscoveryEnv, agent_config: AgentConfig) -> str:
    return env.public_state_text(
        agent_config.max_evidence_items,
        include_evidence_status=agent_config.include_evidence_status_in_prompt,
    )


def initial_user_prompt(env: ScatteredDiscoveryEnv, agent_config: AgentConfig) -> str:
    protocol_line = (
        "This run uses single-answer protocol: finish with ACTION: COMMIT path(...)."
        if env.protocol == "single"
        else "This run uses set-answer protocol: finish with ACTION: COMMIT [path(...); path(...)]."
    )
    return (
        f"{protocol_line}\n"
        f"Final-answer hypotheses are directed paths with exactly {env.config.branch_depth + 1} variables.\n"
        "Write no visible reasoning; output only the ACTION line.\n"
        "Current public state:\n"
        f"{_public_state(env, agent_config)}\n\n"
        "Choose the next action. Return only one ACTION line."
    )


def observation_prompt(
    env: ScatteredDiscoveryEnv, observation: str, agent_config: AgentConfig
) -> str:
    return (
        f"Environment observation:\n{observation}\n\n"
        "Updated public state:\n"
        f"{_public_state(env, agent_config)}\n\n"
        f"Required final path length: {env.config.branch_depth + 1} variables.\n"
        "Choose the next action. Return only one ACTION line."
    )


def final_commit_prompt(env: ScatteredDiscoveryEnv, agent_config: AgentConfig) -> str:
    commit_action = (
        "ACTION: COMMIT path(...)"
        if env.protocol == "single"
        else "ACTION: COMMIT [path(...); path(...)]"
    )
    return (
        "Experiment budget or step limit has been reached. Submit your final answer now.\n"
        f"Use {commit_action}.\n\n"
        f"Required final path length: {env.config.branch_depth + 1} variables.\n"
        "Current public state:\n"
        f"{_public_state(env, agent_config)}"
    )


def repair_prompt(
    env: ScatteredDiscoveryEnv, error: str, agent_config: AgentConfig
) -> str:
    return (
        "Your previous response did not contain a syntactically valid action.\n"
        f"Parser error: {error}\n"
        "Do not explain. Output exactly one line matching one of the allowed action formats.\n"
        f"Required final path length: {env.config.branch_depth + 1} variables.\n"
        "Current public state:\n"
        f"{_public_state(env, agent_config)}"
    )


def finalizer_prompt(
    env: ScatteredDiscoveryEnv,
    thinking_trace: str,
    error: str,
    agent_config: AgentConfig,
) -> str:
    return (
        "You already completed a hidden thinking pass, but produced no final action content.\n"
        "Thinking is disabled for this formatting pass. Use the prior thinking trace only to "
        "choose the next action.\n"
        f"Parser status: {error}\n"
        f"Required final path length: {env.config.branch_depth + 1} variables.\n"
        "Current public state:\n"
        f"{_public_state(env, agent_config)}\n\n"
        "Prior hidden thinking trace:\n"
        f"{thinking_trace}\n\n"
        "Return exactly one line now. No explanation. Use ACTION: INTERVENE, ACTION: TEST, "
        "or ACTION: COMMIT as appropriate."
    )
