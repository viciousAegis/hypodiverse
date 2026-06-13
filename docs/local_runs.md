# Local Runs

There are two local paths:

```text
scattered-discovery-eval    legacy scattered-causal eval configs, useful for Qwen/Ollama baselines
scattered-discovery-local   generic interactive EnvSpec smoke test for any env family
scattered-discovery-viewer  browser viewer for the scattered-causal world and verifier
```

## Ollama Setup

Start Ollama with repo-local state:

```bash
HOME="$PWD/.ollama-home" \
OLLAMA_MODELS="$PWD/.ollama-models" \
ollama serve
```

Pull models:

```bash
HOME="$PWD/.ollama-home" OLLAMA_MODELS="$PWD/.ollama-models" ollama pull qwen3:1.7b
HOME="$PWD/.ollama-home" OLLAMA_MODELS="$PWD/.ollama-models" ollama pull qwen3:4b
```

## Qwen3 Thinking Behavior

The Ollama backend keeps thinking enabled by default. If Qwen returns thinking but an empty final content field, the local eval runner can make a short no-thinking finalizer call using the hidden thinking trace and ask for exactly one DSL action.

Relevant config fields:

```yaml
model:
  think: true
  num_predict: 2048
  finalize_empty_content: true
  finalizer_num_predict: 160
  finalizer_max_thinking_chars: 12000
```

## Scattered-Causal Baselines

Smoke test with Qwen3 1.7B:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-eval \
  --config configs/local_eval/scattered_smoke_qwen3_1_7b.yaml
```

Quick Qwen3 4B single-answer baseline:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-eval \
  --config configs/local_eval/scattered_qwen3_4b_single_quick.yaml \
  --transcripts
```

Quick Qwen3 4B set-answer baseline:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-eval \
  --config configs/local_eval/scattered_qwen3_4b_set_quick.yaml \
  --transcripts
```

Outputs go under `results/<run_name>/`:

```text
config_source.txt
episodes.jsonl
summary.json
```

## Generic Interactive Smoke Tests

Run a fixed HypoSpace causal spec:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-local \
  --spec configs/envspecs/hypospace_causal_single.json \
  --model qwen3:1.7b \
  --num-predict 1024 \
  --transcripts
```

Run Boolean or 3D specs:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-local \
  --spec configs/envspecs/hypospace_boolean_single.json \
  --model qwen3:1.7b

UV_CACHE_DIR=.uv-cache uv run scattered-discovery-local \
  --spec configs/envspecs/hypospace_3d_single.json \
  --model qwen3:1.7b
```

Use these local runs to check prompt/action syntax. They are not substitutes for veRL training rollouts.

## Scattered-Causal Viewer

Run the local viewer:

```bash
UV_CACHE_DIR=.uv-cache uv run scattered-discovery-viewer --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

The viewer uses the real `ScatteredDiscoveryEnv`. The public-state panel matches
the default model-facing state: raw measurements and sample counts, without
accepted/rejected claim labels. The candidate table and graph overlays are human
diagnostics and still show evidence status and verifier labels:

```text
valid final now     true final answer with enough gathered evidence
final unsupported   true final answer without enough gathered evidence
true non-final      true intermediate claim, incomplete as a final answer
false               not true in the hidden world
```

Generate a static gallery of worlds across dispersion values:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run --no-sync python -m scattered_discovery.graph_gallery \
  --output results/scattered_graph_gallery.html \
  --dispersions 0.0,0.25,0.5,0.75,1.0 \
  --samples-per-dispersion 2
```

The default gallery keeps graph shape fixed at `num_branches=4`,
`branch_depth=4`, and `distractors_per_node=1` so the effect of dispersion is
easy to see.
