#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SUFFIX="${RUN_SUFFIX:-lite83_tok25k_mt8_probev3_closed_window_rerun1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export SWE_AGENT_CONFIG_ROOT="$ROOT_DIR"
export MAX_TOKEN_BUDGET="${MAX_TOKEN_BUDGET:-25000}"
export SWE_AGENT_FORCE_STREAM="${SWE_AGENT_FORCE_STREAM:-1}"
export SWE_AGENT_DROP_TEMPERATURE="${SWE_AGENT_DROP_TEMPERATURE:-1}"
export SWE_AGENT_DROP_TOP_P="${SWE_AGENT_DROP_TOP_P:-1}"
export PYTHONPATH="/data/liora/sweagent_deps${PYTHONPATH:+:$PYTHONPATH}"

/home/ubuntu/miniconda3/bin/conda run -n swebenchbrq \
  python -m sweagent.run.run run-batch \
  --config "$ROOT_DIR/config/default.yaml" \
  --config "$ROOT_DIR/config/local_codex.yaml" \
  --config "$ROOT_DIR/config/lite83_run.yaml" \
  --config "$ROOT_DIR/config/lite83_history_compression.yaml" \
  --env_var_path "$ROOT_DIR/.env" \
  --agent.model.name gpt-5.3-codex \
  --agent.model.api_base https://api.tabcode.cc/openai \
  --agent.model.per_instance_cost_limit 0 \
  --agent.model.total_cost_limit 0 \
  --num_workers "$NUM_WORKERS" \
  --suffix "$RUN_SUFFIX" \
  "$@"
