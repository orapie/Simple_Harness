#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-data}"
INDEX_PATH="${INDEX_PATH:-.rag_index/index.json}"
ROLE_PATH="${ROLE_PATH:-configs/roles/default.json}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_BACKEND="${MODEL_BACKEND:-}"
GGUF_N_CTX="${GGUF_N_CTX:-4096}"
GGUF_N_GPU_LAYERS="${GGUF_N_GPU_LAYERS:-0}"
GGUF_CHAT_FORMAT="${GGUF_CHAT_FORMAT:-}"
CHARACTER_ROOT="${CHARACTER_ROOT:-character_system}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-}"
NPC_ID="${NPC_ID:-lu_jiangxian}"
CUTOFF="${CUTOFF:-}"
DEVICE="${DEVICE:-}"
TOP_K="${TOP_K:-5}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-5000}"
MAX_CHARACTER_CHARS="${MAX_CHARACTER_CHARS:-4500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

usage() {
  cat <<'USAGE'
Simple Harness RAG

Usage:
  ./run.sh install
  ./run.sh index
  ./run.sh search "问题"
  ./run.sh prompt "问题"
  ./run.sh chat "问题"
  ./run.sh character-rag-prompt "问题"
  ./run.sh character-rag-chat "问题"
  ./run.sh test

Environment overrides:
  PYTHON_BIN=python3
  DATA_DIR=data
  INDEX_PATH=.rag_index/index.json
  ROLE_PATH=configs/roles/default.json
  EMBEDDING_MODEL=/path/to/local/embedding-model
  MODEL_PATH=/path/to/local/chat-model
  MODEL_BACKEND=transformers|gguf
  GGUF_N_CTX=4096
  GGUF_N_GPU_LAYERS=0
  GGUF_CHAT_FORMAT=
  CHARACTER_ROOT=character_system
  EVIDENCE_ROOT=data/novel_test
  NPC_ID=lu_jiangxian
  CUTOFF=evt-010
  DEVICE=mps|cpu|cuda
  TOP_K=5
  MAX_CONTEXT_CHARS=5000
  MAX_CHARACTER_CHARS=4500
  MAX_NEW_TOKENS=512
  TEMPERATURE=0.7
  TOP_P=0.9

Examples:
  ./run.sh index
  ./run.sh search "世界杯决赛后有什么商业争议？"
  ROLE_PATH=configs/roles/xuan_an.json ./run.sh prompt "请用角色口吻总结资料"
  MODEL_PATH=/path/to/local/chat-model DEVICE=mps ./run.sh chat "解释资料里的票价争议"
  MODEL_PATH=/path/to/model.gguf MODEL_BACKEND=gguf GGUF_N_GPU_LAYERS=35 ./run.sh chat "解释资料里的票价争议"
  NPC_ID=lu_jiangxian CUTOFF=evt-010 ./run.sh character-rag-prompt "用陆江仙的口吻解释世界杯商业化争议"
  NPC_ID=xuan_an CUTOFF=evt-018 MODEL_PATH=/path/to/model.gguf ./run.sh character-rag-chat "结合资料谈谈球迷票价争议"
USAGE
}

require_query() {
  if [[ $# -eq 0 ]]; then
    echo "missing query text" >&2
    usage
    exit 2
  fi
}

model_args=()
if [[ -n "$EMBEDDING_MODEL" ]]; then
  model_args+=(--embedding-model "$EMBEDDING_MODEL")
fi
if [[ -n "$DEVICE" ]]; then
  model_args+=(--device "$DEVICE")
fi

character_args=(--character-root "$CHARACTER_ROOT" --npc "$NPC_ID")
if [[ -n "$EVIDENCE_ROOT" ]]; then
  character_args+=(--evidence-root "$EVIDENCE_ROOT")
fi
if [[ -n "$CUTOFF" ]]; then
  character_args+=(--cutoff "$CUTOFF")
fi

llm_args=()
if [[ -n "$MODEL_BACKEND" ]]; then
  llm_args+=(--model-backend "$MODEL_BACKEND")
fi
if [[ -n "$GGUF_N_CTX" ]]; then
  llm_args+=(--gguf-n-ctx "$GGUF_N_CTX")
fi
if [[ -n "$GGUF_N_GPU_LAYERS" ]]; then
  llm_args+=(--gguf-n-gpu-layers "$GGUF_N_GPU_LAYERS")
fi
if [[ -n "$GGUF_CHAT_FORMAT" ]]; then
  llm_args+=(--gguf-chat-format "$GGUF_CHAT_FORMAT")
fi

command="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$command" in
  help|-h|--help)
    usage
    ;;
  install)
    "$PYTHON_BIN" -m pip install -e .
    ;;
  index)
    "$PYTHON_BIN" -m simple_rag.cli index \
      --data "$DATA_DIR" \
      --index "$INDEX_PATH" \
      "${model_args[@]}"
    ;;
  search)
    require_query "$@"
    "$PYTHON_BIN" -m simple_rag.cli search \
      --index "$INDEX_PATH" \
      --query "$*" \
      --top-k "$TOP_K" \
      "${model_args[@]}"
    ;;
  prompt)
    require_query "$@"
    "$PYTHON_BIN" -m simple_rag.cli prompt \
      --index "$INDEX_PATH" \
      --role "$ROLE_PATH" \
      --query "$*" \
      --top-k "$TOP_K" \
      --max-context-chars "$MAX_CONTEXT_CHARS" \
      "${model_args[@]}"
    ;;
  chat)
    require_query "$@"
    chat_args=()
    if [[ -n "$MODEL_PATH" ]]; then
      chat_args+=(--model "$MODEL_PATH")
    fi
    "$PYTHON_BIN" -m simple_rag.cli chat \
      --index "$INDEX_PATH" \
      --role "$ROLE_PATH" \
      --query "$*" \
      --top-k "$TOP_K" \
      --max-context-chars "$MAX_CONTEXT_CHARS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      "${model_args[@]}" \
      "${llm_args[@]}" \
      "${chat_args[@]}"
    ;;
  character-rag-prompt)
    require_query "$@"
    "$PYTHON_BIN" -m simple_rag.cli character-rag-prompt \
      --index "$INDEX_PATH" \
      --query "$*" \
      --top-k "$TOP_K" \
      --max-context-chars "$MAX_CONTEXT_CHARS" \
      --max-character-chars "$MAX_CHARACTER_CHARS" \
      "${model_args[@]}" \
      "${character_args[@]}"
    ;;
  character-rag-chat)
    require_query "$@"
    chat_args=()
    if [[ -n "$MODEL_PATH" ]]; then
      chat_args+=(--model "$MODEL_PATH")
    fi
    "$PYTHON_BIN" -m simple_rag.cli character-rag-chat \
      --index "$INDEX_PATH" \
      --query "$*" \
      --top-k "$TOP_K" \
      --max-context-chars "$MAX_CONTEXT_CHARS" \
      --max-character-chars "$MAX_CHARACTER_CHARS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      "${model_args[@]}" \
      "${llm_args[@]}" \
      "${character_args[@]}" \
      "${chat_args[@]}"
    ;;
  test)
    "$PYTHON_BIN" -m unittest discover -s tests -v
    "$PYTHON_BIN" -m unittest discover -s character_system/tests -v
    ;;
  *)
    echo "unknown command: $command" >&2
    usage
    exit 2
    ;;
esac
