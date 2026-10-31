#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./run.sh <reason|summarize|encode|train|test|explain|all|help> [CLI arguments]

reason, summarize, encode process the train, val and test splits.
train and test use the prepared features and saved STVA checkpoint.
explain generates GNNExplainer reports for test cases.
Individual steps forward additional arguments to their Python module.
all runs the five steps in order and accepts no additional arguments.

Set CHRONOJURY_LOCAL_ROOT to your local asset directory outside this repository.
Configure model endpoints and CHRONOJURY_MODEL through environment variables.
Activate the installed environment, or set CHRONOJURY_PYTHON to its executable.

Encoder settings:
  CHRONOJURY_ENCODER_BACKEND  roberta (default) or qwen
  CHRONOJURY_ENCODER_PATH     Local weights; defaults to workspace/models/
                             chinese-roberta-wwm-ext-large or your_model_name
  CHRONOJURY_INPUT_DIM        1024 (roberta) or 4096 (qwen), unless overridden

Training settings:
  CHRONOJURY_NUM_AGENTS       Number of juror agents (default: 6)
  CHRONOJURY_STVA_MODE        complete (default), gat_concat, raw_pool or temporal_only
  CHRONOJURY_FEATURES_DIR     Defaults to results/features/<encoder>
  CHRONOJURY_CHECKPOINT_DIR   Defaults to results/checkpoints/<encoder>/<mode>
  CHRONOJURY_EXPLANATIONS_DIR Defaults to results/explanations/<encoder>/<mode>

Examples:
  ./run.sh reason --max-workers 5 --prefetch-cases 2
  ./run.sh train --epochs 1 --batch-size 2
  ./run.sh explain --case-id 1
  ./run.sh all
EOF
}

step="${1:-help}"
if (( $# )); then shift; fi
case "$step" in
  help|-h|--help) usage; exit 0 ;;
  reason|summarize|encode|train|test|explain) ;;
  all)
    if (( $# )); then
      echo "all accepts no extra arguments; configure or run individual steps." >&2
      exit 2
    fi
    ;;
  *) usage >&2; exit 2 ;;
esac

if [[ -z "${CHRONOJURY_LOCAL_ROOT:-}" ]]; then
  echo "Set CHRONOJURY_LOCAL_ROOT to your external local asset directory." >&2
  exit 2
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
local_root="$(realpath -m -- "$CHRONOJURY_LOCAL_ROOT")"
case "$local_root" in
  "$repo_root"|"$repo_root"/*)
    echo "CHRONOJURY_LOCAL_ROOT must be outside this repository." >&2
    exit 2
    ;;
esac
python_bin="$(command -v "${CHRONOJURY_PYTHON:-python}")"
if [[ "$python_bin" != /* ]]; then python_bin="$PWD/$python_bin"; fi
workspace="$local_root/workspace"
results="$local_root/results"
agent_count="${CHRONOJURY_NUM_AGENTS:-6}"
if [[ "$step" == reason || "$step" == train || "$step" == test ]]; then
  previous_arg=""
  for argument in "$@"; do
    if [[ "$previous_arg" == --num-agents ]]; then agent_count="$argument"; fi
    case "$argument" in --num-agents=*) agent_count="${argument#*=}" ;; esac
    previous_arg="$argument"
  done
fi
if [[ ! "$agent_count" =~ ^[0-9]+$ ]] || (( agent_count < 2 )); then
  echo "CHRONOJURY_NUM_AGENTS must be an integer of at least 2." >&2
  exit 2
fi
export CHRONOJURY_NUM_AGENTS="$agent_count"
run_results="$results"
network_dir="$workspace/following_networks"
if (( agent_count != 6 )); then
  run_results="$results/agents_$agent_count"
  network_dir="$workspace/following_networks_$agent_count"
fi
backend="${CHRONOJURY_ENCODER_BACKEND:-roberta}"
case "$backend" in
  roberta) encoder_name=chinese-roberta-wwm-ext-large; default_dim=1024; projection_flag=--no-gat-input-projection ;;
  qwen) encoder_name=your_model_name; default_dim=4096; projection_flag=--gat-input-projection ;;
  *) echo "CHRONOJURY_ENCODER_BACKEND must be roberta or qwen." >&2; exit 2 ;;
esac
encoder_path="$(realpath -m -- "${CHRONOJURY_ENCODER_PATH:-$workspace/models/$encoder_name}")"
input_dim="${CHRONOJURY_INPUT_DIM:-$default_dim}"
stva_mode="${CHRONOJURY_STVA_MODE:-complete}"
if [[ "$step" == train || "$step" == test ]]; then
  previous_arg=""
  for argument in "$@"; do
    if [[ "$previous_arg" == --ablation ]]; then stva_mode="$argument"; fi
    case "$argument" in --ablation=*) stva_mode="${argument#*=}" ;; esac
    previous_arg="$argument"
  done
fi
case "$stva_mode" in
  complete|gat_concat|raw_pool|temporal_only) ;;
  *) echo "CHRONOJURY_STVA_MODE must be complete, gat_concat, raw_pool or temporal_only." >&2; exit 2 ;;
esac
features_dir="$(realpath -m -- "${CHRONOJURY_FEATURES_DIR:-$run_results/features/$backend}")"
checkpoint_dir="$(realpath -m -- "${CHRONOJURY_CHECKPOINT_DIR:-$run_results/checkpoints/$backend/$stva_mode}")"
explanations_dir="$(realpath -m -- "${CHRONOJURY_EXPLANATIONS_DIR:-$run_results/explanations/$backend/$stva_mode}")"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$workspace" "$results"
cd "$results"

run_step() {
  local command="$1"
  shift
  case "$command" in
    reason|summarize|encode)
      local split
      for split in train val test; do
        case "$command" in
          reason)
            "$python_bin" -m chronojury.swer.cli \
              --input-dir "$workspace/inputs/$split" \
              --output-dir "$run_results/Results/$split" --work-dir "$workspace" \
              --num-agents "$agent_count" --network-dir "$network_dir" \
              --max-workers 5 --prefetch-cases 2 "$@"
            ;;
          summarize)
            "$python_bin" -m chronojury.swer.case_summary \
              --data-dir "$workspace/inputs/$split" --segment-dir "$workspace/segment" \
              --output-dir "$results/stage_summaries/$split" "$@"
            ;;
          encode)
            "$python_bin" -m chronojury.stva.encoder \
              --backend "$backend" --model-path "$encoder_path" \
              --results-dir "$run_results/Results/$split" \
              --summaries-dir "$results/stage_summaries/$split" \
              --output-dir "$features_dir/$split" "$@"
            ;;
        esac
      done
      ;;
    train)
      "$python_bin" -m chronojury.stva.cli --mode train --ablation "$stva_mode" \
        --input-dim "$input_dim" "$projection_flag" --train-dir "$features_dir/train" \
        --val-dir "$features_dir/val" --test-dir "$features_dir/test" \
        --num-agents "$agent_count" --network-dir "$network_dir" \
        --save-dir "$checkpoint_dir" "$@"
      ;;
    test)
      "$python_bin" -m chronojury.stva.cli --mode test --ablation "$stva_mode" \
        --input-dim "$input_dim" "$projection_flag" --test-dir "$features_dir/test" \
        --num-agents "$agent_count" --network-dir "$network_dir" \
        --model-path "$checkpoint_dir/latest_best.pth" "$@"
      ;;
    explain)
      "$python_bin" -m chronojury.explanation.explainer \
        --checkpoint "$checkpoint_dir/latest_best.pth" \
        --features-dir "$features_dir/test" --network-dir "$network_dir" \
        --output-dir "$explanations_dir" "$@"
      ;;
  esac
}

if [[ "$step" == all ]]; then
  for step in reason summarize encode train test; do run_step "$step"; done
else
  run_step "$step" "$@"
fi
