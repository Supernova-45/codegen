#!/usr/bin/env bash
set -euo pipefail

# Run multi-shard full experiments, then merge + summarize.
# Usage:
#   bash scripts/run_full_6shards.sh mbpp
#   bash scripts/run_full_6shards.sh humaneval
#   bash scripts/run_full_6shards.sh both
#   bash scripts/run_full_6shards.sh mbpp eig-only
#   bash scripts/run_full_6shards.sh both eig-only

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TARGET="${1:-both}"
STRATEGY_PROFILE="${2:-all}"
NUM_SHARDS="${NUM_SHARDS:-8}"
RESUME_SHARDS="${RESUME_SHARDS:-1}"
SKIP_MBPPPLUS="${SKIP_MBPPPLUS:-1}"
SKIP_EXISTING_FILE="${SKIP_EXISTING_FILE:-}"

case "${STRATEGY_PROFILE}" in
  all)
    STRATEGIES=(
      one-shot
      random-tests
      eig-tests
      self-consistency
      repair
    )
    ;;
  eig-only)
    STRATEGIES=(
      eig-tests
    )
    ;;
  *)
    echo "Unknown strategy profile: ${STRATEGY_PROFILE}"
    echo "Usage: bash scripts/run_full_6shards.sh [mbpp|humaneval|both] [all|eig-only]"
    exit 2
    ;;
esac

run_dataset() {
  local dataset="$1"
  local config="$2"
  local merged_output="$3"
  local summary_dir="$4"
  local shard_prefix="$5"

  echo ""
  echo "== Running ${dataset} with ${NUM_SHARDS} shards =="
  echo "== Strategies: ${STRATEGIES[*]} =="
  mkdir -p results/shards

  local pids=()
  for ((i=0; i<NUM_SHARDS; i++)); do
    local shard_out="shards/${shard_prefix}_shard${i}.jsonl"
    echo "Launching shard ${i}/${NUM_SHARDS} -> results/${shard_out}"
    local resume_flag=()
    if [[ "${RESUME_SHARDS}" == "1" ]]; then
      resume_flag=(--resume)
    fi
    local mbppplus_flag=()
    if [[ "${SKIP_MBPPPLUS}" == "1" ]]; then
      mbppplus_flag=(--skip-mbppplus)
    fi
    local skip_existing_flag=()
    if [[ -n "${SKIP_EXISTING_FILE}" ]]; then
      skip_existing_flag=(--skip-existing-files "${SKIP_EXISTING_FILE}")
    fi
    python scripts/run_experiment.py \
      --config "${config}" \
      --strategies "${STRATEGIES[@]}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${i}" \
      --output-file "${shard_out}" \
      "${mbppplus_flag[@]}" \
      "${skip_existing_flag[@]}" \
      "${resume_flag[@]}" &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "One or more shards failed for ${dataset}."
    exit 1
  fi

  echo "Merging shards -> ${merged_output}"
  : > "${merged_output}"
  for ((i=0; i<NUM_SHARDS; i++)); do
    cat "results/shards/${shard_prefix}_shard${i}.jsonl" >> "${merged_output}"
  done

  echo "Summarizing ${merged_output} -> ${summary_dir}"
  python scripts/summarize_results.py \
    --results "${merged_output}" \
    --output-dir "${summary_dir}"

  echo "Done: ${dataset}"
  echo "  merged: ${merged_output}"
  echo "  summary: ${summary_dir}"
}

case "${TARGET}" in
  mbpp)
    run_dataset \
      "MBPP" \
      "configs/mvp_mbpp_eig_vs_random_tuned_full.yaml" \
      "results/mbpp_eig_vs_random_tuned_full.jsonl" \
      "results/mbpp_eig_vs_random_tuned_full_summary" \
      "mbpp_full"
    ;;
  humaneval)
    run_dataset \
      "HumanEval" \
      "configs/mvp_humaneval_eig_vs_random_full.yaml" \
      "results/humaneval_eig_vs_random_full.jsonl" \
      "results/humaneval_eig_vs_random_full_summary" \
      "humaneval_full"
    ;;
  both)
    run_dataset \
      "MBPP" \
      "configs/mvp_mbpp_eig_vs_random_tuned_full.yaml" \
      "results/mbpp_eig_vs_random_tuned_full.jsonl" \
      "results/mbpp_eig_vs_random_tuned_full_summary" \
      "mbpp_full"
    run_dataset \
      "HumanEval" \
      "configs/mvp_humaneval_eig_vs_random_full.yaml" \
      "results/humaneval_eig_vs_random_full.jsonl" \
      "results/humaneval_eig_vs_random_full_summary" \
      "humaneval_full"
    ;;
  *)
    echo "Unknown target: ${TARGET}"
    echo "Usage: bash scripts/run_full_6shards.sh [mbpp|humaneval|both] [all|eig-only]"
    exit 2
    ;;
esac

echo ""
echo "All requested runs completed."
