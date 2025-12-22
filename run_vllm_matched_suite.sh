#!/usr/bin/env bash
set -euo pipefail

# VLLM-matched EAGLE3 benchmark runner
# Tests both normal and converted throughput EAGLE3 checkpoints with VLLM parameters:
# batch_size=8, context_length=8192, max_new_tokens=24576, session_len=32768

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${ROOT_DIR}/results_eagle3_micro_vllm_match"
mkdir -p "${OUTPUT_DIR}"

echo "[VLLM-Matched EAGLE3 Benchmark Suite]"
echo "  ROOT_DIR: ${ROOT_DIR}"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo ""

# Test scenarios
scenarios=(
    # Normal EAGLE3 checkpoint with multiple spec tokens
    "/workspace/aimo/models/gpt-oss-120b-eagle3" "Speculative_Batch8_Context8K_1tokens_EAGLE3_normal_VLLM" 1"
    "/workspace/aimo/models/gpt-oss-120b-eagle3" "Speculative_Batch8_Context8K_3tokens_EAGLE3_normal_VLLM" 3"
    "/workspace/aimo/models/gpt-oss-120b-eagle3" "Speculative_Batch8_Context8K_4tokens_EAGLE3_normal_VLLM" 4"
    "/workspace/aimo/models/gpt-oss-120b-eagle3" "Speculative_Batch8_Context8K_5tokens_EAGLE3_normal_VLLM" 5"
    
    # Converted throughput EAGLE3 checkpoint (only spec=1 for throughput optimization)
    "/workspace/aimo/models/turbomind_eagle_draft_gpt-oss-120b-Eagle3-throughput" "Speculative_Batch8_Context8K_1tokens_EAGLE3_throughput_converted_VLLM" 1"
)

total_scenarios=${#scenarios[@]}
echo "Total scenarios to run: ${total_scenarios}"
echo ""

# Run each scenario
for i in "${!scenarios[@]}"; do
    ((j=i+1))
    spec_model_path="${scenarios[i]}"
    scenario_name="${scenarios[i+1]}"
    num_spec_tokens="${scenarios[i+2]}"
    
    echo "============================================================"
    echo "Scenario ${j}/${total_scenarios}: ${scenario_name}"
    echo "  SPEC_MODEL_PATH: ${spec_model_path}"
    echo "  NUM_SPEC_TOKENS: ${num_spec_tokens}"
    echo "  BATCH_SIZE: 8, CONTEXT_LENGTH: 8192, MAX_NEW_TOKENS: 24576, SESSION_LEN: 32768"
    echo "============================================================"
    
    # Run with modified benchmark_speculative.py to force VLLM parameters
    /workspace/aimo/miniconda/envs/lmdeploy_8da9/bin/python benchmark_speculative.py \
        --model-path "/workspace/aimo/models/gpt-oss-120b" \
        --spec-model-path "${spec_model_path}" \
        --output-dir "${OUTPUT_DIR}" \
        --scenario custom \
        --warmup-runs 1 \
        --measurement-runs 1 \
        --custom-scenario-name "${scenario_name}" \
        --custom-batch-size 8 \
        --custom-context-length 8192 \
        --custom-max-new-tokens 24576 \
        --custom-num-spec-tokens "${num_spec_tokens}" \
        --custom-session-len 32768 2>&1 | tee "/workspace/aimo/logs/vllm_matched_scenario_${j}_$(date +%Y%m%d-%H%M%S).log"
    
    if [ $? -eq 0 ]; then
        echo "✅ COMPLETED: ${scenario_name}"
    else
        echo "❌ FAILED: ${scenario_name}"
    fi
    echo ""
done

echo "============================================================"
echo "All VLLM-matched EAGLE3 benchmarks completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"