#!/bin/bash
# Run CRAG baseline on MAEDA benchmark
# Pipeline: retrieve → convert → inference → postprocess → eval

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ======================== Environment ========================
# Activate uv venv (or conda env as fallback)
if [ -d "/mnt/public/sichuan_a/nyt/uv_envs/.venv_crag_maeda" ]; then
    source /mnt/public/sichuan_a/nyt/uv_envs/.venv_crag_maeda/bin/activate
    echo "Activated uv venv: .venv_crag_maeda"
elif [ -n "$(conda env list 2>/dev/null | grep huada_docqa_demo_release_v1)" ]; then
    conda activate huada_docqa_demo_release_v1
    echo "Activated conda env: huada_docqa_demo_release_v1"
else
    echo "WARNING: No suitable environment found. Install with:"
    echo "  cd /mnt/public/sichuan_a/nyt/uv_envs && uv venv .venv_crag_maeda --python 3.10"
    echo "  uv pip install -p .venv_crag_maeda/bin/python -r requirements_crag_maeda.txt"
fi

# ======================== Configuration ========================
MODEL_PATH="/mnt/public/sichuan_a/nyt/models/RAG-EDA/models/finetuned-models/generator/Qwen1.5-14B-Chat/fine-tuned-model-step2-merged"
EVALUATOR_PATH="${EVALUATOR_PATH:-/mnt/public/sichuan_a/nyt/models/RAG-EDA/models/finetuned-models/reranker/bge-reranker-large/output_flagembedding}"
BGE_MODEL_PATH="/mnt/public/sichuan_a/nyt/models/RAG-EDA/models/finetuned-models/embedding/bge-large-en-v1.5/output_flagembedding"
CORPUS_PATH="/mnt/public/sichuan_a/nyt/MAEDA/huada-docqa-demo/huada-docqa-demo/resources/knowledge_openroad_MAEDA.json"
FAISS_INDEX_DIR="/mnt/public/sichuan_a/nyt/MAEDA/huada-docqa-demo/huada-docqa-demo/resources/faiss_bge_selfrag"
BENCHMARK_PATH="../../MAEDA-benchmark-300.json"

NDOCS=10
RETRIEVAL_TOPK=10
MAX_NEW_TOKENS=512
METHOD="crag"
UPPER_THRESHOLD=0.5
LOWER_THRESHOLD=-1.0
DECOMPOSE_MODE="selection"

DATA_DIR="./maeda_crag_data"
RESULTS_DIR="./results"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

# ======================== Functions ========================
check_vllm() {
    python3 -c "import vllm; print(f'vllm {vllm.__version__}')" 2>/dev/null || {
        echo "ERROR: vllm not found. Activate conda env: conda activate huada_docqa_demo_release_v1"
        exit 1
    }
}

step_retrieve() {
    echo "=== Step 1: BGE Retrieval ==="
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 retrieve_with_bge.py \
        --corpus "$CORPUS_PATH" \
        --benchmark "$BENCHMARK_PATH" \
        --model_name "$BGE_MODEL_PATH" \
        --index_dir "$FAISS_INDEX_DIR" \
        --top_k "$RETRIEVAL_TOPK" \
        --output "$DATA_DIR/bge_retrieval_results.json"
    echo "Step 1 done."
}

step_convert() {
    echo "=== Step 2: Convert MAEDA → CRAG format ==="
    python3 convert_maeda_to_crag.py \
        --input "$BENCHMARK_PATH" \
        --output_dir "$DATA_DIR" \
        --ndocs "$NDOCS" \
        --retrieval_results "$DATA_DIR/bge_retrieval_results.json"
    echo "Step 2 done."
}

step_inference() {
    echo "=== Step 3: CRAG Inference (Qwen1.5-14B + T5 Evaluator) ==="
    check_vllm
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_USE_V1=0 \
    python3 crag_maeda_inference.py \
        --generator_path "$MODEL_PATH" \
        --evaluator_path "$EVALUATOR_PATH" \
        --input_file "$BENCHMARK_PATH" \
        --retrieval_results "$DATA_DIR/bge_retrieval_results.json" \
        --output_file "$RESULTS_DIR/crag_raw_results.json" \
        --method "$METHOD" \
        --ndocs "$NDOCS" \
        --upper_threshold "$UPPER_THRESHOLD" \
        --lower_threshold "$LOWER_THRESHOLD" \
        --decompose_mode "$DECOMPOSE_MODE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --device "cuda:0"
    echo "Step 3 done."
}

step_postprocess() {
    echo "=== Step 4: Post-process → MAEDA evaluator input ==="
    python3 postprocess_crag_output.py \
        --raw_results "$RESULTS_DIR/crag_raw_results.json" \
        --gt_data "$DATA_DIR/crag_gt.json" \
        --output "$RESULTS_DIR/crag_maeda_eval_input.json"
    echo "Step 4 done."
}

step_eval() {
    echo "=== Step 5: MAEDA Evaluation ==="
    if [ ! -f "$RESULTS_DIR/crag_maeda_eval_input.json" ]; then
        echo "Error: $RESULTS_DIR/crag_maeda_eval_input.json not found. Run 'postprocess' first."
        exit 1
    fi

    EVAL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
    # Resolve absolute paths for input/output
    ABS_INPUT="$(cd "$(dirname "$RESULTS_DIR/crag_maeda_eval_input.json")" && pwd)/$(basename "$RESULTS_DIR/crag_maeda_eval_input.json")"
    ABS_OUTPUT="$(cd "$(dirname "$RESULTS_DIR/crag_maeda_eval_result.json")" 2>/dev/null && pwd)/$(basename "$RESULTS_DIR/crag_maeda_eval_result.json")" || ABS_OUTPUT="$RESULTS_DIR/crag_maeda_eval_result.json"

    cd "$EVAL_ROOT"
    python3 run_eval.py \
        --config "$SCRIPT_DIR/eval_config_crag.json" \
        --input_path "$ABS_INPUT" \
        --gt_path "$ABS_INPUT" \
        --output_path "$ABS_OUTPUT"
    cd "$SCRIPT_DIR"
    echo "Step 5 done."
}

# ======================== Main ========================
mkdir -p "$DATA_DIR" "$RESULTS_DIR"

STEP="${1:-all}"

case "$STEP" in
    retrieve)   step_retrieve ;;
    convert)    step_convert ;;
    inference)  step_inference ;;
    postprocess) step_postprocess ;;
    eval)       step_eval ;;
    all)
        step_retrieve
        step_convert
        step_inference
        step_postprocess
        step_eval
        ;;
    *)
        echo "Usage: $0 {retrieve|convert|inference|postprocess|eval|all}"
        echo "  all         - Run full pipeline (default)"
        echo "  retrieve    - Step 1: BGE retrieval only"
        echo "  convert     - Step 2: Data conversion only"
        echo "  inference   - Step 3: CRAG inference only"
        echo "  postprocess - Step 4: Post-processing only"
        echo "  eval        - Step 5: MAEDA evaluation only"
        exit 1
        ;;
esac

echo "=== Done: $STEP ==="
