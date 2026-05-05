# CRAG Baseline on MAEDA Benchmark

This directory extends the original CRAG (Corrective RAG) implementation to support the **MAEDA benchmark** (OpenROAD EDA domain QA).

## What is CRAG?

CRAG (Corrective Retrieval-Augmented Generation) improves RAG by assessing the quality of retrieved documents before generation:

1. **Retrieve** passages for the query
2. **Evaluate** retrieval quality with a T5-based evaluator (correct / ambiguous / incorrect)
3. **Refine** knowledge based on evaluation:
   - **Correct**: Extract relevant strips from retrieved docs
   - **Ambiguous**: Combine multiple knowledge sources
   - **Incorrect**: Use alternative knowledge (web search in original; BGE top passages in MAEDA)
4. **Generate** answer using the refined knowledge

## MAEDA Integration

### Key Differences from Original CRAG

| Component | Original CRAG | MAEDA CRAG |
|-----------|--------------|------------|
| Retriever | Contriever / BM25 | Fine-tuned BGE-large-en-v1.5 + FAISS |
| External knowledge | Web search (Google) | BGE top passages (no web search for OpenROAD domain) |
| Generator | Self-RAG / Llama2 | Qwen1.5-14B-Chat (fine-tuned on OpenROAD) |
| Evaluator | T5 (gsiresearch/t5-large-compact-v1) | Same T5 evaluator |
| Task | PopQA, PubQA, etc. | OpenROAD EDA QA |

### Why BGE Retrieval?

Same as Self-RAG baseline: using the benchmark's pre-retrieved `reranked_knowledge` would be **cheating**. We use independently retrieved BGE passages for fair evaluation.

### File Structure

```
baselines/CRAG/
├── retrieve_with_bge.py          # Step 1: BGE embedding + FAISS retrieval
├── convert_maeda_to_crag.py      # Step 2: Convert MAEDA benchmark → CRAG input format
├── crag_maeda_inference.py       # Step 3: CRAG inference (T5 evaluator + Qwen1.5-14B generator)
├── postprocess_crag_output.py    # Step 4: Convert CRAG output → MAEDA evaluator input
├── eval_config_crag.json         # Step 5: MAEDA evaluator configuration
├── run_crag_maeda.sh             # Main orchestration script (all 5 steps)
├── maeda_crag_data/              # Generated data directory
│   ├── bge_retrieval_results.json  # BGE retrieval results (from Step 1)
│   ├── crag_queries.txt            # Questions (one per line)
│   ├── crag_retrieved.txt          # Retrieved passages (one per question)
│   ├── crag_eval_input.txt         # T5 evaluator input
│   └── crag_gt.json                # Ground truth data for evaluation
├── results/                      # Inference & evaluation results
│   ├── crag_raw_results.json       # Raw CRAG inference output
│   ├── crag_maeda_eval_input.json  # Post-processed output
│   └── crag_maeda_eval_result.json # Final evaluation results
├── scripts/                      # Original CRAG scripts
│   ├── CRAG_Inference.py
│   ├── internal_knowledge_preparation.py
│   ├── external_knowledge_preparation.py
│   └── ...
├── data/                         # Original CRAG example data
└── README.md                     # Original CRAG README
```

### Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1. Retrieve | `retrieve_with_bge.py` | Loads OpenROAD corpus, builds/loads FAISS index with BGE embeddings, retrieves top-k passages per query |
| 2. Convert | `convert_maeda_to_crag.py` | Parses MAEDA benchmark JSON, uses BGE retrieval results, outputs CRAG-format files |
| 3. Inference | `crag_maeda_inference.py` | T5 evaluator scores retrieval quality, selects/refines knowledge, Qwen1.5-14B-Chat generates answers |
| 4. Post-process | `postprocess_crag_output.py` | Cleans output, structures as MAEDA evaluator input |
| 5. Evaluate | `run_eval.py` (via shell script) | Runs MAEDA evaluator using external LLM API to judge answer quality |

### BGE Retrieval Details

We use the **fine-tuned BGE-large-en-v1.5** model (same as Self-RAG baseline):

- Base model: `BAAI/bge-large-en-v1.5` (335M parameters, 1024-dim embeddings)
- Fine-tuned model path: `/mnt/public/sichuan_a/nyt/models/RAG-EDA/models/finetuned-models/embedding/bge-large-en-v1.5/output_flagembedding`
- FAISS index is shared with Self-RAG baseline at `resources/faiss_bge_selfrag/`
- Query instruction prefix: `"Represent this sentence for searching relevant passages: "`

### Generator

- **Qwen1.5-14B-Chat** (fine-tuned on OpenROAD domain data, step-2 merged)
- Model path: `/mnt/public/sichuan_a/nyt/models/RAG-EDA/models/finetuned-models/generator/Qwen1.5-14B-Chat/fine-tuned-model-step2-merged`
- Uses chat template (`<|im_start|>system/user/assistant`) for prompt formatting
- Runs with vLLM for efficient inference

### T5 Evaluator

- Model: `gsiresearch/t5-large-compact-v1` (from original CRAG paper)
- Scores each retrieved document as correct / ambiguous / incorrect
- Thresholds: `upper_threshold=0.592` (≥ → correct), `lower_threshold=-0.995` (≥ → ambiguous)
- Also used for knowledge refinement (selecting relevant strips from passages)

## Usage

### Prerequisites

1. **Conda environment**: `conda activate huada_docqa_demo_release_v1` (includes `vllm`, `torch`, `transformers`, `sentence_transformers`, `faiss`)
2. **Generator model**: `/mnt/public/sichuan_a/nyt/models/RAG-EDA/models/finetuned-models/generator/Qwen1.5-14B-Chat/fine-tuned-model-step2-merged`
3. **BGE embedding model**: `/mnt/public/sichuan_a/nyt/models/RAG-EDA/models/finetuned-models/embedding/bge-large-en-v1.5/output_flagembedding`
4. **T5 evaluator**: Will auto-download `gsiresearch/t5-large-compact-v1` from HuggingFace (or set `EVALUATOR_PATH` env var to local path)
5. **GPU**: At least 30GB free VRAM for Qwen1.5-14B inference + T5 evaluator

### Run full pipeline

```bash
conda activate huada_docqa_demo_release_v1
CUDA_VISIBLE_DEVICES=4 bash run_crag_maeda.sh
```

### Run individual steps

```bash
bash run_crag_maeda.sh retrieve     # Step 1 only (BGE retrieval)
bash run_crag_maeda.sh convert      # Step 2 only (data conversion)
bash run_crag_maeda.sh inference    # Step 3 only (CRAG inference)
bash run_crag_maeda.sh postprocess  # Step 4 only
bash run_crag_maeda.sh eval         # Step 5 only
```

### Use a different GPU

```bash
CUDA_VISIBLE_DEVICES=2 bash run_crag_maeda.sh
```

### Use a local T5 evaluator model

```bash
EVALUATOR_PATH=/path/to/local/t5-model CUDA_VISIBLE_DEVICES=4 bash run_crag_maeda.sh
```

### Run with simple RAG (no corrective evaluation)

```bash
# Edit METHOD in run_crag_maeda.sh or override:
sed -i 's/METHOD="crag"/METHOD="rag"/' run_crag_maeda.sh
CUDA_VISIBLE_DEVICES=4 bash run_crag_maeda.sh
```

## Key Configuration

- **`run_crag_maeda.sh`**: Controls `MODEL_PATH`, `EVALUATOR_PATH`, `BGE_MODEL_PATH`, `CORPUS_PATH`, `FAISS_INDEX_DIR`, `NDOCS`, `RETRIEVAL_TOPK`, `MAX_NEW_TOKENS`, `METHOD`, `UPPER_THRESHOLD`, `LOWER_THRESHOLD`, `DECOMPOSE_MODE`
- **`eval_config_crag.json`**: Controls evaluator LLM API (`model`, `base_url`, `api_key`), input/output paths
- **`convert_maeda_to_crag.py`**: Accepts `--retrieval_results` for BGE output; without it, falls back to `reranked_knowledge` (CHEATING)

## CRAG vs Self-RAG Baseline

| Feature | Self-RAG | CRAG |
|---------|----------|------|
| Retrieval evaluation | Self-reflection tokens | T5 external evaluator |
| Knowledge refinement | Adaptive retrieval during generation | Pre-generation knowledge assessment |
| Generator | selfrag_llama2_7b | Qwen1.5-14B-Chat (fine-tuned) |
| Model size | 7B | 14B |
| Retrieval timing | During generation (adaptive) | Before generation (corrective) |
