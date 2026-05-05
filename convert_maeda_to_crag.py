#!/usr/bin/env python3
"""
Convert MAEDA-benchmark-300.json to CRAG input format.

CRAG pipeline expects:
  - queries file: one question per line
  - retrieval file: one set of passages per question (joined by [sep])
  - evaluator input file: question [SEP] passage per line (for T5 evaluator scoring)

MAEDA benchmark format:
  - query_id, question, answer, reranked_knowledge, gt_answer, gt_answer_points, label

Two modes:
  1. With --retrieval_results: Uses BGE-retrieved passages (fair evaluation)
  2. Without --retrieval_results: Falls back to benchmark's reranked_knowledge (CHEATING)

Output:
  1. crag_queries.txt     — one question per line
  2. crag_retrieved.txt   — one set of passages per question (joined by [sep])
  3. crag_eval_input.txt  — question [SEP] passage per line (for T5 evaluator)
  4. crag_gt.json         — for MAEDA evaluator (preserves all original fields)
"""

import json
import re
import argparse
import os


def parse_reranked_knowledge(rk_text: str) -> list:
    """Parse reranked_knowledge string into list of {title, text} dicts."""
    if not rk_text or not rk_text.strip():
        return []

    segments = re.split(r'\n(?=id:)', rk_text.strip())
    ctxs = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        lines = seg.split('\n', 1)
        first_line = lines[0]
        doc_id = first_line.replace('id:', '', 1).strip()
        title_match = re.search(r'###\s*(.+)', seg)
        if title_match:
            title = title_match.group(1).strip()
            remaining = seg[title_match.end():]
        else:
            title = doc_id
            remaining = lines[1] if len(lines) > 1 else ""
        text = remaining.strip()
        ctxs.append({"title": title, "text": text, "id": doc_id})
    return ctxs


def convert(input_path: str, output_dir: str, ndocs: int = 10,
            retrieval_results_path: str = None):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    retrieval_map = None
    if retrieval_results_path:
        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
        retrieval_map = {}
        for item in retrieval_data:
            qid = item.get("query_id", 0)
            retrieval_map[qid] = item["retrieved_ctxs"]
        print(f"Loaded BGE retrieval results for {len(retrieval_map)} queries from {retrieval_results_path}")

    queries = []
    passages = []
    eval_lines = []
    gt_data = []

    for item in benchmark_data:
        question = item["question"]
        query_id = item.get("query_id", 0)
        rk_text = item.get("reranked_knowledge", "")

        if retrieval_map and query_id in retrieval_map:
            ctxs = retrieval_map[query_id][:ndocs]
            retrieval_source = "BGE"
        else:
            ctxs = parse_reranked_knowledge(rk_text)[:ndocs]
            retrieval_source = "reranked_knowledge"

        queries.append(question)

        # Format passages as "title // text" joined by [sep]
        psgs = [f"{c['title']} // {c['text'].strip().replace(chr(10), ' ')}" for c in ctxs]
        passages.append(' [sep] '.join(psgs))

        # Evaluator input: question [SEP] passage per doc
        for p in psgs:
            eval_lines.append(f"{question} [SEP] {p}")

        # GT data for MAEDA evaluator
        gt_item = {
            "query_id": query_id,
            "question": question,
            "gt_answer": item.get("gt_answer", ""),
            "gt_answer_points": item.get("gt_answer_points", []),
            "reranked_knowledge": rk_text,
            "background_knowledge": item.get("background_knowledge", ""),
            "error_type": item.get("label", []),
            "gt_reference_doc_ids": item.get("gt_reference_doc_ids", []),
        }
        gt_data.append(gt_item)

    # Save text files for CRAG pipeline
    queries_path = os.path.join(output_dir, "crag_queries.txt")
    passages_path = os.path.join(output_dir, "crag_retrieved.txt")
    eval_path = os.path.join(output_dir, "crag_eval_input.txt")
    gt_path = os.path.join(output_dir, "crag_gt.json")

    with open(queries_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(queries))
    with open(passages_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(passages))
    with open(eval_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(eval_lines))
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(benchmark_data)} items (retrieval: {retrieval_source})")
    print(f"  Queries:      {queries_path}")
    print(f"  Passages:     {passages_path}")
    print(f"  Eval input:   {eval_path}")
    print(f"  GT data:      {gt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="../../MAEDA-benchmark-300.json",
                        help="Path to MAEDA-benchmark-300.json")
    parser.add_argument("--output_dir", type=str,
                        default="./maeda_crag_data",
                        help="Output directory")
    parser.add_argument("--ndocs", type=int, default=10,
                        help="Number of docs per question to include")
    parser.add_argument("--retrieval_results", type=str, default=None,
                        help="Path to BGE retrieval results JSON. "
                             "If not provided, falls back to benchmark's reranked_knowledge (CHEATING).")
    args = parser.parse_args()
    convert(args.input, args.output_dir, args.ndocs, args.retrieval_results)
