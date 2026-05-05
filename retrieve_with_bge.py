#!/usr/bin/env python3
"""
BGE-based retrieval for CRAG on MAEDA benchmark.

Uses the fine-tuned BGE-large-en-v1.5 embedding model to:
1. Build a FAISS index over the OpenROAD knowledge corpus
2. Retrieve top-k relevant passages for each query

This replaces the original CRAG web search retrieval and avoids
using the benchmark's pre-retrieved `reranked_knowledge` (which would
be cheating in evaluation).
"""

import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm


def load_knowledge_corpus(corpus_path):
    """Load OpenROAD knowledge corpus and flatten into passage list."""
    with open(corpus_path, encoding="utf-8") as f:
        doc_list = json.load(f)

    passages = []
    for doc in doc_list:
        source = doc["source"]
        for knowledge in doc["knowledge"]:
            doc_id = knowledge["doc_id"]
            content = knowledge["content"]
            summary = knowledge.get("summary", "")
            if summary and len(summary) > 0:
                text = f"{summary}\n\n{content}"
            else:
                text = content
            passages.append({
                "id": doc_id,
                "title": source,
                "text": text,
            })
    return passages


def build_faiss_index(passages, embed_model, batch_size=64, save_path=None):
    """Build FAISS index from passages using BGE embeddings."""
    import faiss

    texts = [p["text"] for p in passages]
    ids = [p["id"] for p in passages]

    print(f"Encoding {len(texts)} passages with BGE...")
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        embeddings = embed_model.encode(batch, show_progress_bar=False,
                                        normalize_embeddings=True)
        all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings).astype("float32")
    print(f"Embeddings shape: {all_embeddings.shape}")

    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        faiss.write_index(index, os.path.join(save_path, "index.faiss"))
        meta = [{"id": p["id"], "title": p["title"], "text": p["text"]} for p in passages]
        with open(os.path.join(save_path, "passages.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"Index and passages saved to {save_path}")

    return index, passages


def load_faiss_index(index_path):
    """Load pre-built FAISS index and passage metadata."""
    import faiss

    index = faiss.read_index(os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "passages.json"), encoding="utf-8") as f:
        passages = json.load(f)
    print(f"Loaded FAISS index with {index.ntotal} vectors, {len(passages)} passages")
    return index, passages


def retrieve_for_queries(queries, index, passages, embed_model, top_k=10, batch_size=64):
    """Retrieve top-k passages for each query."""
    print(f"Encoding {len(queries)} queries...")
    query_instruction = "Represent this sentence for searching relevant passages: "
    all_query_embeddings = []
    for i in tqdm(range(0, len(queries), batch_size)):
        batch = queries[i:i + batch_size]
        batch_with_instr = [query_instruction + q for q in batch]
        embeddings = embed_model.encode(batch_with_instr, show_progress_bar=False,
                                        normalize_embeddings=True)
        all_query_embeddings.append(embeddings)

    query_embeddings = np.vstack(all_query_embeddings).astype("float32")

    print(f"Searching top-{top_k} for each query...")
    scores, indices = index.search(query_embeddings, top_k)

    results = []
    for i in range(len(queries)):
        retrieved = []
        for j in range(top_k):
            idx = indices[i][j]
            if idx >= 0:
                retrieved.append({
                    "title": passages[idx]["title"],
                    "text": passages[idx]["text"],
                    "score": float(scores[i][j]),
                })
        results.append(retrieved)

    return results


def main():
    parser = argparse.ArgumentParser(description="BGE retrieval for CRAG on MAEDA")
    parser.add_argument("--corpus", type=str,
                        default="/mnt/public/sichuan_a/nyt/MAEDA/huada-docqa-demo/huada-docqa-demo/resources/knowledge_openroad_MAEDA.json",
                        help="Path to OpenROAD knowledge corpus JSON")
    parser.add_argument("--benchmark", type=str,
                        default=None,
                        help="Path to MAEDA benchmark JSON (for queries)")
    parser.add_argument("--model_name", type=str,
                        default="/mnt/public/sichuan_a/nyt/models/RAG-EDA/models/finetuned-models/embedding/bge-large-en-v1.5/output_flagembedding",
                        help="Path to fine-tuned BGE model")
    parser.add_argument("--index_dir", type=str,
                        default=None,
                        help="Directory to save/load FAISS index (default: next to corpus)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of passages to retrieve per query")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file with retrieval results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for embedding model (e.g. cuda:4)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for encoding")
    parser.add_argument("--build_index_only", action="store_true",
                        help="Only build the FAISS index, don't retrieve")
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.index_dir is None:
        corpus_dir = os.path.dirname(args.corpus)
        args.index_dir = os.path.join(corpus_dir, "faiss_bge_selfrag")

    from sentence_transformers import SentenceTransformer
    print(f"Loading BGE model from {args.model_name} on {device}...")
    embed_model = SentenceTransformer(args.model_name, device=device)
    embed_model.encode(["warmup"], normalize_embeddings=True)

    index_path = os.path.join(args.index_dir, "index.faiss")
    if os.path.exists(index_path):
        index, passages = load_faiss_index(args.index_dir)
    else:
        passages = load_knowledge_corpus(args.corpus)
        index, passages = build_faiss_index(passages, embed_model,
                                            batch_size=args.batch_size,
                                            save_path=args.index_dir)

    if args.build_index_only:
        print("Index built. Exiting.")
        return

    if args.benchmark is None:
        print("Error: --benchmark required for retrieval (or use --build_index_only)")
        return

    with open(args.benchmark, encoding="utf-8") as f:
        benchmark_data = json.load(f)

    queries = [item["question"] for item in benchmark_data]
    print(f"Loaded {len(queries)} queries from benchmark")

    retrieval_results = retrieve_for_queries(queries, index, passages,
                                              embed_model, top_k=args.top_k,
                                              batch_size=args.batch_size)

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.benchmark),
                                    "crag_bge_retrieval_results.json")

    output_data = []
    for i, item in enumerate(benchmark_data):
        output_data.append({
            "query_id": item.get("query_id", i),
            "question": item["question"],
            "retrieved_ctxs": retrieval_results[i],
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Retrieval results saved to {args.output}")

    avg_scores = [np.mean([c["score"] for c in ctxs]) for ctxs in retrieval_results if ctxs]
    if avg_scores:
        print(f"Average retrieval score: {np.mean(avg_scores):.4f}")


if __name__ == "__main__":
    main()
