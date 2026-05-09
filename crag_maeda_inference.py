#!/usr/bin/env python3
"""
CRAG inference on MAEDA benchmark with Qwen1.5-14B-Chat generator.

CRAG (Corrective RAG) pipeline:
1. Retrieve passages using BGE (pre-computed)
2. Score retrieved passages with BGE-reranker cross-encoder (correct/ambiguous/incorrect)
3. Based on scores:
   - Correct: refine knowledge internally (extract relevant strips)
   - Ambiguous: combine internal + external knowledge
   - Incorrect: use external knowledge (web search → replaced by BGE top passages)
4. Generate answer with Qwen1.5-14B-Chat using the selected knowledge

For MAEDA, "external knowledge" is replaced by the top BGE-retrieved passages
since we don't have web search access for the OpenROAD domain.

The original CRAG uses a T5 evaluator (gsiresearch/t5-large-compact-v1) for retrieval
assessment. Since HuggingFace is unreachable, we use the local fine-tuned
BGE-reranker-large cross-encoder instead, which serves the same purpose.
"""

import argparse
import json
import os
import re
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams


CONTROL_TOKENS = [
    "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
    "[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]",
    "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
]


def clean_output(text: str) -> str:
    """Remove control tokens and clean up the answer."""
    for token in CONTROL_TOKENS:
        text = text.replace(token, "")
    text = text.replace("</s>", "")
    text = text.strip()
    if text and text[0] == " ":
        text = text[1:]
    return text


def extract_strips_from_psg(psg, mode="excerption"):
    """Decompose a passage into strips for knowledge refinement."""
    if mode == 'fixed_num':
        final_strips = []
        window_length = 50
        words = psg.split(' ')
        buf = []
        for w in words:
            buf.append(w)
            if len(buf) == window_length:
                final_strips.append(' '.join(buf))
                buf = []
        if buf:
            if len(buf) < 10:
                final_strips[-1] += (' ' + ' '.join(buf))
            else:
                final_strips.append(' '.join(buf))
        return final_strips
    elif mode == 'excerption':
        num_concatenate_strips = 3
        question_strips = psg.split('?')
        origin_strips = []
        for qs in question_strips:
            origin_strips += qs.split('. ')
        strips = []
        for s in origin_strips:
            if s in strips:
                continue
            if not strips:
                strips.append(s)
            else:
                if len(s.split()) > 5:
                    strips.append(s)
                else:
                    strips[-1] += s
        final_strips = []
        buf = []
        for strip in strips:
            buf.append(strip)
            if len(buf) == num_concatenate_strips:
                final_strips.append(' '.join(buf))
                buf = []
        if buf:
            final_strips.append(' '.join(buf))
        return final_strips
    elif mode == 'selection':
        return [psg]


def select_relevants(strips, query, tokenizer, model, device, top_n=5):
    """Select most relevant strips from a passage using cross-encoder."""
    model = model.to(device)
    max_length = 512
    strips_data = []
    for i, p in enumerate(strips):
        if len(p.split()) < 4:
            scores = -1.0
        else:
            inputs = tokenizer(query, p, return_tensors="pt",
                              padding=True, truncation=True, max_length=max_length)
            try:
                with torch.no_grad():
                    outputs = model(**{k: v.to(device) for k, v in inputs.items()})
                scores = float(outputs.logits[0][0].cpu())
            except Exception:
                scores = -1.0
        strips_data.append((scores, p, i))

    sorted_results = sorted(strips_data, key=lambda x: x[0], reverse=True)
    ctxs = [s[1] for s in sorted_results[:top_n]]
    return '; '.join(ctxs)


def score_retrieved_docs(queries, passages_list, tokenizer, model, device, n_docs):
    """Score each retrieved document using cross-encoder reranker."""
    model.eval()
    scores = []

    for q_idx, (query, passages_text) in enumerate(zip(queries, passages_list)):
        psgs = passages_text.split(' [sep] ')
        for p_idx, psg in enumerate(psgs[:n_docs]):
            if not psg.strip():
                scores.append(-1.0)
                continue
            inputs = tokenizer(query, psg, return_tensors="pt",
                              padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**{k: v.to(device) for k, v in inputs.items()})
            scores.append(float(outputs.logits[0][0].cpu()))

    return scores


def process_flag(scores, n_docs, threshold1, threshold2):
    """Convert scores to identification flags (0=incorrect, 1=ambiguous, 2=correct)."""
    flags = []
    for score in scores:
        if score >= threshold1:
            flags.append('2')
        elif score >= threshold2:
            flags.append('1')
        else:
            flags.append('0')

    tmp_flag = []
    identification_flag = []
    for i, f in enumerate(flags):
        tmp_flag.append(f)
        if i % n_docs == n_docs - 1:
            if '2' in tmp_flag:
                identification_flag.append(2)
            elif '1' in tmp_flag:
                identification_flag.append(1)
            else:
                identification_flag.append(0)
            tmp_flag = []
    return identification_flag


def format_qwen_prompt(question, paragraph=None):
    """Format prompt for Qwen1.5-14B-Chat model."""
    if paragraph is not None:
        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant specializing in EDA (Electronic Design Automation) "
            f"and OpenROAD. Answer questions accurately based on the provided reference documents. "
            f"If the reference doesn't contain enough information, say so.<|im_end|>\n"
            f"<|im_start|>user\nReference:\n{paragraph}\n\nQuestion: {question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant specializing in EDA (Electronic Design Automation) "
            f"and OpenROAD. Answer questions accurately.<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_path', type=str, required=True,
                        help="Path to Qwen1.5-14B-Chat generator model")
    parser.add_argument('--evaluator_path', type=str, required=True,
                        help="Path to cross-encoder reranker model for retrieval assessment (BGE-reranker or T5)")
    parser.add_argument('--input_file', type=str, required=True,
                        help="Path to MAEDA benchmark JSON")
    parser.add_argument('--retrieval_results', type=str, required=True,
                        help="Path to BGE retrieval results JSON")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Output file for generated answers")
    parser.add_argument('--method', type=str, default="crag",
                        choices=['rag', 'crag', 'no_retrieval'],
                        help="CRAG method: rag (use all), crag (corrective), no_retrieval")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--ndocs', type=int, default=10,
                        help="Number of documents to retrieve per question")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--upper_threshold', type=float, default=0.5,
                        help="Upper threshold for reranker (>= → correct)")
    parser.add_argument('--lower_threshold', type=float, default=-1.0,
                        help="Lower threshold for reranker (>= → ambiguous)")
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help="Max tokens for generation")
    parser.add_argument('--decompose_mode', type=str, default="selection",
                        choices=['selection', 'excerption', 'fixed_num'],
                        help="Strategy to decompose retrieval results for refinement")
    args = parser.parse_args()

    # Load generator
    print(f"Loading generator from {args.generator_path}...")
    generator = LLM(model=args.generator_path, dtype="half",
                    enforce_eager=True, max_logprobs=32016,
                    gpu_memory_utilization=0.90)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0,
                                     max_tokens=args.max_new_tokens,
                                     skip_special_tokens=False)

    # Load cross-encoder reranker (BGE-reranker-large or T5)
    print(f"Loading cross-encoder reranker from {args.evaluator_path}...")
    eval_tokenizer = AutoTokenizer.from_pretrained(args.evaluator_path)
    eval_model = AutoModelForSequenceClassification.from_pretrained(args.evaluator_path, num_labels=1)
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    eval_model.to(device)

    # Load benchmark data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    # Load BGE retrieval results
    with open(args.retrieval_results, 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)
    retrieval_map = {item["query_id"]: item["retrieved_ctxs"] for item in retrieval_data}

    queries = [item["question"] for item in benchmark_data]
    print(f"Loaded {len(queries)} queries")

    if args.method == 'rag':
        # Simple RAG: use all retrieved passages as context
        paragraphs = []
        for item in benchmark_data:
            qid = item.get("query_id", 0)
            ctxs = retrieval_map.get(qid, [])[:args.ndocs]
            para = ' '.join(f"{c['title']} // {c['text'].strip().replace(chr(10), ' ')}" for c in ctxs)
            paragraphs.append(para)

    elif args.method == 'crag':
        # CRAG: score and selectively refine knowledge
        # Build eval input for T5 scorer
        passages_list = []
        for item in benchmark_data:
            qid = item.get("query_id", 0)
            ctxs = retrieval_map.get(qid, [])[:args.ndocs]
            psgs = [f"{c['title']} // {c['text'].strip().replace(chr(10), ' ')}" for c in ctxs]
            passages_list.append(' [sep] '.join(psgs))

        print("Scoring retrieved documents with cross-encoder reranker...")
        scores = score_retrieved_docs(queries, passages_list, eval_tokenizer,
                                      eval_model, device, args.ndocs)
        identification_flag = process_flag(scores, args.ndocs,
                                           args.upper_threshold, args.lower_threshold)

        print(f"Identification flags: correct={identification_flag.count(2)}, "
              f"ambiguous={identification_flag.count(1)}, "
              f"incorrect={identification_flag.count(0)}")

        # Build knowledge based on flags
        paragraphs = []
        for i, (flag, item) in enumerate(zip(identification_flag, benchmark_data)):
            qid = item.get("query_id", 0)
            ctxs = retrieval_map.get(qid, [])[:args.ndocs]
            all_psgs = [f"{c['title']} // {c['text'].strip().replace(chr(10), ' ')}" for c in ctxs]

            if flag == 2:
                # Correct: refine knowledge internally
                strips = []
                for p in all_psgs:
                    strips += extract_strips_from_psg(psg=p, mode=args.decompose_mode)
                refined = select_relevants(strips, queries[i], eval_tokenizer,
                                          eval_model, device, top_n=3)
                paragraphs.append(refined)
            elif flag == 1:
                # Ambiguous: use all retrieved passages
                paragraphs.append(' '.join(all_psgs[:3]))
            else:
                # Incorrect: use top passages anyway (no web search available)
                # In original CRAG, this would use web search results
                paragraphs.append(' '.join(all_psgs[:3]))

    elif args.method == 'no_retrieval':
        paragraphs = [None] * len(queries)

    # Generate answers
    print(f"Generating answers with {args.generator_path}...")
    preds = []
    for i, (q, p) in enumerate(tqdm(zip(queries, paragraphs), total=len(queries))):
        prompt = format_qwen_prompt(q, p)
        pred = generator.generate([prompt], sampling_params)
        answer = clean_output(pred[0].outputs[0].text)
        preds.append(answer)

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(queries)}")

    # Save results
    output_data = {
        "preds": preds,
        "queries": queries,
        "method": args.method,
        "generator": args.generator_path,
        "ndocs": args.ndocs,
    }
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output_file}")

    n_empty = sum(1 for p in preds if not p.strip())
    print(f"  Total: {len(preds)}, Empty: {n_empty}")


if __name__ == '__main__':
    main()
