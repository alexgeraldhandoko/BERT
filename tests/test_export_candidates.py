"""Export top-K span candidates, softmax statistics and hidden states from a frozen QA checkpoint.
 
Run from the repository root:
 
  PYTHONPATH=src python -m qa_baseline.export_candidates \
      --data runs/baseline/validation-v1.1.json \
      --model runs/baseline/model \
      --output-dir exports/v1/heldout \
      --reference runs/baseline/validation-predictions.json
 
Files written to --output-dir:
  predictions.json  rank-1 answer per question
  questions.json    one record per question: gold answers, window ids, rank-1 EM/F1
  candidates.jsonl  one row per candidate span, best first (rank 1 = the baseline answer)
  span_pooled.npy   float16 (n_candidates, hidden): mean last-layer vector over each
                    candidate's tokens, row-aligned with candidates.jsonl
  windows.npz       one row per tokenizer window: h_cls (float16) and softmax statistics
 
Conventions:
  * Candidates are generated exactly like postprocess.best_answers (top n_best_size starts
    x top n_best_size ends per window, same validity filters, score = start + end logit).
  * The same character span seen in two overlapping windows is kept once, with its best score.
  * logp_all uses a softmax over every position of the window;
    logp_ctx uses a softmax over context tokens only. Both are in natural-log units.
  * h_cls belongs to a window, not a question. Candidates point to their window id.
"""


from __future__ import annotations


import argparse
import importlib.util
import json
from pathlib import Path
 
STAT_FIELDS = [
    f"{stat}_{position}"
    for stat in ("lse_all", "lse_ctx", "ent_ctx", "ctx_mass", "cls")
    for position in ("start", "end")
]
 


def window_candidates(start_row, end_row, offsets, context_length, n_best_size, max_answer_tokens):
    """
    Valid spans of one window as
    (score, start_tok, end_tok, char_start, char_end, start_logit, end_logit).
   
    Finds n best start and end logits and returns all valid combinations of them as candidates.
    """
    top_start = sorted(range(len(start_row)), key=lambda i: start_row[i], reverse=True)[:n_best_size]
    top_end = sorted(range(len(end_row)), key=lambda i: end_row[i], reverse=True)[:n_best_size]
    found = []
    for start in top_start:
        for end in top_end:
            if (
                offsets[start] is None
                or offsets[end] is None
                or end < start
                or end - start + 1 > max_answer_tokens
            ):
                continue
            char_start, char_end = offsets[start][0], offsets[end][1]
            if not 0 <= char_start < char_end <= context_length:
                continue
            start_logit, end_logit = float(start_row[start]), float(end_row[end])
            found.append((start_logit + end_logit, start, end, char_start, char_end, start_logit, end_logit))
    return found




def rank_candidates(windows, top_k):
    """
    Merges several windows into one best-first list of distinct character spans.
   
    Windows: [(window_id, [tuples from window_candidates]), ...] in window order.
    Returns [(window_id, score, start_tok, end_tok, char_start, char_end,
              start_logit, end_logit), ...] with at most top_k entries.
    """
    pool = []
    for window_id, found in windows:
        for candidate in sorted(found, key=lambda c:c[0], reverse=True)[:top_k]:
            pool.append((window_id, *candidate))
    pool.sort(key=lambda c:c[1], reverse=True)
    ranked, seen = [], set()
    for candidate in pool:
        span = (candidate[4], candidate[5])
        if span in seen:
            continue
        seen.add(span)
        ranked.append(candidate)
        if len(ranked) == top_k:
            break
    return ranked




def load_official_scorer(path):
    """
    Loads the official SQuAD scorer from a given path.
    """
    spec = importlib.util.spec_from_file_location("official_squad_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
     
def score_answer(scorer, gold_tokens, prediction):
    gold = [text for text in gold_tokens if scorer.normalize_answer(text)] or [""]
    exact = max(scorer.compute_exact(g, prediction) for g in gold)
    f1 = max(scorer.compute_f1(g, prediction) for g in gold)
    return exact, f1




def window_statistics(start_logits, end_logits, in_context):
    """
    Softmax summaries of one window.
   
    Returns:
    lse_all_{side}: logsumexp of all logits for all positions
    lse_ctx_{side}: logsumexp of logits for context positions
    ent_ctx_{side}: entropy of softmax probabilities over context positions
    ctx_mass_{side}: total probability assigned to context positions
    cls_{side}: logit for the [CLS] token
    """
    import torch
   
    stats = {}
    for side, logits in (("start", start_logits), ("end", end_logits)):
        log_all = torch.log_softmax(logits, dim=0)
        context_logits = logits[in_context]
        log_ctx = torch.log_softmax(context_logits, dim=0)
        stats[f"lse_all_{side}"] = float(torch.logsumexp(logits, dim=0))
        stats[f"lse_ctx_{side}"] = float(torch.logsumexp(context_logits, dim=0))
        stats[f"ent_ctx_{side}"] = float(-(log_ctx.exp() * log_ctx).sum())
        stats[f"ctx_mass_{side}"] = float(log_all.exp()[in_context].sum())
        stats[f"cls_{side}"] = float(logits[0])
    return stats




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("runs/baseline/model"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=None, help="candidates kept per question (default: n_best_size)")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N questions (local testing)")
    parser.add_argument("--reference", type=Path, default=None, help="existing predictions.json to compare rank 1 against")
    parser.add_argument("--scorer", type=Path, default=Path("vendor/evaluate-v2.0.py"))
    args = parser.parse_args()
   
    import numpy as np
    import torch
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
 
    from .data import load_squad
    from .features import prediction_features
   
    # Load config, model, tokenizer, and data
    config = json.loads(args.config.read_text(encoding="utf-8"))
    top_k = args.top_k or config["n_best_size"]
    examples = load_squad(args.data)
    if args.limit:
        examples = examples[: args.limit]
    contexts = {example["id"]: example["context"] for example in examples}
    scorer = load_official_scorer(args.scorer)
   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device).eval()
   
    # Prepare features and dataloader
    raw = Dataset.from_list(examples)
    features = raw.map(
        lambda batch: prediction_features(batch, tokenizer, config["max_length"], config["doc_stride"]),
        batched=True,
        remove_columns=raw.column_names,
    )
    loader = DataLoader(
        features.remove_columns(["example_id", "offset_mapping"]).with_format("torch"),
        batch_size=config["eval_batch_size"],
    )
   
    window_stats, h_cls_rows, window_question = [], [], []
    spans_per_window = {}
    v_span_by_window = {}
    next_window = 0
   
    # Run the model over all windows and collect candidates, hidden states, and softmax statistics
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch, output_hidden_states=True)
           
            starts, ends = output.start_logits.float().cpu(), output.end_logits.float().cpu()
            hidden = output.hidden_states[-1].float().cpu()
           
            meta = features[next_window : next_window + starts.shape[0]]
           
            for b in range(starts.shape[0]):
                window_id = next_window + b
                qid, offsets = meta["example_id"][b], meta["offset_mapping"][b]
               
                in_context = torch.tensor([offset is not None for offset in offsets])
                window_stats.append(window_statistics(starts[b], ends[b], in_context))
                h_cls_rows.append(hidden[b, 0].half().numpy())
                window_question.append(qid)
               
                # Generate candidates for this window and store their mean hidden states
                found = window_candidates(
                    starts[b].tolist(), ends[b].tolist(), offsets, len(contexts[qid]),
                    config["n_best_size"], config["max_answer_tokens"],
                )
                found = sorted(found, key=lambda c: c[0], reverse=True)[:top_k]
                v_span_by_window[window_id] = {
                    (c[1], c[2]): hidden[b, c[1] : c[2] + 1].mean(0).half().numpy() for c in found
                }
                spans_per_window.setdefault(qid, []).append((window_id, found))
            next_window += starts.shape[0]
           
    questions, rows, pooled_rows, predictions, per_question_scores = [], [], [], {}, []
   
    for example in examples:
        # Score candidates for this question and collect results
        qid = example["id"]
        gold = [answer["text"] for answer in example["answers"]]
        ranked = rank_candidates(spans_per_window.get(qid, []), top_k)
        scores, top1_text = [], ""
        for rank, (window_id, score, s_tok, e_tok, c_start, c_end, s_logit, e_logit) in enumerate(ranked, start=1):
            text = example["context"][c_start:c_end]
            exact, f1 = score_answer(scorer, gold, text)
            stats = window_stats[window_id]
            rows.append(
                {
                    "qid": qid, "rank": rank, "window": window_id,
                    "tok_start": s_tok, "tok_end": e_tok,
                    "char_start": c_start, "char_end": c_end, "text": text,
                    "score": score, "start_logit": s_logit, "end_logit": e_logit,
                    "logp_all": (s_logit - stats["lse_all_start"]) + (e_logit - stats["lse_all_end"]),
                    "logp_ctx": (s_logit - stats["lse_ctx_start"]) + (e_logit - stats["lse_ctx_end"]),
                    "em": exact, "f1": f1,
                }
            )
            pooled_rows.append(v_span_by_window[window_id][(s_tok, e_tok)])
            scores.append((exact, f1))
            if rank == 1:
                top1_text = text
        if not scores:
            scores = [score_answer(scorer, gold, "")]
           
        # Store the top-1 prediction and per-question statistics    
        predictions[qid] = top1_text
        per_question_scores.append(scores)
        questions.append(
            {
                "id": qid, "gold": gold,
                "windows": [window_id for window_id, _ in spans_per_window.get(qid, [])],
                "top1_window": ranked[0][0] if ranked else None,
                "n_candidates": len(ranked),
                "top1_em": scores[0][0], "top1_f1": scores[0][1],
            }
        )
       
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    question_index = {question["id"]: i for i, question in enumerate(questions)}
    (out / "questions.json").write_text(json.dumps(questions) + "\n", encoding="utf-8")
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    with (out / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    hidden_size = model.config.hidden_size
    np.save(out / "span_pooled.npy", np.stack(pooled_rows) if pooled_rows else np.zeros((0, hidden_size), np.float16))
    np.savez(
        out / "windows.npz",
        h_cls=np.stack(h_cls_rows),
        question_index=np.array([question_index[qid] for qid in window_question]),
        **{name: np.array([stats[name] for stats in window_stats], dtype=np.float32) for name in STAT_FIELDS},
    )
 
    total = len(questions)
    print(f"Questions {total}, windows {len(window_stats)}, candidates {len(rows)}")
    print(f"Rank-1 EM {100 * sum(q['top1_em'] for q in questions) / total:.2f}  "
          f"F1 {100 * sum(q['top1_f1'] for q in questions) / total:.2f}")
   
    # Compute oracle metrics for various top-K values
    for k in sorted({1, 5, 10, top_k}):
        if k > top_k:
            continue
        oracle_em = sum(max(em for em, _ in scores[:k]) for scores in per_question_scores)
        oracle_f1 = sum(max(f1 for _, f1 in scores[:k]) for scores in per_question_scores)
        print(f"Oracle@{k}  EM {100 * oracle_em / total:.2f}  F1 {100 * oracle_f1 / total:.2f}")
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        mismatched = [qid for qid, answer in predictions.items() if reference.get(qid) != answer]
        print(f"Rank-1 answers differing from {args.reference}: {len(mismatched)} of {total}")
        for qid in mismatched[:5]:
            print(f"  {qid}: export_candidate={predictions[qid]!r} reference={reference.get(qid)!r}")
 
 
if __name__ == "__main__":
    main()