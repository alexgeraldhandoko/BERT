"""Convert the candidate export into one table row per candidate, with labels and split parts.
 
  PYTHONPATH=src python -m qa_baseline.candidate_table \
      --exports exports/v1/heldout --split splits/heldout-v1.json \
      --output exports/v1/heldout/table.npz
 
Row i of the table is row i of candidates.jsonl and row i of span_pooled.npy. 
Vectors are not copied into the table; 
vectors_for() fetches them for the rows when needed.
"""


from __future__ import annotations
 
import argparse
import json
from pathlib import Path
 
import numpy as np
 
PARTS = ("head_train", "calib", "test")

SCALAR_NAMES = (
    "score", "start_logit", "end_logit", "logp_all", "logp_ctx", "rank",
    "span_tokens", "span_chars", "score_gap", "logp_ctx_gap",
    "ent_ctx_start", "ent_ctx_end", "cls_start", "cls_end",
)


def build_table(exports_dir, split_path):
    """Read the export and the split, and return a dict of aligned numpy arrays."""
    exports_dir = Path(exports_dir)
    questions = json.loads((exports_dir / "questions.json").read_text(encoding="utf-8"))
    question_index = {question["id"]: i for i, question in enumerate(questions)}
    
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))["parts"]
    part_of = {}
    for number, name in enumerate(PARTS):
        for question_id in split[name]["ids"]:
            if question_id in part_of:
                raise ValueError(f"Question {question_id} appears in multiple parts")
            part_of[question_id] = number
    missing = [qid for qid in question_index if qid not in part_of]
    if missing:
        raise ValueError(f"Questions {missing} are missing from the split")
    
    with (exports_dir / "candidates.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    pooled_rows = np.load(exports_dir / "span_pooled.npy", mmap_mode="r").shape[0]
    if pooled_rows != len(rows):
        raise ValueError(f"span_pooled.npy has {pooled_rows} rows, but candidates.jsonl has {len(rows)}")
    windows = np.load(exports_dir / "windows.npz")
    
    def column(name, dtype=np.float64):
        return np.array([row[name] for row in rows], dtype=dtype)
    
    qid = np.array([row["qid"] for row in rows])
    q_index = np.array([question_index[row["qid"]] for row in rows], dtype=np.int32)
    window = column("window", np.int32)
    rank = column("rank")
    score, logp_ctx = column("score"), column("logp_ctx")
 
    first = rank == 1
    best_score = np.full(len(questions), np.nan)
    best_score[q_index[first]] = score[first]
    best_logp = np.full(len(questions), np.nan)
    best_logp[q_index[first]] = logp_ctx[first]
    
    scalars = np.stack(
        [
            score, column("start_logit"), column("end_logit"), column("logp_all"), logp_ctx, rank,
            column("tok_end") - column("tok_start") + 1,
            column("char_end") - column("char_start"),
            best_score[q_index] - score,
            best_logp[q_index] - logp_ctx,
            windows["ent_ctx_start"][window], windows["ent_ctx_end"][window],
            windows["cls_start"][window], windows["cls_end"][window],
        ],
        axis=1,
    ).astype(np.float32)
 
    return {
        "qid": qid,
        "q_index": q_index,
        "part": np.array([part_of[row["qid"]] for row in rows], dtype=np.int8),
        "rank": rank.astype(np.int16),
        "window": window,
        "em": column("em", np.int8),
        "f1": column("f1", np.float32),
        "scalars": scalars,
        "scalar_names": np.array(SCALAR_NAMES),
        "num_questions": np.array(len(questions)),
    }
    
    
def load_table(path):
    return dict(np.load(path))


def vectors_for(table, rows, span_pooled, h_cls):
    """[span-pooled vector | h_cls of the candidate's window] for the given table rows."""
    rows = np.asarray(rows)
    return np.concatenate([span_pooled[rows], h_cls[table["window"][rows]]], axis=1).astype(np.float32)


def summarize(table, ks=(1, 5, 20)):
    """Per part: sizes, share of correct candidates, and rank-1 / oracle@k EM and F1."""
    total = int(table["num_questions"])
    result = {}
    for number, name in enumerate(PARTS):
        in_part = table["part"] == number
        questions = np.unique(table["q_index"][in_part])
        entry = {
            "questions": len(questions),
            "candidates": int(in_part.sum()),
            "positive_rate": float(table["em"][in_part].mean()),
        }
        for k in ks:
            rows = in_part & (table["rank"] <= k)
            hit = np.zeros(total, dtype=bool)
            hit[table["q_index"][rows][table["em"][rows] == 1]] = True
            best_f1 = np.zeros(total)
            np.maximum.at(best_f1, table["q_index"][rows], table["f1"][rows])
            entry[f"em@{k}"] = 100 * float(hit[questions].mean())
            entry[f"f1@{k}"] = 100 * float(best_f1[questions].mean())
        result[name] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    table = build_table(args.exports, args.split)
    for name, entry in summarize(table).items():
        print(
            f"{name:10s} {entry['questions']:5d} questions {entry['candidates']:7d} candidates  "
            f"correct {100 * entry['positive_rate']:.1f}%  "
            f"rank-1 EM {entry['em@1']:.2f} F1 {entry['f1@1']:.2f}  "
            f"oracle@5 EM {entry['em@5']:.2f}  oracle@20 EM {entry['em@20']:.2f}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **table)
        print(f"Wrote {args.output}")
 
 
if __name__ == "__main__":
    main()