"""First-pass EDA for SQuAD 1.1: shape, keys, field stats."""

import json
import statistics as st
from collections import Counter
from pathlib import Path

RAW = Path("data/raw")


def load(name):
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def describe(label, xs):
    print(
        f"  {label:<28} n={len(xs):<6} min={min(xs):<7} p25={pct(xs,.25):<7} "
        f"med={int(st.median(xs)):<7} p75={pct(xs,.75):<7} p90={pct(xs,.90):<7} "
        f"p99={pct(xs,.99):<7} max={max(xs):<7} mean={st.mean(xs):.1f}"
    )


def analyze(split):
    doc = load(f"{split}-v1.1.json")
    articles = doc["data"]

    n_para = n_qa = 0
    titles, ctxs, questions = [], [], []
    n_answers = Counter()
    ans_lens, ans_starts, q_lens, ctx_lens_tok, ctx_lens_ch, para_lens_ch = [], [], [], [], [], []
    ctx_per_title = Counter()
    ans_in_multi = 0
    unique_ctx_counts = Counter()
    starts_with_article_the = 0

    for art in articles:
        titles.append(art["title"])
        ctx_per_title[art["title"]] += len(art["paragraphs"])
        for p in art["paragraphs"]:
            n_para += 1
            ctx = p["context"]
            ctxs.append(ctx)
            ctx_lens_ch.append(len(ctx))
            ctx_lens_tok.append(len(ctx.split()))
            para_lens_ch.append(len(ctx))
            for qa in p["qas"]:
                n_qa += 1
                questions.append(qa["question"])
                q_lens.append(len(qa["question"].split()))
                answers = qa["answers"]
                n_answers[len(answers)] += 1
                ans_lens.extend(len(a["text"].split()) for a in answers)
                if len({a["text"] for a in answers}) > 1:
                    ans_in_multi += 1
                unique_ctx_counts[ctx] += 1

    for art in articles:
        for p in art["paragraphs"]:
            for qa in p["qas"]:
                idx = p["context"].find(qa["answers"][0]["text"])
                starts = [a["answer_start"] for a in qa["answers"]]
                if all(
                    p["context"][s : s + len(qa["answers"][0]["text"])] != qa["answers"][0]["text"]
                    for s in starts
                ):
                    pass
                ans_starts.append(starts[0])

    print(f"\n{'='*100}\nSPLIT: {split}  (version={doc['version']})\n{'='*100}")
    print(f"  articles (titles)      {len(articles)}   unique={len(set(titles))}")
    print(f"  paragraphs             {n_para}")
    print(f"  questions              {n_qa}")
    print(f"  unique contexts        {len(set(ctxs))}")
    print(f"  avg q/paragraph        {n_qa/max(n_para,1):.2f}")

    print("\n  ANSWER COUNT PER QUESTION")
    for k in sorted(n_answers):
        print(f"    {k} answers: {n_answers[k]:>7}  ({100*n_answers[k]/n_qa:5.1f}%)")
    multi = sum(v for k, v in n_answers.items() if k > 1)
    print(f"    questions with >1 annotated answer but differing text: {ans_in_multi}")

    print("\n  LENGTH DISTRIBUTIONS (whitespace tokens unless noted)")
    describe("context tokens", ctx_lens_tok)
    describe("context chars", ctx_lens_ch)
    describe("question tokens", q_lens)
    describe("answer tokens", ans_lens)
    describe("answer_start (char offset)", ans_starts)

    print("\n  TOP-10 LONGEST CONTEXTS (tokens)")
    top = sorted(zip(ctx_lens_tok, ctxs), reverse=True)[:10]
    for n, c in top:
        print(f"    {n:>5}  {c[:70]!r}")

    print("\n  QUESTION LEADING WORDS (top 15)")
    lead = Counter(q.split()[0].lower() for q in questions)
    for w, c in lead.most_common(15):
        print(f"    {w:<12} {c:>7} ({100*c/n_qa:4.1f}%)")

    print("\n  QUESTION TYPE HEURISTIC (wh-word in first 3 tokens)")
    kinds = Counter()
    for q in questions:
        ql = q.lower().split()[:3]
        tag = "other"
        for w in ("what", "who", "when", "where", "which", "why", "how"):
            if w in ql:
                tag = w
                break
        kinds[tag] += 1
    for w, c in kinds.most_common():
        print(f"    {w:<12} {c:>7} ({100*c/n_qa:4.1f}%)")

    print("\n  DATASET IMBALANCE")
    qpc = list(ctx_per_title.values())
    describe("paragraphs per article", qpc)

    return doc


for split in ("train", "dev"):
    analyze(split)

# ---------- schema / raw shape dump ----------
print(f"\n{'='*100}\nRAW SCHEMA SHAPE\n{'='*100}")
d = load("dev-v1.1.json")
print(
    json.dumps(
        {
            "version": d["version"],
            "data": [
                {
                    "title": d["data"][0]["title"],
                    "paragraphs": [
                        {
                            "context": d["data"][0]["paragraphs"][0]["context"][:120] + " ...",
                            "qas": [
                                {
                                    "id": d["data"][0]["paragraphs"][0]["qas"][0]["id"],
                                    "question": d["data"][0]["paragraphs"][0]["qas"][0]["question"],
                                    "answers": d["data"][0]["paragraphs"][0]["qas"][0]["answers"],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        indent=2,
    )
)

# top-level key inventory
def keys_at(doc):
    a = doc["data"][0]
    p = a["paragraphs"][0]
    q = p["qas"][0]
    return {
        "root": sorted(doc),
        "article": sorted(a),
        "paragraph": sorted(p),
        "qa": sorted(q),
        "answer": sorted(q["answers"][0]),
    }


print("\nKEY INVENTORY")
for k, v in keys_at(d).items():
    print(f"  {k:<10} {v}")
