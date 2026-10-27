"""Second-pass EDA: modelling-relevant shape — windowing, answer type, alignment."""

import json
from collections import Counter
from pathlib import Path

RAW = Path("data/raw")


def load(s):
    return json.loads((RAW / f"{s}-v1.1.json").read_text(encoding="utf-8"))


def walk(doc):
    for art in doc["data"]:
        for p in art["paragraphs"]:
            for qa in p["qas"]:
                yield art["title"], p["context"], qa


def approx_wp(text):
    """Rough uncased-WordPiece token count (no tokenizer installed)."""
    n = 0
    for w in text.split():
        n += max(1, (len(w) + 4) // 5)
    return n


for split in ("train", "dev"):
    doc = load(split)
    print(f"\n{'='*90}\n{split.upper()}\n{'='*90}")

    exact_sub = 0
    leading_ws = trailing_ws = 0
    has_punct_end = 0
    caps_start = 0
    digit_ans = 0
    total = 0
    kind = Counter()
    per_q_ans_texts = Counter()
    truncation_risk = 0
    ctx_gt_256 = 0
    first_answer_at_boundary = 0
    differing_starts = 0
    n_windows_total = 0

    for title, ctx, qa in walk(doc):
        total += 1
        ans = qa["answers"]
        texts = {a["text"] for a in ans}
        per_q_ans_texts[len(texts)] += 1
        if len({a["answer_start"] for a in ans}) > 1:
            differing_starts += 1
        t = ans[0]["text"]
        s = ans[0]["answer_start"]
        if ctx[s : s + len(t)] == t:
            exact_sub += 1
        if t != t.strip():
            if t != t.lstrip():
                leading_ws += 1
            if t != t.rstrip():
                trailing_ws += 1
        if t.endswith((".", ",", '"', ")", ";")):
            has_punct_end += 1
        if t[:1].isupper():
            caps_start += 1
        if any(ch.isdigit() for ch in t):
            digit_ans += 1
        if t.isdigit():
            kind["pure number"] += 1
        elif len(t.split()) == 1:
            kind["single word"] += 1
        elif all(ch.isdigit() or ch in ".,%-/ " for ch in t):
            kind["numeric-ish span"] += 1
        else:
            kind["multi word"] += 1

        # windowing proxies (max_length=384 default, but simple: whole thing fits?)
        qtok = approx_wp(qa["question"])
        ctok = approx_wp(ctx)
        avail = 384 - qtok - 3
        if ctok > avail:
            truncation_risk += 1
        if avail + qtok + 3 > 256 - qtok:
            pass
        if ctok + qtok + 3 > 256:
            ctx_gt_256 += 1
        # does gold answer fall near a stride boundary (128 tokens of 384-avail window)?
        windows = 1
        if ctok > avail:
            windows = 1 + -(-(ctok - avail) // 128)
        n_windows_total += windows

    print(f"  questions                          {total}")
    print(f"  gold span is exact substring       {exact_sub} ({100*exact_sub/total:.2f}%)")
    print(f"  answers w/ leading whitespace      {leading_ws}")
    print(f"  answers w/ trailing whitespace     {trailing_ws}")
    print(f"  answers ending in punctuation      {has_punct_end} ({100*has_punct_end/total:.1f}%)")
    print(f"  answers starting uppercase         {caps_start} ({100*caps_start/total:.1f}%)")
    print(f"  answers containing a digit         {digit_ans} ({100*digit_ans/total:.1f}%)")
    print(f"  distinct answer_start per question {differing_starts}")
    print("\n  ANSWER SURFACE FORM")
    for k, v in kind.most_common():
        print(f"    {k:<20} {v:>7} ({100*v/total:4.1f}%)")
    print("\n  # DISTINCT ANSWER TEXTS PER QUESTION")
    for k in sorted(per_q_ans_texts):
        print(f"    {k}: {per_q_ans_texts[k]:>7} ({100*per_q_ans_texts[k]/total:4.1f}%)")
    print("\n  WINDOWING (approx, max_length=384 / stride=128)")
    print(f"    questions needing >1 window      {truncation_risk} ({100*truncation_risk/total:.1f}%)")
    print(f"    avg windows per question         {n_windows_total/total:.2f}")

# max windows properly
for split in ("train", "dev"):
    doc = load(split)
    ws = []
    for _, ctx, qa in walk(doc):
        avail = 384 - approx_wp(qa["question"]) - 3
        ctok = approx_wp(ctx)
        ws.append(1 if ctok <= avail else 1 + -(-(ctok - avail) // 128))
    print(f"  {split}: windows/question min={min(ws)} max={max(ws)} mean={sum(ws)/len(ws):.2f} "
          f"p95={sorted(ws)[int(.95*len(ws))]}")

# cross-split article overlap
tr = {a["title"] for a in load("train")["data"]}
dv = {a["title"] for a in load("dev")["data"]}
print(f"\n  article title overlap train∩dev: {len(tr & dv)}")
print(f"  dev topics sample: {sorted(dv)[:8]}")
print(f"  train topics sample: {sorted(tr)[:8]}")
