"""Schema and consistency validation for experiment output files (JSONL).

Usage:
    python -m rag_qasper.validate --split dev --file results/dev_baseline.jsonl [--require-answers]
"""
import argparse
import json
import re

from .config import MAX_CHARS, TOP_K

PID_RE = re.compile(r"^(abstract|s\d+:p\d+)$")
REQUIRED = ["question_id", "target_paper_id", "retrieved", "answer", "abstained", "latency_ms", "llm_calls"]
CHUNK_KEYS = ["rank", "chunk_id", "paper_id", "text", "source_spans"]


def validate_records(records, papers, split, answer_qids=frozenset(), require_answers=False):
    errors = []
    expected = {q.qid: p.paper_id for p in papers for q in p.questions}
    para_len = {p.paper_id: {x.pid: len(x.text) for x in p.paragraphs} for p in papers}

    ids = [r.get("question_id") for r in records]
    if len(ids) != len(set(ids)):
        errors.append("duplicate question_id values")
    if set(ids) != set(expected):
        errors.append(f"question ids differ: missing={len(set(expected) - set(ids))} "
                      f"extra={len(set(ids) - set(expected))}")

    for r in records:
        qid = r.get("question_id")
        miss = [k for k in REQUIRED if k not in r]
        if miss:
            errors.append(f"{qid}: missing keys {miss}")
            continue
        if expected.get(qid) != r["target_paper_id"]:
            errors.append(f"{qid}: wrong target_paper_id")
        ret = r["retrieved"]
        if len(ret) != TOP_K:
            errors.append(f"{qid}: {len(ret)} chunks, need {TOP_K}")
        if [c.get("rank") for c in ret] != list(range(1, len(ret) + 1)):
            errors.append(f"{qid}: ranks must be 1..{TOP_K} in order")
        if len({c.get("chunk_id") for c in ret}) != len(ret):
            errors.append(f"{qid}: duplicate chunk_id")
        if sum(len(c.get("text", "")) for c in ret) > MAX_CHARS:
            errors.append(f"{qid}: combined text > {MAX_CHARS} chars")
        for c in ret:
            if any(k not in c for k in CHUNK_KEYS):
                errors.append(f"{qid}: chunk missing keys")
                continue
            if not c["source_spans"]:
                errors.append(f"{c['chunk_id']}: empty source_spans")
            for sp in c["source_spans"]:
                pid, s, e = sp.get("paragraph_id"), sp.get("start"), sp.get("end")
                if not (isinstance(pid, str) and PID_RE.match(pid)):
                    errors.append(f"{c['chunk_id']}: bad paragraph id {pid!r}")
                    continue
                n = para_len.get(c["paper_id"], {}).get(pid)
                if n is None:
                    errors.append(f"{c['chunk_id']}: paragraph {pid} not in paper {c['paper_id']}")
                elif not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= n):
                    errors.append(f"{c['chunk_id']}: span {s}-{e} outside paragraph {pid} (len {n})")
        if not isinstance(r["latency_ms"], (int, float)) or r["latency_ms"] < 0:
            errors.append(f"{qid}: bad latency_ms")
        if not isinstance(r["llm_calls"], int) or r["llm_calls"] < 0:
            errors.append(f"{qid}: bad llm_calls")
        ans, abst = r["answer"], r["abstained"]
        if split == "heldout" and qid in answer_qids:
            if ans is None and abst is None:
                if require_answers:
                    errors.append(f"{qid}: selected question needs an answer")
            elif not isinstance(ans, str) or not isinstance(abst, bool):
                errors.append(f"{qid}: answer must be str and abstained bool")
            elif abst and ans != "":
                errors.append(f"{qid}: abstained=true needs answer=\"\"")
            elif not abst and not ans.strip():
                errors.append(f"{qid}: empty answer needs abstained=true")
        elif ans is not None or abst is not None:
            errors.append(f"{qid}: answer/abstained must be null for this question")
    return errors


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


if __name__ == "__main__":
    from .data import load_corpus
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "heldout"], required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--require-answers", action="store_true")
    a = ap.parse_args()
    corpus, answer_qids = load_corpus()
    errs = validate_records(read_jsonl(a.file), corpus[a.split], a.split, answer_qids, a.require_answers)
    print("OK" if not errs else f"{len(errs)} problems")
    for e in errs[:30]:
        print(" -", e)

