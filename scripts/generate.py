"""Answer the 20 selected held-out questions with Gemini (temperature 0), then score them.
Usage:  set GEMINI_API_KEY, then  python scripts/generate.py --config baseline
        and   python scripts/generate.py --config final
Reads and rewrites results/heldout_{config}.jsonl. Only the 20 ANSWER_QIDS get an answer; all other
records keep answer/abstained = null. Responses are cached on disk (re-runs cost nothing)."""
import argparse
import hashlib
import json
import re
import statistics
import string
import time
from collections import Counter

from rag_qasper.config import ANALYSIS_DIR, CACHE_DIR, GEN_MODEL, RESULTS_DIR

PROMPT = """You answer a question about a scientific paper using ONLY the excerpts below.
Rules:
- Answer concisely: a short phrase or one sentence.
- For yes/no questions answer only "Yes" or "No".
- If the excerpts do not contain the answer, reply exactly: UNANSWERABLE

Excerpts:
{excerpts}

Question: {question}
Answer:"""

_CACHE = CACHE_DIR / "gen_cache.json"


def build_prompt(question, retrieved):
    excerpts = "\n\n".join(f"[{c['rank']}] {c['text']}" for c in retrieved)   # rank order, same for every system
    return PROMPT.format(excerpts=excerpts, question=question)


def parse_response(text):
    """-> (answer, abstained). Deliberate refusal is answer='' and abstained=True."""
    t = (text or "").strip()
    if not t or re.fullmatch(r"unanswerable\W*", t, flags=re.IGNORECASE):
        return "", True
    return t, False


def call_gemini(prompt, retries=3):
    """-> (text, latency_ms). Disk-cached; a cache hit reuses the original latency."""
    cache = json.loads(_CACHE.read_text(encoding="utf-8")) if _CACHE.exists() else {}
    key = hashlib.sha256((GEN_MODEL + "\x00" + prompt).encode()).hexdigest()
    if key in cache:
        return cache[key]["text"], cache[key]["latency_ms"]
    from google import genai
    from google.genai import types
    client = genai.Client()                              
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            resp = client.models.generate_content(
                model=GEN_MODEL, contents=prompt, config=types.GenerateContentConfig(temperature=0))
            latency = (time.perf_counter() - t0) * 1000
            cache[key] = {"text": resp.text or "", "latency_ms": latency}
            _CACHE.write_text(json.dumps(cache), encoding="utf-8")
            return cache[key]["text"], latency
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))


"""scoring"""
def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def gold_answers(question):
    """Answer strings of annotators who did not mark the question unanswerable."""
    out = []
    for a in question.raw_answers:
        if a["unanswerable"]:
            continue
        if a.get("yes_no") is not None:
            out.append("yes" if a["yes_no"] else "no")
        elif a.get("extractive_spans"):
            out.append(", ".join(a["extractive_spans"]))
        elif (a.get("free_form_answer") or "").strip():
            out.append(a["free_form_answer"])
    return out


def score_answer(question, answer, abstained):
    """Gold unanswerable: 1 if the system abstained, else 0.  Answerable: best F1 over annotators, 0 if abstained."""
    if question.unanswerable or not gold_answers(question):
        return 1.0 if abstained else 0.0
    return 0.0 if abstained else max(f1(answer, g) for g in gold_answers(question))


def generate_for_records(records, questions, answer_qids, call=call_gemini):
    """Fill answer/abstained/latency/llm_calls for the selected questions only. Skips ones already answered."""
    done = 0
    for r in records:
        if r["question_id"] not in answer_qids or r["answer"] is not None:
            continue
        text, gen_ms = call(build_prompt(questions[r["question_id"]].text, r["retrieved"]))
        r["answer"], r["abstained"] = parse_response(text)
        r["latency_ms"] = round(r["latency_ms"] + gen_ms, 2)       # retrieval + generation
        r["llm_calls"] += 1
        done += 1
    return done


def summarize(records, questions, answer_qids):
    sel = [r for r in records if r["question_id"] in answer_qids]
    rest = [r for r in records if r["question_id"] not in answer_qids]
    scores = [score_answer(questions[r["question_id"]], r["answer"], r["abstained"]) for r in sel]
    unans = [r for r in sel if questions[r["question_id"]].unanswerable]
    return {
        "n_selected": len(sel),
        "n_gold_unanswerable_in_selected": len(unans),
        "answer_F1": sum(scores) / len(scores) if scores else None,
        "false_answer_rate": (sum(not r["abstained"] for r in unans) / len(unans)) if unans else "undefined (no unanswerable cases)",
        "abstained": sum(bool(r["abstained"]) for r in sel),
        "median_latency_ms_selected_retrieval_plus_generation": statistics.median(r["latency_ms"] for r in sel) if sel else None,
        "median_latency_ms_others_retrieval_only": statistics.median(r["latency_ms"] for r in rest) if rest else None,
        "llm_calls_per_selected_question": sum(r["llm_calls"] for r in sel) / len(sel) if sel else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    from rag_qasper.data import load_corpus
    from scripts.run import write_jsonl
    from rag_qasper.validate import read_jsonl, validate_records
    corpus, answer_qids = load_corpus()
    questions = {q.qid: q for p in corpus["heldout"] for q in p.questions}
    path = RESULTS_DIR / f"heldout_{a.config}.jsonl"
    records = read_jsonl(path)
    n = generate_for_records(records, questions, answer_qids)
    write_jsonl(path, records)
    summary = summarize(records, questions, answer_qids)
    (ANALYSIS_DIR / f"heldout_{a.config}_answers.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"generated {n} new answers")
    print(json.dumps(summary, indent=1))
    errs = validate_records(records, corpus["heldout"], "heldout", answer_qids, require_answers=True)
    print("validation:", "OK" if not errs else f"{len(errs)} problems")
    for e in errs[:20]:
        print(" -", e)


if __name__ == "__main__":
    main()

