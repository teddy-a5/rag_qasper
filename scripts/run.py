"""Run one configuration on one split.
Usage: python scripts/run.py --split dev --config baseline
Writes results/{split}_{config}.jsonl and analysis/{split}_{config}_scores.json."""
import argparse
import json
import statistics
import time

from dotenv import load_dotenv

from rag_qasper.chunking import chunk_paper
from rag_qasper.config import ANALYSIS_DIR, CONFIGS, MAX_CHARS, RESULTS_DIR, TOP_K
from rag_qasper.embed import Embedder
from rag_qasper.metrics import mean_ignoring_none, mrr_at_k, recall_at_k
from rag_qasper.retriever.dense import DenseRetriever
from rag_qasper.retriever.hybrid import HybridRetriever
from rag_qasper.validate import validate_records

load_dotenv()


def build_chunks(papers, cfg):
    return [ch for paper in papers for ch in chunk_paper(paper, cfg)]


def build_retriever(chunks, cfg, embedder):
    if cfg["retriever"] == "dense":
        return DenseRetriever(chunks, embedder=embedder)
    if cfg["retriever"] == "hybrid":
        return HybridRetriever(chunks, embedder=embedder, pool=cfg.get("pool", 100), rrf_k=cfg.get("rrf_k", 60))
    raise ValueError(f"unknown retriever {cfg['retriever']}")


def format_retrieved(chunks, hits):
    """Exactly TOP_K unique chunks, ranked from 1, in the output format."""
    out, seen = [], set()
    for idx, score in hits:
        ch = chunks[idx]
        if ch["chunk_id"] in seen:
            continue
        seen.add(ch["chunk_id"])
        out.append({"rank": len(out) + 1, "chunk_id": ch["chunk_id"], "paper_id": ch["paper_id"],
                    "text": ch["text"], "source_spans": ch["source_spans"], "score": round(float(score), 6)})
        if len(out) == TOP_K:
            break
    if sum(len(c["text"]) for c in out) > MAX_CHARS:
        raise ValueError(f"combined text exceeds {MAX_CHARS} chars; shrink chunks or trim")
    return out


def run_split(papers, cfg, embedder):
    """Search ALL papers of the split for every question (no filtering by target paper)."""
    chunks = build_chunks(papers, cfg)
    retriever = build_retriever(chunks, cfg, embedder)
    para_text = {p.paper_id: {x.pid: x.text for x in p.paragraphs} for p in papers}
    records, per_q = [], []
    for paper in papers:
        for q in paper.questions:
            t0 = time.perf_counter()
            hits = retriever.search(q.text, top_k=TOP_K)       # includes query embedding time
            retrieved = format_retrieved(chunks, hits)
            latency_ms = (time.perf_counter() - t0) * 1000
            records.append({"question_id": q.qid, "target_paper_id": paper.paper_id,
                            "retrieved": retrieved, "answer": None, "abstained": None,
                            "latency_ms": round(latency_ms, 2), "llm_calls": 0})
            pt = para_text[paper.paper_id]
            per_q.append({
                "question_id": q.qid, "paper_id": paper.paper_id, "question": q.text,
                "gold": q.gold_pids,
                "r5": recall_at_k(retrieved, paper.paper_id, q.gold_pids, pt, 5),
                "r10": recall_at_k(retrieved, paper.paper_id, q.gold_pids, pt, 10),
                "mrr": mrr_at_k(retrieved, paper.paper_id, q.gold_pids, pt, 10),
                "top_papers": [c["paper_id"] for c in retrieved],
            })
    summary = {
        "n_papers": len(papers), "n_chunks": len(chunks), "n_questions": len(records),
        "n_scored": sum(1 for r in per_q if r["gold"]),
        "R@5": mean_ignoring_none(r["r5"] for r in per_q),
        "R@10": mean_ignoring_none(r["r10"] for r in per_q),
        "MRR@10": mean_ignoring_none(r["mrr"] for r in per_q),
        "median_latency_ms": statistics.median(r["latency_ms"] for r in records),
        "llm_calls_per_q": sum(r["llm_calls"] for r in records) / len(records),
    }
    return records, per_q, summary


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "heldout"], required=True)
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--cold-queries", action="store_true",
                    help="embed every query fresh (no query cache) so latency includes embedding")
    a = ap.parse_args()

    from rag_qasper.data import load_corpus
    corpus, answer_qids = load_corpus()
    papers = corpus[a.split]
    embedder = Embedder(use_query_cache=not a.cold_queries)
    records, per_q, summary = run_split(papers, CONFIGS[a.config], embedder)
    embedder.save()

    name = f"{a.split}_{a.config}"
    write_jsonl(RESULTS_DIR / f"{name}.jsonl", records)
    (ANALYSIS_DIR / f"{name}_scores.json").write_text(
        json.dumps({"summary": summary, "per_question": per_q}, indent=1), encoding="utf-8")

    errors = validate_records(records, papers, a.split, answer_qids)
    print(json.dumps(summary, indent=1))
    print("validation:", "OK" if not errors else f"{len(errors)} problems")
    for e in errors[:20]:
        print(" -", e)


if __name__ == "__main__":
    main()

