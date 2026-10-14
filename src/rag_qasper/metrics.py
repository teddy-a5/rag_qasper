"""Retrieval evaluation metrics: span coverage, Recall@k, and MRR@10.

Metric definitions:
- Span Coverage: Fraction of gold paragraph whitespace tokens encompassed by retrieved source spans.
- Paragraph Retrieval: A gold paragraph is hit when coverage >= COVERAGE_THRESHOLD.
- Recall@k: Proportion of gold paragraphs retrieved within the top-k results.
- MRR@10: Reciprocal rank of the first top-k chunk that successfully retrieves a gold paragraph.
"""
import re
from collections import defaultdict

from .config import COVERAGE_THRESHOLD


def token_ranges(text):
    """Whitespace tokens as (start, end) character ranges."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _merge(intervals):
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def paragraph_coverage(text, spans):
    """Fraction of the paragraph's tokens fully inside the union of (start, end) spans."""
    toks = token_ranges(text)
    if not toks:
        return 0.0
    merged = _merge(spans)
    covered = sum(any(s <= ts and te <= e for s, e in merged) for ts, te in toks)
    return covered / len(toks)


def _spans_by_paragraph(chunks, target_paper_id):
    out = defaultdict(list)
    for ch in chunks:
        if ch["paper_id"] != target_paper_id:
            continue
        for sp in ch["source_spans"]:
            out[sp["paragraph_id"]].append((sp["start"], sp["end"]))
    return out


def found_paragraphs(chunks, target_paper_id, gold_pids, para_text, combine="union"):
    """Set of gold paragraph ids found by these chunks."""
    if combine == "single":
        found = set()
        for ch in chunks:
            found |= found_paragraphs([ch], target_paper_id, gold_pids, para_text, "union")
        return found
    spans = _spans_by_paragraph(chunks, target_paper_id)
    return {
        pid for pid in gold_pids
        if paragraph_coverage(para_text[pid], spans.get(pid, [])) >= COVERAGE_THRESHOLD
    }


def recall_at_k(retrieved, target_paper_id, gold_pids, para_text, k, combine="union"):
    if not gold_pids:
        return None
    found = found_paragraphs(retrieved[:k], target_paper_id, gold_pids, para_text, combine)
    return len(found) / len(gold_pids)


def mrr_at_k(retrieved, target_paper_id, gold_pids, para_text, k=10, combine="union"):
    if not gold_pids:
        return None
    for r in range(1, min(k, len(retrieved)) + 1):
        if found_paragraphs(retrieved[:r], target_paper_id, gold_pids, para_text, combine):
            return 1.0 / r
    return 0.0


def mean_ignoring_none(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")

