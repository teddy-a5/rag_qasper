import re

SEP = "\n\n"  # separator between paragraphs in a chunk

def paper_layout(paper):
   
    ranges, pos = [], 0
    for p in paper.paragraphs:
        if not p.text.strip():
            continue
        ranges.append((p.pid, pos, pos + len(p.text)))
        pos += len(p.text) + len(SEP)
    return ranges, max(pos - len(SEP), 0)


def spans_for_window(ranges, a, b):
    spans = []
    for pid, s, e in ranges:
        lo, hi = max(a, s), min(b, e)
        if lo < hi:
            spans.append({"paragraph_id": pid, "start": lo - s, "end": hi - s})
    return spans


def make_chunk(paper, para_text, spans, index):
    text = SEP.join(para_text[sp["paragraph_id"]][sp["start"]:sp["end"]] for sp in spans)
    return {"chunk_id": f"{paper.paper_id}:c{index}", "paper_id": paper.paper_id,
            "text": text, "source_spans": spans}


def fixed_chunks(paper, size=800, overlap=200):
    """Baseline chunker. Returns a list of chunk dicts."""
    assert 0 <= overlap < size
    para_text = {p.pid: p.text for p in paper.paragraphs}
    ranges, total = paper_layout(paper)
    chunks, a, step = [], 0, size - overlap
    while a < total:
        b = min(a + size, total)
        spans = spans_for_window(ranges, a, b)
        if spans:
            ch = make_chunk(paper, para_text, spans, len(chunks))
            if ch["text"].strip():
                chunks.append(ch)
        if b >= total:
            break
        a += step
    return chunks


def chunk_corpus(papers, **kwargs):
    return [ch for paper in papers for ch in fixed_chunks(paper, **kwargs)]



"""
Paragraph-aligned chunker (for ingestion) and contextual headings (for retrieval).
"""

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")


def sentence_ranges(text):
    """(start, end) of rough sentences; only whitespace is left between ranges."""
    out, prev = [], 0
    for m in _SENT_SPLIT.finditer(text):
        out.append((prev, m.start()))
        prev = m.end()
    out.append((prev, len(text)))
    return [(s, e) for s, e in out if e > s]


def pack_ranges(text, max_chars):
    """Contiguous pieces of one paragraph, each <= max_chars, cut at sentence boundaries.
    A single sentence longer than max_chars is hard-split."""
    pieces, cur = [], None
    for s, e in sentence_ranges(text):
        while e - s > max_chars:
            if cur:
                pieces.append(cur)
                cur = None
            pieces.append((s, s + max_chars))
            s += max_chars
        if cur is None:
            cur = (s, e)
        elif e - cur[0] <= max_chars:
            cur = (cur[0], e)
        else:
            pieces.append(cur)
            cur = (s, e)
    if cur:
        pieces.append(cur)
    return pieces


def paragraph_chunks(paper, max_chars=1100):
    """One chunk per paragraph; paragraphs over max_chars are split at sentence boundaries.
    max_chars=1100 keeps 10 chunks safely under the 12,000-character limit."""
    para_text = {p.pid: p.text for p in paper.paragraphs}
    chunks = []
    for p in paper.paragraphs:
        if not p.text.strip():
            continue
        for s, e in pack_ranges(p.text, max_chars):
            spans = [{"paragraph_id": p.pid, "start": s, "end": e}]
            ch = make_chunk(paper, para_text, spans, len(chunks))
            if ch["text"].strip():
                chunks.append(ch)
    return chunks


"""
Contextual headings (for retrieval).
"""

def add_headers(paper, chunks):
    """Put title + section in `index_text` (what gets embedded/indexed).
    `text` and `source_spans` stay original, so the output and the offsets are unchanged."""
    section = {p.pid: p.section for p in paper.paragraphs}
    for ch in chunks:
        sec = section.get(ch["source_spans"][0]["paragraph_id"], "")
        head = f"Title: {paper.title}\n" + (f"Section: {sec}\n" if sec else "")
        ch["index_text"] = head + ch["text"]
    return chunks


def chunk_paper(paper, cfg):
    """Chunker switch + optional contextual headers."""
    if cfg["chunker"] == "fixed":
        chunks = fixed_chunks(paper, size=cfg["size"], overlap=cfg["overlap"])
    elif cfg["chunker"] == "paragraph":
        chunks = paragraph_chunks(paper, max_chars=cfg.get("max_chars", 1100))
    else:
        raise ValueError(f"unknown chunker {cfg['chunker']}")
    return add_headers(paper, chunks) if cfg.get("headers") else chunks

