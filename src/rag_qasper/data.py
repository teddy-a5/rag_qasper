import random
import re
from dataclasses import dataclass

from datasets import load_dataset

from .config import DATA_URL, SEED


@dataclass
class Paragraph:
    pid: str  
    text: str
    section: str 


@dataclass
class Question:
    qid: str
    paper_id: str
    text: str
    gold_pids: list                
    unanswerable: bool        
    raw_answers: list

@dataclass
class Paper:
    paper_id: str
    title: str
    paragraphs: list
    questions: list


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def make_splits(ds):
    paper_ids = sorted(set(ds["id"]))
    random.Random(SEED).shuffle(paper_ids)

    build_ids, held_out_ids = paper_ids[:80], paper_ids[80:130]
    held = set(held_out_ids)

    qids = sorted({
        str(qid) for paper in ds if paper["id"] in held
        for qid in (paper["qas"]["question_id"] if isinstance(paper["qas"], dict)
                    else [q["question_id"] for q in paper["qas"]])
    })

    random.Random(SEED).shuffle(qids)

    return build_ids, held_out_ids, set(qids[:20])


def parse_paper(rec) -> Paper:
    names = rec["full_text"].get("section_name") or []
    paras = [Paragraph("abstract", rec["abstract"] or "", "Abstract")]
    for si, section in enumerate(rec["full_text"]["paragraphs"]):
        name = names[si] if si < len(names) and names[si] else ""
        for pi, text in enumerate(section):
            paras.append(Paragraph(f"s{si}:p{pi}", text or "", name))

    lookup = {}
    for p in paras:
        if _normalize_text(p.text):
            lookup.setdefault(_normalize_text(p.text), p.pid)

    qas = rec["qas"]
    questions = []
    for i, qid in enumerate(qas["question_id"]):
        anns = qas["answers"][i]["answer"]
        gold, unans = [], []
        for a in anns:
            unans.append(bool(a["unanswerable"]))
            for ev in a["evidence"]:
                if not _normalize_text(ev):
                    continue
                pid = lookup.get(_normalize_text(ev)) 
                if pid and pid not in gold:
                    gold.append(pid)
        questions.append(Question(
            qid=str(qid), paper_id=rec["id"], text=qas["question"][i],
            gold_pids=gold, unanswerable=sum(unans) > len(unans) / 2, raw_answers=anns,
        ))
    return Paper(rec["id"], rec["title"], paras, questions)


def load_corpus():
    ds = load_dataset("parquet", data_files={"validation": DATA_URL}, split="validation")
    build_ids, held_ids, answer_qids = make_splits(ds)
    by_id = {r["id"]: r for r in ds}
    corpus = {
        "dev": [parse_paper(by_id[i]) for i in build_ids],
        "heldout": [parse_paper(by_id[i]) for i in held_ids],
    }
    return corpus, answer_qids

