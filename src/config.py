"""Central settings. Every experiment is a dict of switches in CONFIGS (add them later)."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DATA_URL = (
    "hf://datasets/allenai/qasper@0fbdb8edab3c45c9df13ed1e1cbb4d64e96dbd46/"
    "qasper/validation-00000-of-00001.parquet"
)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# BGE v1.5: prefix queries (never documents) with this instruction. Keep it fixed across configs.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K = 10            # chunks returned per question
MAX_CHARS = 12_000    # combined text limit for the 10 chunks
COVERAGE_THRESHOLD = 0.8
GEN_MODEL = "gemini-3.5-flash-lite"  # temperature 0, 20 held-out questions only

ANALYSIS_DIR = ROOT / "analysis"   # per-question scores for failure analysis (not part of results/)
ANALYSIS_DIR.mkdir(exist_ok=True)

# One dict of switches per experiment. Add "A", "B", "C", "final" here later.
CONFIGS = {
    "baseline": {"chunker": "fixed", "size": 800, "overlap": 200, "retriever": "dense"},
    # A: paragraph-aligned chunks (ingestion)
    "A": {"chunker": "paragraph", "max_chars": 1100, "retriever": "dense"},
    # B: contextual headings, title + section prepended to the embedded text only (ingestion)
    "B": {"chunker": "fixed", "size": 800, "overlap": 200, "headers": True, "retriever": "dense"},
    # C: hybrid BM25 + dense with RRF (retrieval), on the baseline chunks
    "C": {"chunker": "fixed", "size": 800, "overlap": 200, "retriever": "hybrid"},
    # final = A + B + C (best on dev: R@10 0.286 vs 0.246 without headers)
    "final": {"chunker": "paragraph", "max_chars": 1100, "headers": True, "retriever": "hybrid"},
    # extra ablation row: A + C without headers
    "final_nohdr": {"chunker": "paragraph", "max_chars": 1100, "retriever": "hybrid"},
}