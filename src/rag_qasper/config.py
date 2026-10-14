"""Global configuration and experiment definitions for the RAG pipeline."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
ANALYSIS_DIR = ROOT / "analysis"

CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)


DATA_URL = (
    "hf://datasets/allenai/qasper@0fbdb8edab3c45c9df13ed1e1cbb4d64e96dbd46/"
    "qasper/validation-00000-of-00001.parquet"
)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K = 10            
MAX_CHARS = 12_000   
COVERAGE_THRESHOLD = 0.8

ANALYSIS_DIR = ROOT / "analysis"   
ANALYSIS_DIR.mkdir(exist_ok=True)


CONFIGS = {
    # Baseline: fixed-size character chunking with dense retrieval
    "baseline": {"chunker": "fixed", "size": 800, "overlap": 200, "retriever": "dense"},
    # Ablation A: paragraph-aligned chunking bounded by sentence limits
    "A": {"chunker": "paragraph", "max_chars": 1100, "retriever": "dense"},
    # Ablation B: contextual document and section headers prepended to index text
    "B": {"chunker": "fixed", "size": 800, "overlap": 200, "headers": True, "retriever": "dense"},
    # Ablation C: hybrid dense + BM25 retrieval merged via RRF
    "C": {"chunker": "fixed", "size": 800, "overlap": 200, "retriever": "hybrid"},
    # Final configuration (A + B + C)
    "final": {"chunker": "paragraph", "max_chars": 1100, "headers": True, "retriever": "hybrid"},
    # Ablation without headers (A + C)
    "final_nohdr": {"chunker": "paragraph", "max_chars": 1100, "retriever": "hybrid"},
}

