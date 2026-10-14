"""rag_qasper package."""

import sys

from scripts import run

from .config import ANALYSIS_DIR, CACHE_DIR, CONFIGS, RESULTS_DIR, ROOT
from .data import load_corpus


def main():
    """CLI entry-point: delegates to scripts/run.py logic."""

    run.main()
