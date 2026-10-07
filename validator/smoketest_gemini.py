"""Smoketest — validate 1 run across all 10 cases for Gemini-3.1 Flash-Lite."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "simulation"))

from validate import run_validation

SIM_OUTPUT = SCRIPT_DIR.parent / "simulation" / "output"
RESULTS_DIR = SCRIPT_DIR / "results" / "smoketest_gemini"

# 10 cases x 5 protocols x 1 run x 15 turns = 750 turns
run_validation(
    model="Gemini-3.1 Flash-Lite",
    input_dir=SIM_OUTPUT,
    output_dir=RESULTS_DIR,
    limit=750,
    model_prefix="gemini",
)
