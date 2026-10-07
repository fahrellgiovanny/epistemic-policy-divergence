"""Validate Gemini-3.1 Flash-Lite simulation results — full sweep."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "simulation"))

from validate import run_validation

SIM_OUTPUT = SCRIPT_DIR.parent / "simulation" / "output"
RESULTS_DIR = SCRIPT_DIR / "results" / "gemini"

run_validation(
    model="Gemini-3.1 Flash-Lite",
    input_dir=SIM_OUTPUT,
    output_dir=RESULTS_DIR,
    model_prefix="gemini",
)
