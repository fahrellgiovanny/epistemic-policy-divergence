"""Smoketest — validate 1 run across all 10 cases for GLM-4.5-Air."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "simulation"))

from validate import run_validation

SIM_OUTPUT = SCRIPT_DIR.parent / "simulation" / "output"
RESULTS_DIR = SCRIPT_DIR / "results" / "smoketest_glm"

run_validation(
    model="GLM-4.5-Air",
    input_dir=SIM_OUTPUT,
    output_dir=RESULTS_DIR,
    limit=750,
    model_prefix="glm",
)
