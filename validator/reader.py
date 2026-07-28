"""Reader for simulation CSV output files."""

import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def read_csv_dir(output_dir, model_prefix=None, batch_start=None, batch_end=None):
    """Read CSV batch files from a directory. Optionally filter by batch range.
    
    Args:
        output_dir: path to CSV directory
        model_prefix: file prefix (e.g. "gpt", "gemini", "glm")
        batch_start: 1-based index of first batch to read
        batch_end: 1-based index of last batch to read (inclusive)
    """
    turns = []
    pattern = f"{model_prefix}_batch_*.csv" if model_prefix else "gpt_batch_*.csv"
    files = sorted(Path(output_dir).glob(pattern))
    
    if batch_start is not None and batch_end is not None:
        files = files[batch_start - 1 : batch_end]
    
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row["turn"] = int(row.get("turn", 0))
                row["run"] = int(row.get("run", 0))
                row["isInjection"] = row.get("isInjection", "") == "True"
                turns.append(row)
    return turns
