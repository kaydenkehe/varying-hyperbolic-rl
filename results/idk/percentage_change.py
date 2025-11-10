#!/usr/bin/env python3
"""Compute percent change in final mean returns from EPPO to HPPO results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict


RESULTS_DIR = Path(__file__).resolve().parent


def extract_final_mean(file_path: Path) -> float:
    """Return the mean_returns value from the last data row in the file."""
    last_line = None
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("iter"):
                continue
            last_line = stripped

    if last_line is None:
        raise ValueError(f"No data rows found in {file_path}")

    columns = last_line.split("\t")
    if len(columns) <= 3:
        raise ValueError(f"Row missing mean_returns column in {file_path}")

    return float(columns[3])


def percent_change(eppo: float, hppo: float) -> float:
    """Compute percent change from EPPO to HPPO with EPPO as the baseline."""
    if eppo == 0:
        if hppo == 0:
            return 0.0
        return math.inf if hppo > 0 else -math.inf
    return ((hppo - eppo) / abs(eppo)) * 100.0


def main() -> None:
    runs: Dict[str, Dict[str, float]] = {"eppo": {}, "hppo": {}}

    for file_path in RESULTS_DIR.glob("*.txt"):
        prefix, _, game = file_path.stem.partition("_")
        prefix = prefix.lower()
        if not game or prefix not in runs:
            continue
        runs[prefix][game] = extract_final_mean(file_path)

    games = sorted(set(runs["eppo"]) & set(runs["hppo"]))
    if not games:
        print("No games have both EPPO and HPPO results.")
        return

    header = f"{'Game':<15}{'EPPO':>12}{'HPPO':>12}{'% Change':>14}"
    print(header)
    print("-" * len(header))

    for game in games:
        eppo_value = runs["eppo"][game]
        hppo_value = runs["hppo"][game]
        change = percent_change(eppo_value, hppo_value)
        change_str = f"{change:>13.2f}%" if math.isfinite(change) else f"{'undefined':>13}"
        print(
            f"{game:<15}"
            f"{eppo_value:12.3f}"
            f"{hppo_value:12.3f}"
            f"{change_str}"
        )


if __name__ == "__main__":
    main()
