#!/usr/bin/env python3
"""Extract key thermo-probe metrics from a SWE-agent trajectory file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _get_total_tokens(data: dict) -> int:
    model_stats = data.get("info", {}).get("model_stats", {})
    if "total_tokens" in model_stats:
        return int(model_stats["total_tokens"])
    sent = int(model_stats.get("tokens_sent", 0))
    received = int(model_stats.get("tokens_received", 0))
    return sent + received


def _get_thermo_counts(data: dict) -> tuple[int, int, int]:
    summary = data.get("info", {}).get("thermo_probe", {})
    useful = int(summary.get("useful_work", 0))
    friction = int(summary.get("friction_loss", 0))
    if useful or friction:
        return useful, friction, int(summary.get("unknown", 0))

    useful = friction = unknown = 0
    for step in data.get("trajectory", []):
        label = (
            step.get("extra_info", {})
            .get("thermo_probe", {})
            .get("classification", "unknown")
        )
        if label == "useful_work":
            useful += 1
        elif label == "friction_loss":
            friction += 1
        else:
            unknown += 1
    return useful, friction, unknown


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SWE-agent thermo probe trajectory.")
    parser.add_argument("trajectory", type=Path, help="Path to *.traj / trajectory.json")
    args = parser.parse_args()

    data = _load(args.trajectory)
    total_tokens = _get_total_tokens(data)
    useful, friction, unknown = _get_thermo_counts(data)

    denom = useful + friction
    friction_ratio = (friction / denom) if denom else 0.0

    final_status = data.get("info", {}).get("exit_status", "unknown")

    print(f"Total Token Consumption: {total_tokens}")
    print(
        "Friction Loss Ratio (friction/(useful+friction)): "
        f"{friction_ratio:.4f} (useful={useful}, friction={friction}, unknown={unknown})"
    )
    print(f"Final Task Status: {final_status}")


if __name__ == "__main__":
    main()
