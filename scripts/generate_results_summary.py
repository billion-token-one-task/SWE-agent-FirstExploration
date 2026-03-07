#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_SUMMARIES_DIR = ROOT / "experiments" / "eval_summaries"
EVAL_HISTORY_ROOT = ROOT / "results" / "evaluation_history"
TRAJ_ROOT = ROOT / "results" / "trajectories"
TRAJ_RUNS_ROOT = ROOT / "results" / "trajectories" / "ubuntu"
OUT_DIR = ROOT / "results" / "summary"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def collect_eval_summaries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(EVAL_SUMMARIES_DIR.glob("*.json")):
        data = _load_json(path)
        if not isinstance(data, dict):
            continue

        stem = path.stem
        if stem.startswith("summary__"):
            remainder = stem[len("summary__") :]
            if "__" in remainder:
                run_id, eval_tag = remainder.rsplit("__", 1)
            else:
                run_id, eval_tag = remainder, ""
        elif ".lite" in stem:
            idx = stem.index(".lite")
            run_id = stem[:idx]
            eval_tag = stem[idx + 1 :]
        elif ".partial_eval" in stem:
            idx = stem.index(".partial_eval")
            run_id = stem[:idx]
            eval_tag = stem[idx + 1 :]
        else:
            run_id = stem
            eval_tag = ""

        total = _safe_int(data.get("total_instances"))
        submitted = _safe_int(data.get("submitted_instances"))
        completed = _safe_int(data.get("completed_instances"))
        resolved = _safe_int(data.get("resolved_instances"))
        unresolved = _safe_int(data.get("unresolved_instances"))
        errors = _safe_int(data.get("error_instances"))
        empty_patch = _safe_int(data.get("empty_patch_instances"))

        rows.append(
            {
                "file_name": path.name,
                "run_id": run_id,
                "eval_tag": eval_tag,
                "total_instances": total,
                "submitted_instances": submitted,
                "completed_instances": completed,
                "resolved_instances": resolved,
                "unresolved_instances": unresolved,
                "error_instances": errors,
                "empty_patch_instances": empty_patch,
                "resolved_over_total_pct": _pct(resolved, total),
                "resolved_over_submitted_pct": _pct(resolved, submitted),
                "completed_over_total_pct": _pct(completed, total),
            }
        )
    return rows


def _parse_report_path(path: Path) -> tuple[str, str, str]:
    rel = path.relative_to(EVAL_HISTORY_ROOT)
    source = rel.parts[0] if len(rel.parts) >= 1 else "unknown_source"
    eval_session = rel.parts[1] if len(rel.parts) >= 2 else "unknown_eval_session"

    # Typical forms:
    # from_swe_agent_src/<eval_session>/<run_id>/<instance>/report.json
    # from_workspace_root/<run_id>/<run_id>/<instance>/report.json
    run_id = rel.parts[-3] if len(rel.parts) >= 3 else "unknown_run"
    return source, eval_session, run_id


def collect_instance_reports() -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "instances": 0,
            "resolved_true": 0,
            "resolved_false": 0,
            "resolved_unknown": 0,
            "patch_exists_true": 0,
            "patch_applied_true": 0,
        }
    )

    report_file_count = 0
    for report_path in sorted(EVAL_HISTORY_ROOT.rglob("report.json")):
        report_file_count += 1
        report_data = _load_json(report_path)
        if not isinstance(report_data, dict):
            continue

        source, eval_session, run_id = _parse_report_path(report_path)

        for instance_id, payload in report_data.items():
            if not isinstance(payload, dict):
                continue

            resolved = payload.get("resolved")
            patch_exists = bool(payload.get("patch_exists"))
            patch_applied = bool(payload.get("patch_successfully_applied"))

            if resolved is True:
                resolved_label = "true"
            elif resolved is False:
                resolved_label = "false"
            else:
                resolved_label = "unknown"

            rows.append(
                {
                    "source": source,
                    "eval_session": eval_session,
                    "run_id": run_id,
                    "instance_id": instance_id,
                    "resolved": resolved_label,
                    "patch_exists": patch_exists,
                    "patch_successfully_applied": patch_applied,
                    "report_path": str(report_path.relative_to(ROOT)),
                }
            )

            key = (source, eval_session, run_id)
            aggregate[key]["instances"] += 1
            if resolved is True:
                aggregate[key]["resolved_true"] += 1
            elif resolved is False:
                aggregate[key]["resolved_false"] += 1
            else:
                aggregate[key]["resolved_unknown"] += 1
            if patch_exists:
                aggregate[key]["patch_exists_true"] += 1
            if patch_applied:
                aggregate[key]["patch_applied_true"] += 1

    agg_rows: list[dict[str, Any]] = []
    for (source, eval_session, run_id), stats in sorted(aggregate.items()):
        instances = _safe_int(stats["instances"])
        resolved_true = _safe_int(stats["resolved_true"])
        patch_applied = _safe_int(stats["patch_applied_true"])
        agg_rows.append(
            {
                "source": source,
                "eval_session": eval_session,
                "run_id": run_id,
                "instances": instances,
                "resolved_true": resolved_true,
                "resolved_false": _safe_int(stats["resolved_false"]),
                "resolved_unknown": _safe_int(stats["resolved_unknown"]),
                "patch_exists_true": _safe_int(stats["patch_exists_true"]),
                "patch_applied_true": patch_applied,
                "resolved_rate_pct": _pct(resolved_true, instances),
                "patch_applied_rate_pct": _pct(patch_applied, instances),
            }
        )

    return rows, agg_rows, report_file_count


def collect_trajectory_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not TRAJ_RUNS_ROOT.exists():
        return rows

    for run_dir in sorted([p for p in TRAJ_RUNS_ROOT.iterdir() if p.is_dir()]):
        traj_count = 0
        pred_count = 0
        debug_log_count = 0
        info_log_count = 0
        trace_log_count = 0

        for file_path in run_dir.rglob("*"):
            if not file_path.is_file():
                continue
            name = file_path.name
            if name.endswith(".traj"):
                traj_count += 1
            elif name.endswith(".pred"):
                pred_count += 1
            elif name.endswith(".debug.log"):
                debug_log_count += 1
            elif name.endswith(".info.log"):
                info_log_count += 1
            elif name.endswith(".trace.log"):
                trace_log_count += 1

        rows.append(
            {
                "trajectory_run": run_dir.name,
                "has_run_batch_config": (run_dir / "run_batch.config.yaml").exists(),
                "traj_files": traj_count,
                "pred_files": pred_count,
                "debug_logs": debug_log_count,
                "info_logs": info_log_count,
                "trace_logs": trace_log_count,
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *data_lines])


def write_markdown_summary(
    eval_rows: list[dict[str, Any]],
    instance_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    report_file_count: int,
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    total_instances = len(instance_rows)
    resolved_true = sum(1 for row in instance_rows if row["resolved"] == "true")
    resolved_false = sum(1 for row in instance_rows if row["resolved"] == "false")
    resolved_unknown = sum(1 for row in instance_rows if row["resolved"] == "unknown")

    total_traj_files = len(list(TRAJ_ROOT.rglob("*.traj"))) if TRAJ_ROOT.exists() else 0

    top_eval = sorted(eval_rows, key=lambda r: _safe_int(r["resolved_instances"]), reverse=True)[:10]
    top_agg = sorted(aggregate_rows, key=lambda r: float(r["resolved_rate_pct"]), reverse=True)[:10]

    lines: list[str] = []
    lines.append("# SWE-agent Results Summary")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Eval summary JSON files: {len(eval_rows)}")
    lines.append(f"- Report files scanned: {report_file_count}")
    lines.append(f"- Instance-level records: {total_instances}")
    lines.append(f"- Resolved true/false/unknown: {resolved_true}/{resolved_false}/{resolved_unknown}")
    lines.append(f"- Trajectory run directories: {len(trajectory_rows)}")
    lines.append(f"- Total `.traj` files: {total_traj_files}")
    lines.append("")

    if top_eval:
        lines.append("## Top Eval Runs (by resolved_instances)")
        lines.append("")
        lines.append(
            _md_table(
                [
                    "run_id",
                    "resolved",
                    "completed",
                    "submitted",
                    "total",
                    "resolved/total %",
                    "file",
                ],
                [
                    [
                        row["run_id"],
                        row["resolved_instances"],
                        row["completed_instances"],
                        row["submitted_instances"],
                        row["total_instances"],
                        row["resolved_over_total_pct"],
                        row["file_name"],
                    ]
                    for row in top_eval
                ],
            )
        )
        lines.append("")

    if top_agg:
        lines.append("## Top Report Aggregates (by resolved_rate_pct)")
        lines.append("")
        lines.append(
            _md_table(
                [
                    "source",
                    "eval_session",
                    "run_id",
                    "instances",
                    "resolved_true",
                    "resolved_rate %",
                ],
                [
                    [
                        row["source"],
                        row["eval_session"],
                        row["run_id"],
                        row["instances"],
                        row["resolved_true"],
                        row["resolved_rate_pct"],
                    ]
                    for row in top_agg
                ],
            )
        )
        lines.append("")

    (OUT_DIR / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_rows = collect_eval_summaries()
    instance_rows, aggregate_rows, report_file_count = collect_instance_reports()
    trajectory_rows = collect_trajectory_runs()

    write_csv(
        OUT_DIR / "eval_summaries_overview.csv",
        eval_rows,
        [
            "file_name",
            "run_id",
            "eval_tag",
            "total_instances",
            "submitted_instances",
            "completed_instances",
            "resolved_instances",
            "unresolved_instances",
            "error_instances",
            "empty_patch_instances",
            "resolved_over_total_pct",
            "resolved_over_submitted_pct",
            "completed_over_total_pct",
        ],
    )

    write_csv(
        OUT_DIR / "instance_reports.csv",
        instance_rows,
        [
            "source",
            "eval_session",
            "run_id",
            "instance_id",
            "resolved",
            "patch_exists",
            "patch_successfully_applied",
            "report_path",
        ],
    )

    write_csv(
        OUT_DIR / "report_run_aggregates.csv",
        aggregate_rows,
        [
            "source",
            "eval_session",
            "run_id",
            "instances",
            "resolved_true",
            "resolved_false",
            "resolved_unknown",
            "patch_exists_true",
            "patch_applied_true",
            "resolved_rate_pct",
            "patch_applied_rate_pct",
        ],
    )

    write_csv(
        OUT_DIR / "trajectory_runs_overview.csv",
        trajectory_rows,
        [
            "trajectory_run",
            "has_run_batch_config",
            "traj_files",
            "pred_files",
            "debug_logs",
            "info_logs",
            "trace_logs",
        ],
    )

    write_markdown_summary(
        eval_rows=eval_rows,
        instance_rows=instance_rows,
        aggregate_rows=aggregate_rows,
        trajectory_rows=trajectory_rows,
        report_file_count=report_file_count,
    )

    print(f"Summary generated under: {OUT_DIR}")


if __name__ == "__main__":
    main()
