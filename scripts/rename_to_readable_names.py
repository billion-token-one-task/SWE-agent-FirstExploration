#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def safe_token(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-._") or "x"


def short_num(raw: str) -> str:
    raw = raw.strip()
    if raw.lower() == "none":
        return "none"
    try:
        f = float(raw)
        if f.is_integer():
            return str(int(f))
        s = f"{f:.4f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")
    except Exception:
        return safe_token(raw)


def model_alias(model: str) -> str:
    m = model.strip().lower()
    table = {
        "gpt-5.3-codex": "gpt53-codex",
        "gpt-5.4": "gpt54",
    }
    return table.get(m, safe_token(m))


def dataset_alias(dataset: str) -> str:
    d = dataset.strip().lower()
    table = {
        "lite_test_local83": "lite83",
        "local_instance": "local",
        "swe_bench_lite_dev": "swebench-lite-dev",
    }
    return table.get(d, safe_token(d))


def parse_run_id(run_id: str) -> dict[str, str] | None:
    # default__<model>__t-...__p-...__c-...___<dataset>__<run_label>
    if not run_id.startswith("default__") or "___" not in run_id:
        return None
    try:
        left, right = run_id.split("___", 1)
        left_parts = left.split("__")
        if len(left_parts) < 5:
            return None
        _, model, t_part, p_part, c_part = left_parts[:5]
        if "__" in right:
            dataset, run_label = right.split("__", 1)
        else:
            # Some probe runs only provide dataset without explicit run label
            dataset, run_label = right, "base"
    except ValueError:
        return None

    t = short_num(t_part.removeprefix("t-"))
    p = short_num(p_part.removeprefix("p-"))
    c = short_num(c_part.removeprefix("c-"))
    params = f"t{t}_p{p}_c{c}"

    m_alias = model_alias(model)
    d_alias = dataset_alias(dataset)

    run_norm = run_label.strip()
    if d_alias == "lite83" and run_norm.startswith("lite83_"):
        run_norm = run_norm[len("lite83_") :]
    run_norm = safe_token(run_norm)

    run_short = f"{d_alias}__{m_alias}__{run_norm}__{params}"
    return {
        "run_id": run_id,
        "model_alias": m_alias,
        "dataset_alias": d_alias,
        "run_label": run_label,
        "run_short": run_short,
        "params": params,
    }


def uniq_target(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    idx = 2
    while True:
        candidate = path.with_name(f"{stem}__v{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def rename_path(src: Path, dst: Path, mappings: list[tuple[str, str]]) -> Path:
    if src == dst:
        return src
    dst = uniq_target(dst)
    src.rename(dst)
    mappings.append((str(src.relative_to(ROOT)), str(dst.relative_to(ROOT))))
    return dst


def stage_from_eval_tag(tag: str, run_label: str) -> str:
    t = tag.lower()
    run = run_label.lower()

    if "partial_eval" in t:
        return "partial"

    if "_eval" in t:
        if "_eval_" in t:
            pre, post = t.split("_eval_", 1)
        else:
            pre, post = t.split("_eval", 1)
            post = post.lstrip("_")

        post = re.sub(r"(?:^|_)(\d{8})(_\d{3,6})?$", "", post).strip("_")

        pre_clean = pre
        if pre_clean.startswith("lite83_"):
            pre_clean = pre_clean[len("lite83_") :]
        run_clean = run
        if run_clean.startswith("lite83_"):
            run_clean = run_clean[len("lite83_") :]

        if pre_clean in {"smoke2", "lite83_smoke2"}:
            stage = "smoke2"
        elif pre_clean in {"submitted", "lite83_submitted"}:
            stage = "submitted"
        elif pre_clean == run_clean or pre_clean == "":
            stage = "main"
        else:
            stage = pre_clean

        if post:
            stage = post if stage == "main" else f"{stage}_{post}"

        return safe_token(stage)

    return "main"


def stage_from_session(session_name: str) -> str:
    s = session_name.lower()
    m = re.match(r"^session__.+__(.+?)(?:__attempt\d+)?$", s)
    if m:
        stage = re.sub(r"(?:^|_)(\d{8})(_\d{3,6})?$", "", m.group(1)).strip("_")
        return safe_token(stage or "main")
    if "partial_eval" in s:
        return "partial"
    if "_eval_" in s:
        pre, post = s.split("_eval_", 1)
        post = re.sub(r"(?:^|_)(\d{8})(_\d{3,6})?$", "", post).strip("_")
        if pre.endswith("smoke2"):
            base = "smoke2"
        elif pre.endswith("submitted"):
            base = "submitted"
        else:
            base = "main"
        if post:
            return safe_token(post if base == "main" else f"{base}_{post}")
        return base
    return "main"


def _looks_like_run_short(name: str) -> bool:
    return bool(re.match(r"^[a-z0-9-]+__[a-z0-9.-]+__.+__t[^_]+_p[^_]+_c[^_]+$", name))


def main() -> None:
    mappings: list[tuple[str, str]] = []

    eval_dir = ROOT / "experiments" / "eval_summaries"
    traj_dir = ROOT / "results" / "trajectories" / "ubuntu"
    hist_src_dir = ROOT / "results" / "evaluation_history" / "from_swe_agent_src"
    hist_ws_dir = ROOT / "results" / "evaluation_history" / "from_workspace_root"

    runlabel_to_short: dict[str, str] = {}

    # 1) Rename eval summary files
    for path in sorted(eval_dir.glob("*.json")):
        stem = path.stem
        if stem.startswith("summary__"):
            continue
        if ".lite" in stem:
            idx = stem.index(".lite")
            old_run_id, tag = stem[:idx], stem[idx + 1 :]
        elif ".partial_eval" in stem:
            idx = stem.index(".partial_eval")
            old_run_id, tag = stem[:idx], stem[idx + 1 :]
        elif "." in stem:
            old_run_id, tag = stem.split(".", 1)
        else:
            continue
        info = parse_run_id(old_run_id)
        if info is None:
            continue

        runlabel_to_short[info["run_label"].lower()] = info["run_short"]
        stage = stage_from_eval_tag(tag=tag, run_label=info["run_label"])
        new_name = f"summary__{info['run_short']}__{stage}.json"
        rename_path(path, path.with_name(new_name), mappings)

    # 2) Rename trajectory run dirs
    if traj_dir.exists():
        for run_path in sorted([p for p in traj_dir.iterdir() if p.is_dir()]):
            info = parse_run_id(run_path.name)
            if info is None:
                # Keep non-standard names but normalize obvious cmdprobe style
                if run_path.name == "cmdprobe2_norun":
                    new_name = "probe-setup__cmdprobe2_norun"
                    rename_path(run_path, run_path.with_name(new_name), mappings)
                continue
            rename_path(run_path, run_path.with_name(info["run_short"]), mappings)

    # 3) Rename from_swe_agent_src sessions and inner run dirs
    if hist_src_dir.exists():
        planned_session_names: dict[str, int] = {}
        for session in sorted([p for p in hist_src_dir.iterdir() if p.is_dir()]):
            inner_runs = sorted([p for p in session.iterdir() if p.is_dir()])

            session_run_short = None
            for inner in inner_runs:
                info = parse_run_id(inner.name)
                if info is None and _looks_like_run_short(inner.name):
                    session_run_short = inner.name
                    continue
                if info is None:
                    continue
                new_inner = rename_path(inner, inner.with_name(info["run_short"]), mappings)
                session_run_short = new_inner.name

            if session_run_short is None:
                # Infer from session name
                s_low = session.name.lower()
                matched_short = None
                if s_low.startswith("session__"):
                    rem = s_low[len("session__") :]
                    rem = rem.split("__attempt", 1)[0]
                    if "__" in rem:
                        cand = rem.rsplit("__", 1)[0]
                        if cand.startswith("session__"):
                            cand = cand[len("session__") :]
                        matched_short = cand
                # Longest-match first
                if matched_short is None:
                    for run_label, run_short in sorted(runlabel_to_short.items(), key=lambda x: len(x[0]), reverse=True):
                        if run_label in s_low:
                            matched_short = run_short
                            break
                if matched_short is None:
                    base = re.sub(r"_eval(_[^_]*)?(_\d{8}(?:_\d{3,6})?)?$", "", s_low)
                    matched_short = safe_token(base)
                session_run_short = matched_short

            stage = stage_from_session(session.name)
            base_name = f"session__{session_run_short}__{stage}"
            idx = planned_session_names.get(base_name, 0) + 1
            planned_session_names[base_name] = idx
            if idx > 1:
                base_name = f"{base_name}__attempt{idx}"

            rename_path(session, session.with_name(base_name), mappings)

    # 4) Rename from_workspace_root dirs and inner run dirs
    if hist_ws_dir.exists():
        for outer in sorted([p for p in hist_ws_dir.iterdir() if p.is_dir()]):
            outer_name = outer.name
            while outer_name.startswith("workspace__"):
                outer_name = outer_name[len("workspace__") :]
            outer_info = parse_run_id(outer_name)
            if outer_info:
                outer_short = outer_info["run_short"]
            elif _looks_like_run_short(outer_name):
                outer_short = outer_name
            else:
                outer_short = safe_token(outer_name)

            for inner in sorted([p for p in outer.iterdir() if p.is_dir()]):
                inner_info = parse_run_id(inner.name)
                if inner_info is None and _looks_like_run_short(inner.name):
                    continue
                if inner_info is None:
                    continue
                rename_path(inner, inner.with_name(inner_info["run_short"]), mappings)

            rename_path(outer, outer.with_name(f"workspace__{outer_short}"), mappings)

    # 5) Rename run script to readable name
    old_script = ROOT / "experiments" / "run_scripts" / "run_lite83_tok25k_mt8_probev3_closed_window.sh"
    if old_script.exists():
        rename_path(
            old_script,
            old_script.with_name("run_probe_lite83_tok25k_closed_window_mt8.sh"),
            mappings,
        )

    # 6) Write mapping
    out_map = ROOT / "experiments" / "notes" / "rename_map.tsv"
    out_map.parent.mkdir(parents=True, exist_ok=True)
    lines = ["old_path\tnew_path"]
    lines.extend(f"{old}\t{new}" for old, new in mappings)
    out_map.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Renamed entries: {len(mappings)}")
    print(f"Mapping file: {out_map}")


if __name__ == "__main__":
    main()
