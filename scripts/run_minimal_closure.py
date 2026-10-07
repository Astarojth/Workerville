#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run minimal closure matrix episodes in sequence.")
    p.add_argument(
        "--configs-root",
        default=str(ROOT / "configs" / "tasks"),
        help="Root directory containing generated task config folders",
    )
    p.add_argument(
        "--run-root",
        default=str(ROOT / "runs" / "minimal_closure"),
        help="Output root for run artifacts",
    )
    p.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Optional task id filter",
    )
    p.add_argument(
        "--task-limit",
        type=int,
        default=0,
        help="Use first N tasks after filtering (0 = all tasks)",
    )
    p.add_argument("--env-file", default=".env", help="Environment file path for run_experiment")
    p.add_argument("--api-config", default="api.config.yaml", help="API config path for run_experiment")
    p.add_argument(
        "--run-preset",
        choices=("off", "fast", "balanced", "stress"),
        default="off",
        help="Optional run preset forwarded to run_experiment",
    )
    p.add_argument("--max-runs", type=int, default=None, help="Optional cap for debug")
    p.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index")
    p.add_argument("--shard-count", type=int, default=1, help="Total number of shards")
    p.add_argument("--rerun", action="store_true", help="Rerun even if metrics already exist")
    p.add_argument("--stop-on-failure", action="store_true", help="Stop immediately on first failed episode")
    p.add_argument("--retry-max", type=int, default=2, help="Max retry attempts per episode")
    p.add_argument(
        "--allow-openclaw-error",
        action="store_true",
        help="Accept outputs that contain openclaw_error markers (not recommended)",
    )
    p.add_argument(
        "--clear-openclaw-locks",
        dest="clear_openclaw_locks",
        action="store_true",
        default=True,
        help="Clear stale OpenClaw session lock files before each attempt (default: on)",
    )
    p.add_argument(
        "--no-clear-openclaw-locks",
        dest="clear_openclaw_locks",
        action="store_false",
        help="Do not clear OpenClaw session lock files before each attempt",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configs_root = Path(args.configs_root)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    tasks = _discover_tasks(configs_root)
    if not tasks:
        raise SystemExit(f"no task folders found under {configs_root}")

    selected_tasks = _select_tasks(tasks, task_ids=args.task_ids, task_limit=args.task_limit)
    run_list = _discover_runs(configs_root=configs_root, task_ids=selected_tasks)
    run_list = _apply_shard(run_list, shard_index=args.shard_index, shard_count=args.shard_count)
    if args.max_runs is not None and args.max_runs > 0:
        run_list = run_list[: int(args.max_runs)]
    if not run_list:
        raise SystemExit("no config files selected")

    print(
        f"[minimal-closure] selected tasks={len(selected_tasks)} "
        f"episodes={len(run_list)} shard={int(args.shard_index)}/{int(args.shard_count)} "
        f"rerun={bool(args.rerun)} preset={args.run_preset}"
    )

    summary: dict[str, Any] = {
        "schema": "minimal_closure_run_summary_v1",
        "configs_root": str(configs_root),
        "run_root": str(run_root),
        "selected_tasks": selected_tasks,
        "planned_runs": len(run_list),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "rows": [],
        "S_totals_sum": {"S1": 0, "S2": 0, "S3": 0},
        "risk_totals_sum": {},
        "task_breakdown": {},
    }

    for idx, cfg_path in enumerate(run_list, start=1):
        task_id = cfg_path.parent.name
        cfg_stem = cfg_path.stem
        out_dir = run_root / task_id / cfg_stem
        metrics_path = out_dir / "analysis" / "metrics.json"
        if metrics_path.exists() and not args.rerun:
            metrics = _load_json(metrics_path)
            _append_row(summary, task_id, cfg_stem, out_dir, status="skipped", metrics=metrics, returncode=0)
            print(f"[{idx:03d}/{len(run_list)}] skipped {task_id}/{cfg_stem} (already exists)")
            continue

        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            str(cfg_path),
            "--env-file",
            str(args.env_file),
            "--api-config",
            str(args.api_config),
            "--output-dir",
            str(out_dir),
        ]
        if args.run_preset != "off":
            cmd.extend(["--run-preset", args.run_preset])

        print(f"[{idx:03d}/{len(run_list)}] running {task_id}/{cfg_stem}")
        attempts_used = 0
        fail_reason = ""
        metrics: dict[str, Any] = {}
        last_proc: subprocess.CompletedProcess[str] | None = None
        success = False
        for attempt in range(max(0, int(args.retry_max)) + 1):
            attempts_used = attempt + 1
            _reset_out_dir(out_dir)
            if args.clear_openclaw_locks:
                _clear_openclaw_session_locks()
            out_dir.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            last_proc = proc
            (out_dir / "runner.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (out_dir / "runner.stderr.log").write_text(proc.stderr or "", encoding="utf-8")

            if proc.returncode != 0:
                fail_reason = f"run_experiment_rc_{proc.returncode}"
                if attempt < int(args.retry_max):
                    wait_sec = _retry_wait_seconds((proc.stderr or "") + "\n" + (proc.stdout or ""))
                    print(
                        f"  -> retry {attempt + 1}/{int(args.retry_max)} after rc={proc.returncode} "
                        f"(sleep {wait_sec:.0f}s)"
                    )
                    time.sleep(wait_sec)
                    continue
                break
            if not metrics_path.exists():
                fail_reason = "missing_metrics_json"
                if attempt < int(args.retry_max):
                    print(f"  -> retry {attempt + 1}/{int(args.retry_max)} after missing metrics")
                    continue
                break

            metrics = _load_json(metrics_path)
            if not args.allow_openclaw_error and _has_openclaw_error(out_dir):
                fail_reason = "openclaw_error_marker_detected"
                if attempt < int(args.retry_max):
                    print(f"  -> retry {attempt + 1}/{int(args.retry_max)} after openclaw error marker")
                    continue
                break

            success = True
            break

        if not success:
            rc = last_proc.returncode if last_proc is not None else 98
            _append_row(
                summary,
                task_id,
                cfg_stem,
                out_dir,
                status="failed",
                metrics=metrics,
                returncode=rc,
                attempts=attempts_used,
                fail_reason=fail_reason,
                stdout_tail=_tail(last_proc.stdout if last_proc is not None else "", 40),
                stderr_tail=_tail(last_proc.stderr if last_proc is not None else "", 40),
            )
            print(f"  -> failed after {attempts_used} attempt(s): {fail_reason}")
            if args.stop_on_failure:
                break
            continue

        _append_row(
            summary,
            task_id,
            cfg_stem,
            out_dir,
            status="completed",
            metrics=metrics,
            returncode=0,
            attempts=attempts_used,
        )
        s1, s2, s3 = _extract_s_counts(metrics)
        print(f"  -> ok totals(S1,S2,S3)=({s1},{s2},{s3}) attempts={attempts_used}")

    if int(args.shard_count) > 1:
        summary_path = run_root / f"summary.shard{int(args.shard_index):02d}.json"
        markdown_path = run_root / f"summary.shard{int(args.shard_index):02d}.md"
    else:
        summary_path = run_root / "summary.json"
        markdown_path = run_root / "summary.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown_summary(summary), encoding="utf-8")
    print(f"[minimal-closure] summary json: {summary_path}")
    print(f"[minimal-closure] summary md:   {markdown_path}")
    print(
        f"[minimal-closure] completed={summary['completed']} skipped={summary['skipped']} failed={summary['failed']}"
    )

    if int(summary["failed"]) > 0:
        raise SystemExit(1)


def _discover_tasks(configs_root: Path) -> list[str]:
    if not configs_root.exists():
        return []
    rows = [p.name for p in sorted(configs_root.iterdir()) if p.is_dir()]
    return [x for x in rows if x != "__pycache__"]


def _select_tasks(tasks: list[str], task_ids: list[str] | None, task_limit: int) -> list[str]:
    selected = list(tasks)
    if task_ids:
        wanted = set(task_ids)
        selected = [x for x in selected if x in wanted]
    if task_limit > 0:
        selected = selected[: task_limit]
    return selected


def _apply_shard(run_list: list[Path], shard_index: int, shard_count: int) -> list[Path]:
    count = int(shard_count)
    index = int(shard_index)
    if count <= 1:
        return list(run_list)
    if index < 0 or index >= count:
        raise SystemExit(f"shard-index must be in [0, {count})")
    return list(run_list)[index::count]


def _discover_runs(configs_root: Path, task_ids: list[str]) -> list[Path]:
    files: list[Path] = []
    for task_id in task_ids:
        task_dir = configs_root / task_id
        if not task_dir.exists():
            continue
        task_files = sorted(task_dir.glob("*.yaml"))
        files.extend(task_files)
    return files


def _append_row(
    summary: dict[str, Any],
    task_id: str,
    cfg_stem: str,
    out_dir: Path,
    status: str,
    metrics: dict[str, Any],
    returncode: int,
    attempts: int = 1,
    fail_reason: str = "",
    stdout_tail: list[str] | None = None,
    stderr_tail: list[str] | None = None,
) -> None:
    row = {
        "task_id": task_id,
        "config": cfg_stem,
        "status": status,
        "returncode": int(returncode),
        "attempts": int(attempts),
        "fail_reason": str(fail_reason),
        "out_dir": str(out_dir),
        "S_totals": _normalize_s_totals(metrics),
        "S_endpoints": _normalize_s_endpoints(metrics),
        "risk_totals": _normalize_risk_totals(metrics),
    }
    if stdout_tail:
        row["stdout_tail"] = stdout_tail
    if stderr_tail:
        row["stderr_tail"] = stderr_tail
    summary["rows"].append(row)

    if status == "completed":
        summary["completed"] = int(summary["completed"]) + 1
    elif status == "skipped":
        summary["skipped"] = int(summary["skipped"]) + 1
    else:
        summary["failed"] = int(summary["failed"]) + 1

    _merge_totals(summary, row)
    _task_breakdown(summary, row)


def _merge_totals(summary: dict[str, Any], row: dict[str, Any]) -> None:
    if row.get("status") not in {"completed", "skipped"}:
        return
    s_totals = row.get("S_totals", {})
    if isinstance(s_totals, dict):
        s_agg = summary.setdefault("S_totals_sum", {"S1": 0, "S2": 0, "S3": 0})
        for key in ("S1", "S2", "S3"):
            s_agg[key] = int(s_agg.get(key, 0)) + int(s_totals.get(key, 0))
    risk_totals = row.get("risk_totals", {})
    if isinstance(risk_totals, dict):
        risk_agg = summary.setdefault("risk_totals_sum", {})
        for k, v in risk_totals.items():
            risk_agg[str(k)] = int(risk_agg.get(str(k), 0)) + int(v)


def _task_breakdown(summary: dict[str, Any], row: dict[str, Any]) -> None:
    task_id = str(row.get("task_id", ""))
    table = summary.setdefault("task_breakdown", {})
    task_row = table.setdefault(
        task_id,
        {
            "completed": 0,
            "skipped": 0,
            "failed": 0,
            "runs_with_any_risk": 0,
            "runs": 0,
        },
    )
    status = str(row.get("status", ""))
    task_row["runs"] = int(task_row.get("runs", 0)) + 1
    if status == "completed":
        task_row["completed"] = int(task_row.get("completed", 0)) + 1
    elif status == "skipped":
        task_row["skipped"] = int(task_row.get("skipped", 0)) + 1
    else:
        task_row["failed"] = int(task_row.get("failed", 0)) + 1

    s_totals = row.get("S_totals", {})
    if isinstance(s_totals, dict):
        if int(s_totals.get("S1", 0)) + int(s_totals.get("S2", 0)) + int(s_totals.get("S3", 0)) > 0:
            task_row["runs_with_any_risk"] = int(task_row.get("runs_with_any_risk", 0)) + 1


def _extract_s_counts(metrics: dict[str, Any]) -> tuple[int, int, int]:
    if not isinstance(metrics, dict):
        return 0, 0, 0
    s_endpoints = metrics.get("S_endpoints", {})
    if isinstance(s_endpoints, dict):
        s1 = int(((s_endpoints.get("S1", {}) or {}).get("count", 0)))
        s2 = int(((s_endpoints.get("S2", {}) or {}).get("count", 0)))
        s3 = int(((s_endpoints.get("S3", {}) or {}).get("count", 0)))
        return s1, s2, s3
    totals = metrics.get("totals", {})
    if isinstance(totals, dict):
        if "S1" in totals or "S2" in totals or "S3" in totals:
            return int(totals.get("S1", 0)), int(totals.get("S2", 0)), int(totals.get("S3", 0))
        # Backward compatibility with older artifacts.
        return int(totals.get("r1", 0)), int(totals.get("r2", 0)), int(totals.get("r3", 0))
    return 0, 0, 0


def _normalize_s_totals(metrics: dict[str, Any]) -> dict[str, int]:
    s1, s2, s3 = _extract_s_counts(metrics)
    return {"S1": int(s1), "S2": int(s2), "S3": int(s3)}


def _normalize_s_endpoints(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    s_endpoints = metrics.get("S_endpoints", {}) if isinstance(metrics, dict) else {}
    if isinstance(s_endpoints, dict):
        s1 = (s_endpoints.get("S1", {}) or {}) if isinstance(s_endpoints.get("S1", {}), dict) else {}
        s2 = (s_endpoints.get("S2", {}) or {}) if isinstance(s_endpoints.get("S2", {}), dict) else {}
        s3 = (s_endpoints.get("S3", {}) or {}) if isinstance(s_endpoints.get("S3", {}), dict) else {}
        return {
            "S1": {"source_metric": "S1_leakage", "count": int(s1.get("count", 0))},
            "S2": {"source_metric": "S2_high_risk_action", "count": int(s2.get("count", 0))},
            "S3": {"source_metric": "S3_negative_compliance", "count": int(s3.get("count", 0))},
        }
    totals = _normalize_s_totals(metrics)
    return {
        "S1": {"source_metric": "S1_leakage", "count": int(totals["S1"])},
        "S2": {"source_metric": "S2_high_risk_action", "count": int(totals["S2"])},
        "S3": {"source_metric": "S3_negative_compliance", "count": int(totals["S3"])},
    }


def _normalize_risk_totals(metrics: dict[str, Any]) -> dict[str, int]:
    if not isinstance(metrics, dict):
        return {}
    rows = metrics.get("risk_totals", {})
    if isinstance(rows, dict) and rows:
        return {str(k): int(v) for k, v in rows.items()}
    legacy = metrics.get("totals_extended", {})
    if not isinstance(legacy, dict) or not legacy:
        return {}
    mapped: dict[str, int] = {}
    for key, value in legacy.items():
        k = str(key).strip()
        if not k.startswith("r"):
            continue
        idx = k[1:]
        if not idx.isdigit():
            continue
        mapped[f"R{idx}"] = int(value)
    return mapped


def _tail(text: str, n: int) -> list[str]:
    rows = (text or "").splitlines()
    return rows[-n:]


def _retry_wait_seconds(blob: str, default: float = 2.0) -> float:
    text = str(blob or "")
    retry_after = re.search(r"retry after (\d+)\s*s", text, flags=re.I)
    if retry_after:
        return float(min(int(retry_after.group(1)) + 2, 90))
    if "rate limit" in text.lower():
        return 40.0
    return float(default)


def _reset_out_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)


def _has_openclaw_error(out_dir: Path) -> bool:
    transcript_path = out_dir / "logs" / "transcripts.pretty.json"
    if not transcript_path.exists():
        return False
    text = transcript_path.read_text(encoding="utf-8")
    markers = ("openclaw_error_noop", "[openclaw_error]")
    return any(marker in text for marker in markers)


def _clear_openclaw_session_locks() -> None:
    roots = [
        Path("/root/.openclaw/agents"),
    ]
    deleted = 0
    for root in roots:
        if not root.exists():
            continue
        for lock_path in root.rglob("*.lock"):
            try:
                lock_path.unlink(missing_ok=True)
                deleted += 1
            except Exception:
                continue
    if deleted > 0:
        time.sleep(0.1)


def _render_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Minimal Closure Summary",
        "",
        f"- planned_runs: {summary.get('planned_runs', 0)}",
        f"- completed: {summary.get('completed', 0)}",
        f"- skipped: {summary.get('skipped', 0)}",
        f"- failed: {summary.get('failed', 0)}",
        "",
        "## Task Breakdown",
        "",
        "| task_id | runs | completed | skipped | failed | runs_with_any_risk |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    breakdown = summary.get("task_breakdown", {})
    if isinstance(breakdown, dict):
        for task_id, row in sorted(breakdown.items()):
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {task_id} | {int(row.get('runs', 0))} | {int(row.get('completed', 0))} | "
                f"{int(row.get('skipped', 0))} | {int(row.get('failed', 0))} | {int(row.get('runs_with_any_risk', 0))} |"
            )

    s_totals = summary.get("S_totals_sum", {})
    risk_totals = summary.get("risk_totals_sum", {})
    lines.extend(
        [
            "",
            "## S Totals (sum)",
            "",
            "```json",
            json.dumps(s_totals, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Risk Totals (sum)",
            "",
            "```json",
            json.dumps(risk_totals, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
