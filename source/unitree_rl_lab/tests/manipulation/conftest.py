"""Pytest hooks for manipulation test suite — per-test outcome capture + report write."""

from __future__ import annotations

import json
import pathlib
import time

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def pytest_runtest_makereport(item, call):
    """Capture per-test outcome into session._valve_results."""
    if call.when == "call":
        if not hasattr(item.session, "_valve_results"):
            item.session._valve_results = {}
        outcome = "passed" if call.excinfo is None else "failed"
        message = str(call.excinfo.value) if call.excinfo is not None else ""
        item.session._valve_results[item.nodeid] = {
            "outcome": outcome,
            "message": message,
        }


def pytest_sessionfinish(session, exitstatus):
    """Write structured JSON + human-readable report to results/ after session."""
    store: dict = getattr(session, "_valve_results", {})
    if not store:
        return

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    report_passed = [k for k, v in store.items() if v["outcome"] == "passed"]
    report_failed = [k for k, v in store.items() if v["outcome"] in ("failed", "error")]

    # Infer module number from first nodeid (e.g. test_09_... → 09)
    first_key = next(iter(store))
    try:
        module_num = int(first_key.split("test_")[1][:2])
    except (IndexError, ValueError):
        module_num = 0

    data = {
        "module": module_num,
        "title": f"Module #{module_num:02d} test results",
        "timestamp": timestamp,
        "exit_status": int(exitstatus),
        "summary": {
            "total": len(store),
            "passed": len(report_passed),
            "failed": len(report_failed),
        },
        "results": store,
    }

    tag = f"module_{module_num:02d}"
    (RESULTS_DIR / f"{tag}_report.json").write_text(json.dumps(data, indent=2))

    lines = [
        f"Module #{module_num:02d} — {data['title']}",
        f"Run at : {timestamp}",
        f"Status : {'PASS' if exitstatus == 0 else 'FAIL'}",
        f"Total  : {data['summary']['total']}  "
        f"Passed: {data['summary']['passed']}  "
        f"Failed: {data['summary']['failed']}",
        "",
        "--- Results ---",
    ]
    for nodeid, info in store.items():
        icon = "✓" if info["outcome"] == "passed" else "✗"
        lines.append(f"  {icon} {nodeid}")
        if info.get("message"):
            lines.append(f"      {info['message']}")

    (RESULTS_DIR / f"{tag}_report.txt").write_text("\n".join(lines) + "\n")
