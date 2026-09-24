#!/usr/bin/env python3
"""Run the frozen binary-backend workloads and print a scoreboard.

Every row is one run of the `prove_hash_binary` example: the same objectives, the same
heights, at one thread and at every thread the machine has. The example reports its own
timings and security as JSON; peak memory comes from wrapping the run, so nothing has to
be measured from inside the prover.

    python3 scripts/scoreboard.py --format markdown
    python3 scripts/scoreboard.py --format json > scoreboard.json

Comparing two machines means running the same command on each. The workloads are frozen
here rather than passed in, so a scoreboard is not quietly measured on a different shape
from the one it is compared against.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
from typing import Any, Iterator, Sequence


# One entry per frozen workload: the objective, and the heights it is measured at.
#
# Keccak-f spends 25 rows on one permutation, so its heights are four higher than the
# compression objectives' to prove a comparable number of hashes. The heights are chosen
# to run in seconds rather than minutes: large enough that setup does not dominate, small
# enough that a full scoreboard is one coffee rather than one afternoon.
WORKLOADS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("keccak-f-permutations", (14, 16)),
    ("blake-3-compressions", (10, 12)),
    ("sha-256-compressions", (10, 12)),
    ("blake-2s-compressions", (10, 12)),
)

# Rows one hash occupies, per objective. Keccak-f spends a row on each round plus one
# for the output; a compression fits on a single row. Dividing the trace by this gives
# the hashes a run actually proves, which is the figure published comparisons quote.
ROWS_PER_HASH = {"keccak-f-permutations": 25}

# The security every workload is proved at, so a faster row is never a weaker one.
SECURITY_BITS = 96

# Thread counts every workload runs at. `None` means every thread the machine has.
THREAD_COUNTS: tuple[int | None, ...] = (1, None)

# Peak resident set size, as the host's `time` utility reports it.
#
#     macOS   ->  "maximum resident set size" in bytes
#     GNU     ->  "Maximum resident set size (kbytes)"
PEAK_RSS_PATTERNS = (
    (re.compile(r"(\d+)\s+maximum resident set size"), 1),
    (re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)"), 1024),
)


def time_flag(system: str) -> str:
    """The flag that makes the host's `time` report peak memory.

    macOS reports it under `-l`, GNU under `-v`, and the two spell the line differently
    as well, which is why both are parsed.
    """
    return "-l" if system == "Darwin" else "-v"


def available_objectives(help_text: str) -> set[str]:
    """The objectives this build of the example accepts.

    A workload naming an objective the build does not have is skipped rather than run,
    so one scoreboard script serves a tree where an objective has not landed yet.
    """
    found = re.search(r"possible values: ([^\]]+)\]", help_text)
    return {value.strip() for value in found.group(1).split(",")} if found else set()


def workload_rows(
    objectives: set[str] | None = None,
    quick: bool = False,
) -> Iterator[tuple[str, int, int | None]]:
    """Every (objective, height, threads) the scoreboard measures, in order.

    `quick` keeps only the smallest height of each workload, which is the shape a
    regression gate wants: the same proof and the same security as the full run, in a
    fraction of the time.
    """
    for objective, heights in WORKLOADS:
        if objectives is not None and objective not in objectives:
            continue
        for height in heights[:1] if quick else heights:
            for threads in THREAD_COUNTS:
                yield objective, height, threads


def peak_rss_bytes(measurements: str) -> int | None:
    """The peak resident set size in the `time` output, in bytes.

    Returns `None` when the output holds no line this understands, so an unfamiliar
    `time` costs the memory column rather than the whole run.
    """
    for pattern, scale in PEAK_RSS_PATTERNS:
        found = pattern.search(measurements)
        if found:
            return int(found.group(1)) * scale
    return None


def prover_command(objective: str, height: int, binary: str | None) -> list[str]:
    """The command that proves one workload and prints its report as JSON."""
    run = (
        [binary]
        if binary
        else [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--package",
            "p3-examples",
            "--features",
            "parallel",
            "--example",
            "prove_hash_binary",
            "--",
        ]
    )
    return run + [
        "--objective",
        objective,
        "--log-trace-length",
        str(height),
        "--security-bits",
        str(SECURITY_BITS),
        "--format",
        "json",
    ]


def measure(
    objective: str, height: int, threads: int | None, binary: str | None, time_path: str
) -> dict[str, Any]:
    """Prove one workload once and return its report, with peak memory added.

    The report is the example's own JSON, so the timings and the security are the
    prover's rather than this script's reading of them.
    """
    environment = dict(os.environ)
    if threads is not None:
        environment["RAYON_NUM_THREADS"] = str(threads)

    command = [time_path, time_flag(platform.system())] + prover_command(
        objective, height, binary
    )
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{objective} at 2^{height} failed:\n{completed.stderr}"
        )

    report = json.loads(completed.stdout)
    report["objective"] = objective
    report["log_trace_length"] = height
    report["peak_rss_bytes"] = peak_rss_bytes(completed.stderr)
    return report


def hashes_proved(objective: str, height: int) -> int:
    """How many hashes one run of this workload proves."""
    return (1 << height) // ROWS_PER_HASH.get(objective, 1)


def throughput(report: dict[str, Any]) -> float:
    """Hashes proved per second of proving, the figure published comparisons quote."""
    hashes = hashes_proved(report["objective"], report["log_trace_length"])
    return hashes / report["prove_seconds"]


def gate_failures(
    reports: Sequence[dict[str, Any]],
    baseline: dict[str, Any],
    tolerance: float,
) -> list[str]:
    """Every way this run falls short of the baseline it is gated against.

    Proof size and security are exact. The same workload at the same parameters emits
    the same proof on every machine and at every thread count, so a change there is a
    change in the protocol rather than in the weather. That is also why the baseline is
    keyed on the workload alone: a machine with a different core count must still be
    held to the same proof.

    Proving time is not portable, so it is given `tolerance` slack and gated only where
    a row carries a budget for it.
    """
    budgets = {
        (run["objective"], run["log_trace_length"]): run for run in baseline["runs"]
    }
    failures = []

    # A proof that changes with the thread count is a bug on its own, whatever the
    # baseline says, so the run is checked against itself first.
    proofs: dict[tuple[str, int], tuple[int, int]] = {}
    for report in reports:
        key = (report["objective"], report["log_trace_length"])
        seen = proofs.setdefault(key, (report["proof_bytes"], report["threads"]))
        if seen[0] != report["proof_bytes"]:
            failures.append(
                f"{key[0]} at 2^{key[1]}: proof is {report['proof_bytes']} bytes on "
                f"{report['threads']} thread(s) but {seen[0]} on {seen[1]}"
            )

    for report in reports:
        budget = budgets.get((report["objective"], report["log_trace_length"]))
        if budget is None:
            continue
        where = (
            f"{report['objective']} at 2^{report['log_trace_length']} on "
            f"{report['threads']} thread(s)"
        )

        if report["proof_bytes"] != budget["proof_bytes"]:
            failures.append(
                f"{where}: proof is {report['proof_bytes']} bytes, baseline is "
                f"{budget['proof_bytes']}"
            )
        if report["security_bits"] < budget["security_bits"] - 0.01:
            failures.append(
                f"{where}: security is {report['security_bits']:.2f} bits, baseline is "
                f"{budget['security_bits']:.2f}"
            )
        allowed = budget.get("prove_seconds_budget")
        if allowed is not None and report["prove_seconds"] > allowed * tolerance:
            failures.append(
                f"{where}: proving took {report['prove_seconds']:.3f} s, budget is "
                f"{allowed:.3f} s with {tolerance}x slack"
            )
    return failures


def machine() -> dict[str, str]:
    """What the scoreboard was measured on, so two of them can be told apart."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }


def markdown(reports: Sequence[dict[str, Any]]) -> str:
    """The scoreboard as one table, with the security report beside every row."""
    header = (
        "| Objective | Rows | Width | Threads | Witness | Prove | Hashes/s | Verify | "
        "Serialize | Proof | Peak RSS | Security |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for report in reports:
        peak = report.get("peak_rss_bytes")
        lines.append(
            "| {objective} | 2^{height} | {width} | {threads} | {witness:.3f} s | "
            "{prove:.3f} s | {rate:,.0f} | {verify:.3f} s | {serialize:.3f} s | "
            "{proof:.0f} KB | {peak} | {security:.1f} |".format(
                objective=report["objective"],
                height=report["log_trace_length"],
                width=report["width"],
                threads=report["threads"],
                witness=report["witness_seconds"],
                prove=report["prove_seconds"],
                rate=throughput(report),
                verify=report["verify_seconds"],
                serialize=report["serialize_seconds"],
                proof=report["proof_bytes"] / 1024,
                peak=f"{peak / (1 << 20):.0f} MiB" if peak else "n/a",
                security=report["security_bits"],
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="how to print the scoreboard (default: markdown)",
    )
    parser.add_argument(
        "--binary",
        help="an already built prove_hash_binary to run, instead of building it",
    )
    parser.add_argument(
        "--time",
        default="/usr/bin/time",
        help="the time utility that reports peak memory (default: /usr/bin/time)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="measure only the smallest height of each workload",
    )
    parser.add_argument(
        "--gate",
        type=Path,
        help="a scoreboard to gate this run against, failing on any regression",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.5,
        help="slack allowed on a proving-time budget (default: 1.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands the scoreboard would run, and run none of them",
    )
    args = parser.parse_args(argv)

    objectives = None
    if not args.dry_run:
        help_text = subprocess.run(
            prover_command("keccak-f-permutations", 10, args.binary)[:-8] + ["--help"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        objectives = available_objectives(help_text) or None
        for objective, _ in WORKLOADS:
            if objectives is not None and objective not in objectives:
                print(f"skipping {objective}: this build has no such objective", file=sys.stderr)

    if args.dry_run:
        for objective, height, threads in workload_rows(quick=args.quick):
            prefix = f"RAYON_NUM_THREADS={threads} " if threads is not None else ""
            command = shlex.join(prover_command(objective, height, args.binary))
            print(f"{prefix}{command}")
        return 0

    reports = [
        measure(objective, height, threads, args.binary, args.time)
        for objective, height, threads in workload_rows(objectives, args.quick)
    ]

    if args.format == "json":
        print(json.dumps({"machine": machine(), "runs": reports}, indent=2))
    else:
        print(markdown(reports))

    if args.gate is not None:
        failures = gate_failures(
            reports, json.loads(args.gate.read_text()), args.tolerance
        )
        for failure in failures:
            print(f"regression: {failure}", file=sys.stderr)
        if failures:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
