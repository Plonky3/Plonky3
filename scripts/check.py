#!/usr/bin/env python3
"""Shared local and CI command recipes for the Plonky3 workspace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


LINT_COMMANDS = {
    "sort": ["cargo", "+stable", "sort", "--workspace", "--grouped", "--check"],
    "toml": ["taplo", "fmt", "--check"],
    "deps": ["cargo", "machete", "--with-metadata"],
    "clippy": ["cargo", "+stable", "clippy", "--all-targets", "--", "-D", "warnings"],
    "docs": [
        "cargo",
        "+stable",
        "doc",
        "--no-deps",
        "--workspace",
        "--document-private-items",
    ],
    "fmt": ["cargo", "+nightly", "fmt", "--all", "--", "--check"],
    "scripts": [sys.executable, "-m", "unittest", "scripts/test_check.py", "-v"],
}


def cargo_metadata(workspace_root: Path) -> dict[str, Any]:
    command = ["cargo", "metadata", "--no-deps", "--format-version", "1"]
    completed = subprocess.run(
        command,
        cwd=workspace_root,
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return json.loads(completed.stdout)


def workspace_packages(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    members = set(metadata["workspace_members"])
    return sorted(
        (package for package in metadata["packages"] if package["id"] in members),
        key=lambda package: package["name"],
    )


def plonky3_metadata(package: dict[str, Any]) -> dict[str, Any]:
    metadata = package.get("metadata") or {}
    value = metadata.get("plonky3", {})
    return value if isinstance(value, dict) else {}


def embedded_packages(metadata: dict[str, Any]) -> list[str]:
    return [
        package["name"]
        for package in workspace_packages(metadata)
        if any("lib" in target["kind"] for target in package["targets"])
        and plonky3_metadata(package).get("embedded", True) is not False
    ]


def package_test_features(
    metadata: dict[str, Any], package_name: str | None, parallel: bool
) -> list[tuple[str, str]]:
    runs = []
    for package in workspace_packages(metadata):
        if package_name is not None and package["name"] != package_name:
            continue
        ci = plonky3_metadata(package).get("ci", {})
        if not isinstance(ci, dict):
            continue
        features = ci.get("test-features", [])
        for feature in features:
            if not isinstance(feature, str):
                raise ValueError(
                    f"{package['name']} metadata.plonky3.ci.test-features entries must be strings"
                )
        if features:
            features = list(features)
            if parallel and "parallel" in package.get("features", {}):
                features.append("parallel")
            runs.append((package["name"], ",".join(features)))
    return runs


def package_args(package: str | None) -> list[str]:
    return ["-p", package] if package else []


def feature_args(parallel: bool) -> list[str]:
    return ["--features", "parallel"] if parallel else []


def test_commands(
    metadata: dict[str, Any], package: str | None, parallel: bool, doctest: bool
) -> list[list[str]]:
    if doctest:
        base = ["cargo", "test", "--doc"]
    else:
        base = ["cargo", "nextest", "run"]
    feature_runs = package_test_features(metadata, package, parallel)
    baseline = base + package_args(package) + feature_args(parallel)
    if package and feature_runs and not doctest:
        baseline += ["--no-tests", "warn"]
    commands = [baseline]
    for name, features in feature_runs:
        commands.append(base + ["-p", name, "--features", features])
    return commands


def lint_commands(check: str | None) -> list[list[str]]:
    if check:
        return [LINT_COMMANDS[check]]
    return list(LINT_COMMANDS.values())


def wasm_commands(step: str, build_target: str, run_target: str) -> list[list[str]]:
    commands = {
        "build": [
            "cargo",
            "build",
            "--verbose",
            "--target",
            build_target,
            "-p",
            "p3-goldilocks",
        ],
        "test": [
            "cargo",
            "test",
            "--release",
            "--target",
            run_target,
            "-p",
            "p3-goldilocks",
            "--lib",
            "wasm32_simd128",
        ],
        "smoke": [
            "cargo",
            "run",
            "--release",
            "--target",
            run_target,
            "--bin",
            "wasm_smoke",
            "-p",
            "p3-goldilocks",
        ],
        "bench": [
            "cargo",
            "run",
            "--release",
            "--target",
            run_target,
            "--bin",
            "wasm_bench",
            "-p",
            "p3-goldilocks",
        ],
        "merkle": [
            "cargo",
            "run",
            "--release",
            "--target",
            run_target,
            "--example",
            "wasm_merkle_bench",
            "-p",
            "p3-goldilocks",
        ],
    }
    return list(commands.values()) if step == "all" else [commands[step]]


def commands_for(args: argparse.Namespace) -> list[list[str]]:
    command = args.command
    metadata = None
    if command in {"full", "test", "doctest", "embedded"}:
        metadata = cargo_metadata(args.workspace_root)

    if command == "fast":
        scope = package_args(args.package) if args.package else ["--workspace"]
        return [["cargo", "check", *scope, "--all-targets", *feature_args(args.parallel)]]
    if command == "test":
        return test_commands(metadata, args.package, args.parallel, False)
    if command == "doctest":
        return test_commands(metadata, args.package, args.parallel, True)
    if command == "lint":
        return lint_commands(args.check)
    if command == "full":
        return [
            ["cargo", "check", "--workspace", "--all-targets"],
            *test_commands(metadata, None, False, False),
            *test_commands(metadata, None, False, True),
            *test_commands(metadata, None, True, False),
            *test_commands(metadata, None, True, True),
            *lint_commands(None),
        ]
    if command == "architecture":
        return [
            [
                "cargo",
                "build",
                "--target",
                args.target,
                "--all-targets",
                *feature_args(args.parallel),
            ]
        ]
    if command == "embedded":
        return [
            [
                "cargo",
                "build",
                "--verbose",
                "--target",
                args.target,
                "-p",
                package,
                "--lib",
            ]
            for package in embedded_packages(metadata)
        ]
    if command == "bench":
        return [
            [
                "cargo",
                "bench",
                "--workspace",
                "--exclude",
                "p3-dft",
                "--features",
                "parallel",
                "--",
                "--test",
            ]
        ]
    if command == "wasm":
        return wasm_commands(args.step, args.build_target, args.run_target)
    if command == "whir-exhaustive":
        return [
            [
                "cargo",
                "test",
                "-p",
                "p3-whir",
                *feature_args(args.parallel),
                "pcs::tests::test_whir_end_to_end_exhaustive",
                "--",
                "--ignored",
                "--exact",
            ]
        ]
    if command == "binary-large":
        return [
            [
                "cargo",
                "test",
                "-p",
                "p3-binary-pcs",
                "--release",
                "--test",
                "end_to_end",
                "large_configuration_2_16_round_trips",
                "--",
                "--ignored",
                "--exact",
            ]
        ]
    raise AssertionError(f"unhandled command: {command}")


def add_package_and_parallel(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--package", metavar="NAME", help="limit the command to one package")
    parser.add_argument("--parallel", action="store_true", help="enable the parallel feature")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dry-run", action="store_true", help="print commands without running them")
    result.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help=argparse.SUPPRESS,
    )
    subparsers = result.add_subparsers(dest="command", required=True)

    fast = subparsers.add_parser("fast", help="check host targets without running tests")
    add_package_and_parallel(fast)
    subparsers.add_parser("full", help="run all host checks used by CI")
    test = subparsers.add_parser("test", help="run tests with cargo-nextest")
    add_package_and_parallel(test)
    doctest = subparsers.add_parser("doctest", help="run Rust documentation tests")
    add_package_and_parallel(doctest)
    lint = subparsers.add_parser("lint", help="run formatting, lint, dependency and doc checks")
    lint.add_argument("--check", choices=LINT_COMMANDS, help="run one lint check")
    architecture = subparsers.add_parser(
        "architecture", help="compile all targets for a non-runnable architecture"
    )
    architecture.add_argument("--target", required=True)
    architecture.add_argument("--parallel", action="store_true")
    embedded = subparsers.add_parser(
        "embedded", help="build metadata-selected libraries for an embedded target"
    )
    embedded.add_argument("--target", default="thumbv7em-none-eabi")
    subparsers.add_parser("bench", help="run each benchmark body once")
    wasm = subparsers.add_parser("wasm", help="build or run wasm SIMD smoke coverage")
    wasm.add_argument(
        "--step", choices=("all", "build", "test", "smoke", "bench", "merkle"), default="all"
    )
    wasm.add_argument("--build-target", default="wasm32-unknown-unknown")
    wasm.add_argument("--run-target", default="wasm32-wasip1")
    whir = subparsers.add_parser("whir-exhaustive", help="run the ignored exhaustive WHIR sweep")
    whir.add_argument("--parallel", action="store_true")
    subparsers.add_parser("binary-large", help="run the ignored binary PCS 2^16 round trip")
    return result


def display(command: Sequence[str]) -> str:
    return shlex.join(command)


def command_environment(command: Sequence[str]) -> dict[str, str] | None:
    if command[:3] != ["cargo", "+stable", "doc"]:
        return None
    environment = os.environ.copy()
    deny_broken_links = "-D rustdoc::broken_intra_doc_links"
    current = environment.get("RUSTDOCFLAGS", "")
    if deny_broken_links not in current:
        environment["RUSTDOCFLAGS"] = f"{current} {deny_broken_links}".strip()
    return environment


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.workspace_root = args.workspace_root.resolve()
    try:
        commands = commands_for(args)
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
    for command in commands:
        print(f"+ {display(command)}", flush=True)
        if not args.dry_run:
            completed = subprocess.run(
                command,
                cwd=args.workspace_root,
                env=command_environment(command),
                check=False,
            )
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
