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


def target_args(target: str | None) -> list[str]:
    """Build for one triple explicitly, which is what keeps pinned flags off host artifacts.

    Cargo applies the flag variables to build scripts and proc macros as well, unless a
    triple is named.

    A leg that pins instructions the runner may lack therefore has to name its own triple,
    even when that triple is the host.
    """
    return ["--target", target] if target else []


def test_commands(
    metadata: dict[str, Any],
    package: str | None,
    parallel: bool,
    doctest: bool,
    target: str | None = None,
) -> list[list[str]]:
    if doctest:
        base = ["cargo", "test", "--doc"]
    else:
        base = ["cargo", "nextest", "run"]
    base = base + target_args(target)
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
        return test_commands(metadata, args.package, args.parallel, False, args.target)
    if command == "doctest":
        return test_commands(metadata, args.package, args.parallel, True, args.target)
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


def add_optional_target(parser: argparse.ArgumentParser) -> None:
    """Accept an explicit build triple, which a leg pinning instructions needs."""
    parser.add_argument(
        "--target",
        default=None,
        metavar="TRIPLE",
        help="build for one triple, keeping pinned flags off host build scripts",
    )


def add_target_feature(parser: argparse.ArgumentParser) -> None:
    """Accept the target features a CI leg pins, so the same leg is reproducible locally."""
    parser.add_argument(
        "--target-feature",
        default=None,
        help="target features to pin, joined with an equals sign: --target-feature=+avx2",
    )


def attach_leading_minus_values(argv: Sequence[str]) -> list[str]:
    """Join a value that starts with a minus onto the option it belongs to.

    A feature is disabled by prefixing it with a minus, and the argument parser reads such a
    value as another option instead.

    Rewriting the pair into the equals form lets both spellings work, so a leg that turns a
    feature off is not a special case for the caller to remember.
    """
    joined = []
    argv = list(argv)
    index = 0
    while index < len(argv):
        argument = argv[index]
        takes_value = argument == "--target-feature" and index + 1 < len(argv)
        if takes_value and argv[index + 1].startswith("-"):
            joined.append(f"{argument}={argv[index + 1]}")
            index += 2
            continue
        joined.append(argument)
        index += 1
    return joined


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
    add_optional_target(test)
    add_target_feature(test)
    doctest = subparsers.add_parser("doctest", help="run Rust documentation tests")
    add_package_and_parallel(doctest)
    add_optional_target(doctest)
    add_target_feature(doctest)
    lint = subparsers.add_parser("lint", help="run formatting, lint, dependency and doc checks")
    lint.add_argument("--check", choices=LINT_COMMANDS, help="run one lint check")
    architecture = subparsers.add_parser(
        "architecture", help="compile all targets for a non-runnable architecture"
    )
    architecture.add_argument("--target", required=True)
    architecture.add_argument("--parallel", action="store_true")
    add_target_feature(architecture)
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


# Separator Cargo uses inside the encoded flag variables, one unit per argument.
ENCODED_SEPARATOR = "\x1f"


def append_compiler_flag(
    environment: dict[str, str], plain: str, encoded: str, flag: Sequence[str]
) -> str:
    """Put one compiler flag where Cargo will read it, and name the variable used.

    Cargo ignores the plain variable whenever the encoded one is set.

    So the flag has to follow whichever variable the caller chose, or it is silently dropped.

    The flag is appended, never merged into an existing setting.

    For a target feature that is what makes the request effective, since the last setting of
    a given feature is the one that takes effect.
    """
    if encoded in environment:
        current = environment[encoded]
        units = current.split(ENCODED_SEPARATOR) if current else []
        environment[encoded] = ENCODED_SEPARATOR.join([*units, *flag])
        return encoded
    current = environment.get(plain, "")
    environment[plain] = " ".join([current, *flag]).strip()
    return plain


def command_environment(
    command: Sequence[str], target_feature: str | None = None
) -> tuple[dict[str, str] | None, list[str]]:
    """The environment one command runs under, plus the variables it changed.

    Nothing is returned as the environment when the command inherits the caller's unchanged.

    Target features reach a compiler through its flag variable, never through the argv.

    That is how the CI legs pin them, and it is what decides `cfg(target_feature = ..)`.

    A doctest is compiled by the documentation tool, not by the ordinary one.

    So a feature has to reach both, or the library gets it while its doctests keep the
    baseline values.
    """
    # Documentation tests and the documentation build are the two commands the
    # documentation tool compiles, so both need its own flag variable.
    doc_build = command[:3] == ["cargo", "+stable", "doc"]
    doctest = command[:2] == ["cargo", "test"] and "--doc" in command

    if not doc_build and not target_feature:
        return None, []

    environment = os.environ.copy()
    touched = []

    if target_feature:
        flag = ["-C", f"target-feature={target_feature}"]
        touched.append(
            append_compiler_flag(environment, "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", flag)
        )
        if doc_build or doctest:
            touched.append(
                append_compiler_flag(
                    environment, "RUSTDOCFLAGS", "CARGO_ENCODED_RUSTDOCFLAGS", flag
                )
            )

    if doc_build:
        deny = ["-D", "rustdoc::broken_intra_doc_links"]
        name = append_compiler_flag(
            environment, "RUSTDOCFLAGS", "CARGO_ENCODED_RUSTDOCFLAGS", deny
        )
        if name not in touched:
            touched.append(name)

    return environment, touched


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(
        attach_leading_minus_values(sys.argv[1:] if argv is None else argv)
    )
    args.workspace_root = args.workspace_root.resolve()
    try:
        commands = commands_for(args)
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
    # Only the subcommands that compile or run Rust accept target features.
    target_feature = getattr(args, "target_feature", None)
    for command in commands:
        environment, touched = command_environment(command, target_feature)
        # Announce where the requested feature landed, so a dry run shows the real variable.
        if target_feature:
            for name in touched:
                print(f"+ {name} += -C target-feature={target_feature}", flush=True)
        print(f"+ {display(command)}", flush=True)
        if not args.dry_run:
            try:
                completed = subprocess.run(
                    command,
                    cwd=args.workspace_root,
                    env=environment,
                    check=False,
                )
            except OSError as error:
                # A missing tool is a prerequisite the contributor has not installed yet.
                #
                # Name it, rather than ending a long run in a stack trace.
                print(
                    f"error: {command[0]} not found ({error}); "
                    "see CONTRIBUTING.md for the required tools",
                    file=sys.stderr,
                )
                return 127
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
