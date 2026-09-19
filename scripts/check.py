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

DOC_ONLY_NAMES = {"CHANGELOG.md", "LICENSE-APACHE", "LICENSE-MIT"}
DOC_ONLY_PATHS = {"CONTRIBUTING.md", "README.md", "RELEASING.md"}
DOC_ONLY_PREFIXES = ("audits/", "docs/", ".github/ISSUE_TEMPLATE/")
FULL_CI_PATHS = {"Cargo.lock", "Cargo.toml", "rust-toolchain.toml", "rustfmt.toml"}
FULL_CI_PREFIXES = (".cargo/", ".github/workflows/", "scripts/")


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


def changed_paths(workspace_root: Path, base: str, head: str) -> list[str]:
    """Return both sides of each changed or renamed path."""
    # Three-dot comparison matches the pull request's merge-base semantics.
    command = ["git", "diff", "--name-status", "-z", "--find-renames", f"{base}...{head}"]
    completed = subprocess.run(
        command,
        cwd=workspace_root,
        stdout=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)

    # Rename and copy records carry an old and a new path.
    fields = completed.stdout.decode("utf-8").split("\0")
    paths = []
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index]
        path_count = 2 if status[0] in {"R", "C"} else 1
        paths.extend(fields[index + 1 : index + 1 + path_count])
        index += 1 + path_count
    return sorted(set(paths))


def ci_plan(
    metadata: dict[str, Any], paths: Sequence[str], force_full: bool = False
) -> dict[str, Any]:
    """Select workspace packages whose tests can observe the changed files."""
    packages = workspace_packages(metadata)
    all_names = {package["name"] for package in packages}
    workspace_root = Path(metadata["workspace_root"]).resolve()

    # Package roots are the stable bridge from Git paths to Cargo's graph.
    roots = {
        Path(package["manifest_path"]).resolve().parent: package["name"]
        for package in packages
    }
    ordered_roots = sorted(roots, key=lambda root: len(root.parts), reverse=True)

    # Global build inputs may affect every package.
    full = force_full
    changed_packages = set()
    for raw_path in paths:
        path = Path(raw_path)
        if raw_path in FULL_CI_PATHS or raw_path.startswith(FULL_CI_PREFIXES):
            full = True
            continue

        absolute = (workspace_root / path).resolve()
        owner = next(
            (
                roots[root]
                for root in ordered_roots
                if absolute == root or root in absolute.parents
            ),
            None,
        )
        doc_only = (
            path.name in DOC_ONLY_NAMES
            or raw_path in DOC_ONLY_PATHS
            or raw_path.startswith(DOC_ONLY_PREFIXES)
        )
        if doc_only:
            continue
        if owner is None:
            # Unknown shared inputs fail closed to full CI.
            full = True
        else:
            changed_packages.add(owner)

    if full:
        affected = all_names
    else:
        # Production and build dependencies propagate through dependent libraries.
        reverse_production = {name: set() for name in all_names}
        reverse_dev = {name: set() for name in all_names}
        root_names = {root: name for root, name in roots.items()}
        for package in packages:
            for dependency in package["dependencies"]:
                dependency_path = dependency.get("path")
                if dependency_path is None:
                    continue
                dependency_name = root_names.get(Path(dependency_path).resolve())
                if dependency_name is None:
                    continue
                reverse = reverse_dev if dependency.get("kind") == "dev" else reverse_production
                reverse[dependency_name].add(package["name"])

        # Walk only production edges to avoid dev-dependency cycles.
        production_affected = set(changed_packages)
        pending = list(changed_packages)
        while pending:
            dependency = pending.pop()
            for dependent in reverse_production[dependency]:
                if dependent not in production_affected:
                    production_affected.add(dependent)
                    pending.append(dependent)

        # Direct dev consumers own integration tests for the affected libraries.
        affected = set(production_affected)
        for dependency in production_affected:
            affected.update(reverse_dev[dependency])

    selected = sorted(affected)
    selected_metadata = [package for package in packages if package["name"] in affected]
    embedded = set(embedded_packages(metadata, set(selected)))
    any_toml = any(Path(path).suffix == ".toml" for path in paths)
    any_manifest = any(Path(path).name == "Cargo.toml" for path in paths)
    scripts = force_full or any(path.startswith("scripts/") for path in paths)

    # Scalar outputs are easy to consume from every GitHub Actions shell.
    return {
        "packages": selected,
        "full": full,
        "rust": bool(selected),
        "parallel": any(
            "parallel" in package.get("features", {}) for package in selected_metadata
        ),
        "embedded": bool(embedded),
        "wasm": "p3-goldilocks" in affected,
        "keccak": "p3-keccak" in affected,
        "sha_ni": "p3-sha256" in affected,
        "gfni": "p3-binary-field" in affected,
        "toml": full or any_toml,
        "manifests": full or any_manifest,
        "scripts": full or scripts,
        "lint": bool(selected) or full or any_toml or scripts,
    }


def emit_ci_plan(plan: dict[str, Any], github_output: Path | None) -> None:
    """Print a readable plan and optional GitHub Actions outputs."""
    # The JSON summary makes local planning reproducible.
    print(json.dumps(plan, indent=2, sort_keys=True))
    if github_output is None:
        return

    # Every output is a single line and contains no shell syntax.
    outputs = []
    for name, value in plan.items():
        if isinstance(value, list):
            value = ",".join(value)
        elif isinstance(value, bool):
            value = str(value).lower()
        outputs.append(f"{name}={value}")
    with github_output.open("a", encoding="utf-8") as output:
        output.write("\n".join(outputs) + "\n")


def plonky3_metadata(package: dict[str, Any]) -> dict[str, Any]:
    metadata = package.get("metadata") or {}
    value = metadata.get("plonky3", {})
    return value if isinstance(value, dict) else {}


def embedded_packages(
    metadata: dict[str, Any], selected: set[str] | None = None
) -> list[str]:
    # Host-only crates opt out in their package metadata.
    return [
        package["name"]
        for package in workspace_packages(metadata)
        if selected is None or package["name"] in selected
        if any("lib" in target["kind"] for target in package["targets"])
        and plonky3_metadata(package).get("embedded", True) is not False
    ]


def package_test_features(
    metadata: dict[str, Any], selected: set[str] | None, parallel: bool
) -> list[tuple[str, str]]:
    # Feature-only suites remain separate from each package's baseline tests.
    runs = []
    for package in workspace_packages(metadata):
        if selected is not None and package["name"] not in selected:
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


def selected_packages(args: argparse.Namespace) -> list[str] | None:
    # The comma-separated form keeps a generated CI value portable across shells.
    packages = list(args.package or [])
    if args.packages:
        packages.extend(name for name in args.packages.split(",") if name)
    return sorted(set(packages)) or None


def package_args(packages: Sequence[str] | None) -> list[str]:
    # Cargo accepts one package selector per workspace member.
    return [item for package in packages or [] for item in ("-p", package)]


def feature_args(
    parallel: bool,
    metadata: dict[str, Any] | None = None,
    packages: Sequence[str] | None = None,
) -> list[str]:
    if not parallel:
        return []
    if packages is None:
        return ["--features", "parallel"]

    # Qualified features enable only supported configurations of selected packages.
    selected = set(packages)
    features = [
        f"{package['name']}/parallel"
        for package in workspace_packages(metadata)
        if package["name"] in selected and "parallel" in package.get("features", {})
    ]
    return ["--features", ",".join(features)] if features else []


def test_commands(
    metadata: dict[str, Any], packages: Sequence[str] | None, parallel: bool, doctest: bool
) -> list[list[str]]:
    # Doctests use Cargo because nextest only runs binary test targets.
    if doctest:
        base = ["cargo", "test", "--doc"]
    else:
        base = ["cargo", "nextest", "run"]
    selected = set(packages) if packages is not None else None
    feature_runs = package_test_features(metadata, selected, parallel)
    baseline = base + package_args(packages) + feature_args(parallel, metadata, packages)
    if packages and feature_runs and not doctest:
        baseline += ["--no-tests", "warn"]
    commands = [baseline]
    for name, features in feature_runs:
        commands.append(base + ["-p", name, "--features", features])
    return commands


def lint_commands(
    check: str | None, packages: Sequence[str] | None = None
) -> list[list[str]]:
    # Repository-wide checks ignore package selection.
    commands = dict(LINT_COMMANDS)
    scope = package_args(packages)
    if packages:
        commands["clippy"] = [
            "cargo",
            "+stable",
            "clippy",
            *scope,
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ]
        commands["docs"] = [
            "cargo",
            "+stable",
            "doc",
            "--no-deps",
            *scope,
            "--document-private-items",
        ]
        commands["fmt"] = ["cargo", "+nightly", "fmt", *scope, "--", "--check"]
    if check:
        return [commands[check]]
    return list(commands.values())


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
    packages = selected_packages(args) if hasattr(args, "package") else None
    metadata = None
    if command in {"full", "fast", "test", "doctest", "architecture", "embedded"}:
        metadata = cargo_metadata(args.workspace_root)

    if command == "fast":
        scope = package_args(packages) if packages else ["--workspace"]
        return [
            [
                "cargo",
                "check",
                *scope,
                "--all-targets",
                *feature_args(args.parallel, metadata, packages),
            ]
        ]
    if command == "test":
        return test_commands(metadata, packages, args.parallel, False)
    if command == "doctest":
        return test_commands(metadata, packages, args.parallel, True)
    if command == "lint":
        return lint_commands(args.check, packages)
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
        target = [
            "--target",
            args.target,
            *package_args(packages),
            "--all-targets",
            *feature_args(args.parallel, metadata, packages),
        ]
        # The baseline lint job cannot reach code behind a target-feature gate,
        # so each leg lints the configuration only it compiles.
        return [
            ["cargo", "build", *target],
            ["cargo", "clippy", *target, "--", "-D", "warnings"],
        ]
    if command == "embedded":
        selected = set(packages) if packages is not None else None
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
            for package in embedded_packages(metadata, selected)
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
    if command == "slow-regressions":
        return [
            [
                "cargo",
                "test",
                "-p",
                "p3-multilinear-util",
                "poly::test::test_compress_suffix",
                "--",
                "--ignored",
                "--exact",
            ],
            [
                "cargo",
                "test",
                "-p",
                "p3-multi-stark",
                "zerocheck::tests::staged_zerocheck_mixed_poseidon2_blake3_fib",
                "--",
                "--ignored",
                "--exact",
            ],
            [
                "cargo",
                "test",
                "-p",
                "p3-whir",
                "pcs::zk::tests::zk_whir_end_to_end_multi_round",
                "--",
                "--ignored",
                "--exact",
            ],
            [
                "cargo",
                "test",
                "-p",
                "p3-whir",
                "pcs::zk::tests::zk_whir_code_switch_overhead_accounting",
                "--",
                "--ignored",
                "--exact",
            ],
            [
                "cargo",
                "test",
                "-p",
                "p3-uni-stark",
                "--test",
                "stir_fibonacci",
                "test_public_value",
                "--",
                "--ignored",
                "--exact",
            ],
            [
                "cargo",
                "test",
                "-p",
                "p3-uni-stark",
                "--test",
                "stir_fibonacci",
                "test_short_public_values_rejected",
                "--",
                "--ignored",
                "--exact",
            ],
            [
                "cargo",
                "test",
                "-p",
                "p3-binary-dft",
                "lch::tests::the_schedule_matches_an_independent_walk_at_every_blocked_shape",
                "--",
                "--ignored",
                "--exact",
            ],
            # The same sweep with threads: the worker clamp reshapes every schedule, and the
            # tile and staging tasks run side by side rather than one after another.
            [
                "cargo",
                "test",
                "-p",
                "p3-binary-dft",
                "--features",
                "parallel",
                "lch::tests::the_schedule_matches_an_independent_walk_at_every_blocked_shape",
                "--",
                "--ignored",
                "--exact",
            ],
            [
                "cargo",
                "test",
                "-p",
                "p3-binary-dft",
                "--test",
                "commit",
                "polynomial_commit_matches_naive_for_both_orders",
                "--",
                "--ignored",
                "--exact",
            ],
            [
                "cargo",
                "test",
                "-p",
                "p3-stir",
                "--test",
                "stir",
                "babybear_pcs::assert_stir_proof_smaller_than_binary_fri",
                "--",
                "--ignored",
                "--exact",
            ],
            [
                "cargo",
                "test",
                "-p",
                "p3-examples",
                "tests::test_end_to_end_koalabear_keccak_hashes_parallel_dft_keccak_merkle_tree_stir",
                "--",
                "--ignored",
                "--exact",
            ],
            [
                "cargo",
                "test",
                "-p",
                "p3-examples",
                "tests::test_end_to_end_mersenne_31_keccak_hashes_keccak_merkle_tree",
                "--",
                "--ignored",
                "--exact",
            ],
            [
                "cargo",
                "test",
                "-p",
                "p3-examples",
                "tests::test_end_to_end_mersenne31_blake3_hashes_keccak_merkle_tree",
                "--",
                "--ignored",
                "--exact",
            ],
        ]
    raise AssertionError(f"unhandled command: {command}")


def add_package_and_parallel(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--package",
        action="append",
        metavar="NAME",
        help="limit the command to a package; repeat for more packages",
    )
    parser.add_argument(
        "--packages",
        help="limit the command to a comma-separated package list",
    )
    parser.add_argument("--parallel", action="store_true", help="enable the parallel feature")


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

    plan = subparsers.add_parser("plan", help="select packages affected by a Git change")
    plan.add_argument("--base", help="base Git revision")
    plan.add_argument("--head", default="HEAD", help="head Git revision")
    plan.add_argument("--full", action="store_true", help="select the full workspace")
    plan.add_argument("--github-output", type=Path, help=argparse.SUPPRESS)
    fast = subparsers.add_parser("fast", help="check host targets without running tests")
    add_package_and_parallel(fast)
    subparsers.add_parser("full", help="run all host checks used by CI")
    test = subparsers.add_parser("test", help="run tests with cargo-nextest")
    add_package_and_parallel(test)
    add_target_feature(test)
    doctest = subparsers.add_parser("doctest", help="run Rust documentation tests")
    add_package_and_parallel(doctest)
    add_target_feature(doctest)
    lint = subparsers.add_parser("lint", help="run formatting, lint, dependency and doc checks")
    lint.add_argument("--check", choices=LINT_COMMANDS, help="run one lint check")
    lint.add_argument("--package", action="append", metavar="NAME")
    lint.add_argument("--packages")
    architecture = subparsers.add_parser(
        "architecture", help="compile all targets for a non-runnable architecture"
    )
    architecture.add_argument("--target", required=True)
    add_package_and_parallel(architecture)
    add_target_feature(architecture)
    embedded = subparsers.add_parser(
        "embedded", help="build metadata-selected libraries for an embedded target"
    )
    embedded.add_argument("--target", default="thumbv7em-none-eabi")
    embedded.add_argument("--package", action="append", metavar="NAME")
    embedded.add_argument("--packages")
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
    subparsers.add_parser(
        "slow-regressions", help="run the ignored compute-heavy regression tests"
    )
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

    if args.command == "plan":
        if not args.full and not args.base:
            print("error: --base is required unless --full is set", file=sys.stderr)
            return 2
        try:
            # Full runs do not need a comparison revision.
            paths = [] if args.full else changed_paths(args.workspace_root, args.base, args.head)
            metadata = cargo_metadata(args.workspace_root)
            emit_ci_plan(ci_plan(metadata, paths, args.full), args.github_output)
        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
        return 0

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
