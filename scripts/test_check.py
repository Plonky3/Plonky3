#!/usr/bin/env python3

import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
import unittest.mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

SCRIPT = Path(__file__).with_name("check.py")
REPO = SCRIPT.parent.parent


class CheckCliTests(unittest.TestCase):
    def run_check(self, *args, cwd=REPO, env=None):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--workspace-root", str(cwd), *args],
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def dry_run_lines(self, *args, cwd=REPO):
        result = self.run_check("--dry-run", *args, cwd=cwd)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.splitlines()

    def test_test_and_doctest_match_ci_recipes(self):
        self.assertEqual(
            self.dry_run_lines("test", "--parallel")[0],
            "+ cargo nextest run --features parallel",
        )
        self.assertEqual(
            self.dry_run_lines("doctest", "--package", "p3-util"),
            ["+ cargo test --doc -p p3-util"],
        )

    def test_architecture_build_is_compile_only(self):
        self.assertEqual(
            self.dry_run_lines(
                "architecture", "--target", "x86_64-unknown-linux-gnu", "--parallel"
            ),
            [
                "+ cargo build --target x86_64-unknown-linux-gnu --all-targets --features parallel"
            ],
        )

    def test_wasm_steps_preserve_build_and_runtime_coverage(self):
        self.assertEqual(
            self.dry_run_lines("wasm"),
            [
                "+ cargo build --verbose --target wasm32-unknown-unknown -p p3-goldilocks",
                "+ cargo test --release --target wasm32-wasip1 -p p3-goldilocks --lib wasm32_simd128",
                "+ cargo run --release --target wasm32-wasip1 --bin wasm_smoke -p p3-goldilocks",
                "+ cargo run --release --target wasm32-wasip1 --bin wasm_bench -p p3-goldilocks",
                "+ cargo run --release --target wasm32-wasip1 --example wasm_merkle_bench -p p3-goldilocks",
            ],
        )

    def test_heavy_and_benchmark_recipes_are_exact(self):
        self.assertEqual(
            self.dry_run_lines("bench"),
            [
                "+ cargo bench --workspace --exclude p3-dft --features parallel -- --test"
            ],
        )
        self.assertEqual(
            self.dry_run_lines("whir-exhaustive", "--parallel"),
            [
                "+ cargo test -p p3-whir --features parallel pcs::tests::test_whir_end_to_end_exhaustive -- --ignored --exact"
            ],
        )
        self.assertEqual(
            self.dry_run_lines("binary-large"),
            [
                "+ cargo test -p p3-binary-pcs --release --test end_to_end large_configuration_2_16_round_trips -- --ignored --exact"
            ],
        )

    def test_lint_can_run_each_existing_ci_check_independently(self):
        expected = {
            "sort": "+ cargo +stable sort --workspace --grouped --check",
            "toml": "+ taplo fmt --check",
            "deps": "+ cargo machete --with-metadata",
            "clippy": "+ cargo +stable clippy --all-targets -- -D warnings",
            "docs": "+ cargo +stable doc --no-deps --workspace --document-private-items",
            "fmt": "+ cargo +nightly fmt --all -- --check",
            "scripts": "+ "
            + shlex.join([sys.executable, "-m", "unittest", "scripts/test_check.py", "-v"]),
        }
        for check, command in expected.items():
            with self.subTest(check=check):
                self.assertEqual(self.dry_run_lines("lint", "--check", check), [command])

    def test_unknown_command_has_usage_error(self):
        result = self.run_check("unknown")
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_subprocess_exit_status_is_propagated(self):
        with tempfile.TemporaryDirectory() as temp:
            bin_dir = Path(temp)
            fake = bin_dir / "cargo"
            fake.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            (bin_dir / "cargo.bat").write_text("@exit /b 23\r\n", encoding="utf-8")
            env = os.environ.copy()
            env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
            result = self.run_check("fast", env=env)
        self.assertEqual(result.returncode, 23)

    def test_docs_preserve_rustdocflags_and_deny_broken_links(self):
        with tempfile.TemporaryDirectory() as temp:
            bin_dir = Path(temp)
            capture = bin_dir / "rustdocflags.txt"
            fake = bin_dir / "cargo"
            fake.write_text(
                '#!/bin/sh\nprintf "%s" "$RUSTDOCFLAGS" > "$CHECK_CAPTURE"\n',
                encoding="utf-8",
            )
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            (bin_dir / "cargo.bat").write_text(
                '@echo|set /p="%RUSTDOCFLAGS%">"%CHECK_CAPTURE%"\r\n', encoding="utf-8"
            )
            env = os.environ.copy()
            env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
            env["CHECK_CAPTURE"] = str(capture)
            env["RUSTDOCFLAGS"] = "-D warnings"
            result = self.run_check("lint", "--check", "docs", env=env)
            captured_flags = capture.read_text(encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            captured_flags, "-D warnings -D rustdoc::broken_intra_doc_links"
        )


class CargoMetadataTests(unittest.TestCase):
    def make_workspace(self, root):
        root = Path(root)
        (root / "Cargo.toml").write_text(
            textwrap.dedent(
                """
                [workspace]
                resolver = "2"
                members = ["alpha", "beta", "runner", "featured"]
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        self.make_package(root, "alpha", "p3-alpha", "")
        self.make_package(
            root,
            "beta",
            "p3-beta",
            "[package.metadata.plonky3]\nembedded = false\n",
        )
        self.make_package(root, "runner", "p3-runner", "", target="bin")
        self.make_package(
            root,
            "featured",
            "p3-featured",
            """
            [features]
            backend-a = []
            backend-b = []
            parallel = []

            [package.metadata.plonky3.ci]
            test-features = ["backend-a", "backend-b"]
            """,
        )

    def make_package(self, root, directory, name, extra, target="lib"):
        package = root / directory
        (package / "src").mkdir(parents=True)
        (package / "Cargo.toml").write_text(
            textwrap.dedent(
                f"""
                [package]
                name = "{name}"
                version = "0.1.0"
                edition = "2021"

                {extra}
                """
            ),
            encoding="utf-8",
        )
        source = package / "src" / ("lib.rs" if target == "lib" else "main.rs")
        source.write_text("pub fn marker() {}\n" if target == "lib" else "fn main() {}\n")

    def run_fixture(self, root, *args):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--workspace-root",
                str(root),
                "--dry-run",
                *args,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_embedded_uses_real_workspace_metadata_and_library_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            self.make_workspace(temp)
            result = self.run_fixture(temp, "embedded", "--target", "thumb-test")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "+ cargo build --verbose --target thumb-test -p p3-alpha --lib",
                "+ cargo build --verbose --target thumb-test -p p3-featured --lib",
            ],
        )

    def test_workspace_member_added_to_metadata_is_automatically_embedded(self):
        with tempfile.TemporaryDirectory() as temp:
            self.make_workspace(temp)
            root = Path(temp)
            manifest = root / "Cargo.toml"
            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    '"featured"]', '"featured", "new-lib"]'
                ),
                encoding="utf-8",
            )
            self.make_package(root, "new-lib", "p3-new-lib", "")
            result = self.run_fixture(root, "embedded")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-p p3-new-lib --lib", result.stdout)

    def test_test_features_from_metadata_are_enabled_together(self):
        with tempfile.TemporaryDirectory() as temp:
            self.make_workspace(temp)
            result = self.run_fixture(temp, "test", "--parallel")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "+ cargo nextest run --features parallel",
                "+ cargo nextest run -p p3-featured --features backend-a,backend-b,parallel",
            ],
        )

    def test_targeted_feature_package_runs_after_empty_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            self.make_workspace(temp)
            root = Path(temp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            capture = root / "commands.txt"
            real_cargo = shutil.which("cargo")
            self.assertIsNotNone(real_cargo)
            fake = bin_dir / "cargo"
            fake.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    if [ "$1" = metadata ]; then exec "$REAL_CARGO" "$@"; fi
                    printf '%s\\n' "$*" >> "$CHECK_CAPTURE"
                    case " $* " in
                      *" --features "*) exit 0 ;;
                      *" --no-tests warn "*) exit 0 ;;
                      *) exit 4 ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            (bin_dir / "cargo.bat").write_text(
                textwrap.dedent(
                    """\
                    @echo off
                    if "%1"=="metadata" (
                      "%REAL_CARGO%" %*
                      exit /b %errorlevel%
                    )
                    echo %*>>"%CHECK_CAPTURE%"
                    echo %* | findstr /c:"--features" >nul && exit /b 0
                    echo %* | findstr /c:"--no-tests warn" >nul && exit /b 0
                    exit /b 4
                    """
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
            env["REAL_CARGO"] = real_cargo
            env["CHECK_CAPTURE"] = str(capture)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--workspace-root",
                    str(root),
                    "test",
                    "--package",
                    "p3-featured",
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            commands = capture.read_text(encoding="utf-8").splitlines()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            commands,
            [
                "nextest run -p p3-featured --no-tests warn",
                "nextest run -p p3-featured --features backend-a,backend-b",
            ],
        )

    def test_repository_metadata_explicitly_excludes_host_only_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--workspace-root",
                str(REPO),
                "--dry-run",
                "embedded",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # Pin the exclusions as a set, not a sample.
        #
        # A manifest opting out would otherwise shrink embedded coverage silently.
        #
        # Nothing in the test diff would say that a crate stopped being checked.
        selected = {
            shlex.split(line)[shlex.split(line).index("-p") + 1]
            for line in result.stdout.splitlines()
        }
        members = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        )
        library_members = {
            package["name"]
            for package in json.loads(members.stdout)["packages"]
            if any("lib" in target["kind"] for target in package["targets"])
        }
        self.assertEqual(library_members - selected, {"p3-examples", "p3-field-testing"})
        self.assertIn("p3-binary-pcs", selected)
        for line in result.stdout.splitlines():
            self.assertTrue(line.endswith(" --lib"), line)


class TargetFeatureTests(unittest.TestCase):
    def test_target_feature_reaches_the_flags_and_not_the_argv(self):
        # A leg pins its features through the compiler's flag variable, since that is what
        # decides cfg(target_feature = ..).
        #
        # The argv must therefore stay identical to the unpinned command.
        for command, expected in [
            (["architecture", "--target", "x86_64-unknown-linux-gnu"], "cargo build"),
            (["test"], "cargo nextest run"),
            (["doctest"], "cargo test --doc"),
        ]:
            with self.subTest(command=command[0]):
                plain = _run_dry(command)
                pinned = _run_dry([*command, "--target-feature=+avx2,+vpclmulqdq"])
                self.assertTrue(
                    any(line.startswith(f"+ {expected}") for line in plain), plain
                )
                self.assertEqual(
                    [line for line in pinned if "target-feature" not in line], plain
                )

    def test_a_leading_minus_value_is_accepted_in_both_spellings(self):
        # Turning a feature off spells the value with a leading minus.
        #
        # Both the joined and the separated spelling must reach the compiler.
        joined = _run_dry(["test", "--package", "p3-keccak", "--target-feature=-sha3"])
        separated = _run_dry(["test", "--package", "p3-keccak", "--target-feature", "-sha3"])
        self.assertEqual(joined, separated)
        self.assertIn("+ RUSTFLAGS += -C target-feature=-sha3", joined)

    def test_a_requested_feature_outranks_an_inherited_disable(self):
        # The last setting of a feature is the one that takes effect.
        #
        #     inherited : +avx2,-avx2   -> avx2 off
        #     appended  : +avx2         -> avx2 on
        #
        # So the flag is appended rather than matched against what is already there.
        import check  # noqa: PLC0415

        with unittest.mock.patch.dict(
            os.environ, {"RUSTFLAGS": "-C target-feature=+avx2,-avx2"}, clear=True
        ):
            environment, _ = check.command_environment(["cargo", "build"], "+avx2")
        self.assertTrue(environment["RUSTFLAGS"].endswith("-C target-feature=+avx2"))

    def test_an_inherited_plain_variable_is_preserved(self):
        # CI already exports a debug-info setting, so the append must not replace it.
        import check  # noqa: PLC0415

        with unittest.mock.patch.dict(
            os.environ, {"RUSTFLAGS": "-C debuginfo=0"}, clear=True
        ):
            environment, touched = check.command_environment(["cargo", "build"], "+avx2")
        self.assertEqual(
            environment["RUSTFLAGS"], "-C debuginfo=0 -C target-feature=+avx2"
        )
        self.assertEqual(touched, ["RUSTFLAGS"])

    def test_the_encoded_variable_is_used_when_the_caller_set_it(self):
        # Cargo ignores the plain variable whenever the encoded one is set.
        #
        # Writing the plain one there would announce a feature that never reaches rustc.
        import check  # noqa: PLC0415

        encoded = f"-C{check.ENCODED_SEPARATOR}debuginfo=0"
        with unittest.mock.patch.dict(
            os.environ, {"CARGO_ENCODED_RUSTFLAGS": encoded}, clear=True
        ):
            environment, touched = check.command_environment(["cargo", "build"], "+sve2")
        self.assertEqual(
            environment["CARGO_ENCODED_RUSTFLAGS"].split(check.ENCODED_SEPARATOR),
            ["-C", "debuginfo=0", "-C", "target-feature=+sve2"],
        )
        self.assertNotIn("RUSTFLAGS", environment)
        self.assertEqual(touched, ["CARGO_ENCODED_RUSTFLAGS"])

    def test_a_doctest_pins_the_documentation_compiler_too(self):
        # Doctests are compiled by the documentation tool, not the ordinary one.
        #
        # Without its own flag variable the library gets the feature and its doctests do not.
        import check  # noqa: PLC0415

        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            environment, touched = check.command_environment(
                ["cargo", "test", "--doc"], "+sve2"
            )
        self.assertEqual(environment["RUSTFLAGS"], "-C target-feature=+sve2")
        self.assertEqual(environment["RUSTDOCFLAGS"], "-C target-feature=+sve2")
        self.assertEqual(touched, ["RUSTFLAGS", "RUSTDOCFLAGS"])

    def test_a_doctest_uses_the_encoded_documentation_variable_when_set(self):
        import check  # noqa: PLC0415

        encoded = f"-C{check.ENCODED_SEPARATOR}debuginfo=0"
        with unittest.mock.patch.dict(
            os.environ, {"CARGO_ENCODED_RUSTDOCFLAGS": encoded}, clear=True
        ):
            environment, _ = check.command_environment(
                ["cargo", "test", "--doc"], "+sve2"
            )
        self.assertEqual(
            environment["CARGO_ENCODED_RUSTDOCFLAGS"].split(check.ENCODED_SEPARATOR),
            ["-C", "debuginfo=0", "-C", "target-feature=+sve2"],
        )
        self.assertNotIn("RUSTDOCFLAGS", environment)

    def test_no_target_feature_inherits_the_environment_unchanged(self):
        import check  # noqa: PLC0415

        self.assertEqual(check.command_environment(["cargo", "build"], None), (None, []))

    def test_docs_deny_broken_links_alongside_a_target_feature(self):
        import check  # noqa: PLC0415

        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            environment, _ = check.command_environment(
                ["cargo", "+stable", "doc"], "+avx2"
            )
        self.assertEqual(
            environment["RUSTDOCFLAGS"],
            "-C target-feature=+avx2 -D rustdoc::broken_intra_doc_links",
        )

    def test_docs_deny_broken_links_with_no_target_feature(self):
        import check  # noqa: PLC0415

        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            environment, touched = check.command_environment(
                ["cargo", "+stable", "doc"], None
            )
        self.assertEqual(
            environment["RUSTDOCFLAGS"], "-D rustdoc::broken_intra_doc_links"
        )
        self.assertEqual(touched, ["RUSTDOCFLAGS"])


def _run_dry(command):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--workspace-root", str(REPO), "--dry-run", *command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


if __name__ == "__main__":
    unittest.main()
