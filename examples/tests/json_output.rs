//! Process-level check that the binary prover example emits clean JSON.
//!
//! Serializing the report in-process cannot see what else the executable prints.
//!
//! This test runs the real executable and parses its entire standard output.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs, mem};

use serde_json::Value;

/// The example whose standard output this test parses.
const EXAMPLE: &str = "prove_hash_binary";

/// Cargo profile the test binary itself was built with.
///
/// Rerunning the example under the same profile reuses its build artifacts.
fn current_profile() -> String {
    // The test binary lives at `<target>/<profile dir>/deps/<name>`.
    let exe = env::current_exe().expect("the test binary has a path");
    let profile_dir = exe
        .parent()
        .and_then(Path::parent)
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .expect("the test binary sits under a profile directory");

    // The dev profile is the only one whose directory name differs from its own.
    match profile_dir {
        "debug" => "dev".to_string(),
        other => other.to_string(),
    }
}

/// The example executable built alongside this test, while no source has changed since.
///
/// A default `cargo test` or `cargo nextest run` builds every example of the package with the
/// test targets, under the same profile and features:
///
///     <target>/<profile dir>/deps/json_output-<hash>
///     <target>/<profile dir>/examples/prove_hash_binary
///
/// A `cargo run` from inside this test resolves features for this package alone. A run over
/// several packages unifies them across all of those, so the two builds share few artifacts and
/// the second one rebuilds the example's whole dependency graph.
///
/// Beside the executable, Cargo writes a dep-info rule naming every local source it was compiled
/// from. A build that selects only this test does not rebuild the example, so the executable is
/// used only while none of those sources is newer than it.
fn prebuilt_example() -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    let examples = exe.parent()?.parent()?.join("examples");
    let executable = examples.join(format!("{EXAMPLE}{}", env::consts::EXE_SUFFIX));
    let built = fs::metadata(&executable)
        .and_then(|metadata| metadata.modified())
        .ok()?;

    // The rule reads `<executable>: <source> <source> ...`.
    let dep_info = fs::read_to_string(examples.join(format!("{EXAMPLE}.d"))).ok()?;
    let (_, sources) = dep_info.split_once(": ")?;
    let current = dep_info_paths(sources).iter().all(|source| {
        fs::metadata(source)
            .and_then(|metadata| metadata.modified())
            .is_ok_and(|modified| modified <= built)
    });
    current.then_some(executable)
}

/// Splits the sources of a dep-info rule, in which Cargo escapes a space inside a path as `\ `.
fn dep_info_paths(sources: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let mut path = String::new();
    let mut chars = sources.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' && chars.peek() == Some(&' ') {
            path.push(' ');
            chars.next();
        } else if c.is_whitespace() {
            if !path.is_empty() {
                paths.push(mem::take(&mut path));
            }
        } else {
            path.push(c);
        }
    }
    if !path.is_empty() {
        paths.push(path);
    }
    paths
}

#[test]
fn json_format_prints_only_one_json_object_on_stdout() {
    // Run the example built with this test, or build it under the same profile and features.
    let mut command = prebuilt_example().map_or_else(
        || {
            let mut cargo = Command::new(env!("CARGO"));
            cargo
                .args(["run", "--quiet", "--profile", &current_profile()])
                .args(["--example", EXAMPLE]);
            if cfg!(feature = "parallel") {
                cargo.args(["--features", "parallel"]);
            }
            cargo.arg("--");
            cargo
        },
        Command::new,
    );
    command.current_dir(env!("CARGO_MANIFEST_DIR"));

    // Four Blake-3 compressions keep the proof fast in an unoptimized build.
    command.args([
        "--objective",
        "blake-3-compressions",
        "--log-trace-length",
        "2",
        "--format",
        "json",
    ]);

    let output = command.output().expect("the example starts");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "the example failed:\n{stderr}");

    // The whole of standard output must parse as a single JSON value.
    //
    // Any progress line or tracing span on stdout makes this fail:
    //
    //     Proving 4 Blake-3 compressions      <- not JSON
    //     INFO prove [ 12ms | ... ]           <- not JSON
    //     {"rows":4,...}
    let stdout = String::from_utf8(output.stdout).expect("stdout is UTF-8");
    let report: Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|error| panic!("stdout is not one JSON value ({error}):\n{stdout}"));

    // That value is the report of the run that was asked for.
    //
    //     2^2 rows, measured by the caller, proved and verified
    assert!(report.is_object(), "stdout is not a JSON object:\n{stdout}");
    assert_eq!(report["rows"], 4);
    assert!(report["witness_seconds"].is_f64());
    assert!(report["verify_seconds"].is_f64());
}
