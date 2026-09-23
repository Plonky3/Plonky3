//! Process-level check that the binary prover example emits clean JSON.
//!
//! Serializing the report in-process cannot see what else the executable prints.
//!
//! This test runs the real executable and parses its entire standard output.

use std::env;
use std::path::Path;
use std::process::Command;

use serde_json::Value;

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

#[test]
fn json_format_prints_only_one_json_object_on_stdout() {
    // Build and run the example with the same profile and features as this test.
    let mut command = Command::new(env!("CARGO"));
    command
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["run", "--quiet", "--profile", &current_profile()])
        .args(["--example", "prove_hash_binary"]);
    if cfg!(feature = "parallel") {
        command.args(["--features", "parallel"]);
    }

    // Four Blake-3 compressions keep the proof fast in an unoptimized build.
    command.args([
        "--",
        "--objective",
        "blake-3-compressions",
        "--log-trace-length",
        "2",
        "--format",
        "json",
    ]);

    let output = command.output().expect("cargo runs");
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
