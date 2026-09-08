# Contributing to Plonky3

Plonky3 uses stable Rust for builds and Clippy, nightly rustfmt for formatting, and Python 3.9 or newer for the shared repository commands. Install the Rust targets or tools only when the check you are running needs them.

Start with a host build and the test suite for the crate you are changing:

```bash
python3 scripts/check.py fast
python3 scripts/check.py test --package p3-field
python3 scripts/check.py doctest --package p3-field
```

On Windows, invoke the same script with `python` in place of `python3`. The script runs from any current directory and executes Cargo in the workspace root.

The `test` command uses `cargo-nextest`, matching CI. Install the CI development tools when you need their checks:

```bash
cargo install cargo-nextest cargo-sort cargo-machete taplo-cli
rustup component add clippy
rustup toolchain install nightly --component rustfmt
```

Run `python3 scripts/check.py full` before opening a pull request. It covers the host check, sequential and `parallel` tests and doctests, dependency checks, Clippy, docs, and formatting. Use `--dry-run` before the command name to inspect a recipe without executing build or test commands. Commands that depend on the workspace package list still run the read-only `cargo metadata` query in dry-run mode.

## Command reference

| Command | Purpose | Options |
| --- | --- | --- |
| `fast` | Check all host targets without running them | `--package NAME`, `--parallel` |
| `full` | Run all routine host checks | none |
| `test` | Run tests through `cargo-nextest` | `--package NAME`, `--parallel` |
| `doctest` | Run Rust documentation tests | `--package NAME`, `--parallel` |
| `lint` | Run all CI script/lint/doc/format checks | `--check scripts\|sort\|toml\|deps\|clippy\|docs\|fmt` |
| `architecture` | Compile all targets for a non-runnable target | `--target TARGET`, `--parallel` |
| `embedded` | Build every eligible workspace library with `--lib` | `--target TARGET` |
| `wasm` | Run the wasm compile and SIMD smoke recipes | `--step build\|test\|smoke\|bench\|merkle` |
| `bench` | Execute every benchmark body once, excluding the large `p3-dft` sweep | none |
| `whir-exhaustive` | Run the ignored exhaustive WHIR test | `--parallel` |
| `binary-large` | Run the ignored binary PCS `2^16` round trip | none |

`architecture` is a compile-only command. CI uses `test` and `doctest` only on architectures whose runner can execute the selected instructions, and uses `architecture` for AVX-512, VPCLMULQDQ, and SVE2 coverage where runner hardware is not guaranteed.

The embedded package list comes from `cargo metadata`. Every workspace package with a library target is included unless its manifest declares:

```toml
[package.metadata.plonky3]
embedded = false
```

`p3-examples` and `p3-field-testing` are explicit host-only exclusions. Building each selected package with `--lib` prevents host-only binaries and examples from breaking an otherwise `no_std` library build. New library crates therefore enter embedded coverage automatically.

A package whose meaningful tests require a set of opt-in features can declare one additional CI invocation:

```toml
[package.metadata.plonky3.ci]
test-features = ["backend-a", "backend-b"]
```

The listed features are enabled together. A parallel CI leg also enables `parallel` when that package defines it.

See [the architecture guide](docs/architecture.md) for the crate map and backend capabilities. Keep pull requests focused, add behavioral coverage for fixes, and document security assumptions at the public API where they apply.
