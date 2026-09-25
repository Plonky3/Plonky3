# Binary backend scoreboards

`scripts/scoreboard.py` runs a frozen set of workloads through the `prove_hash_binary`
example: the same objectives at the same heights, single-threaded and on every thread the
machine has, with the security each run proved beside its timings.

```bash
cargo build --release -p p3-examples --features parallel --example prove_hash_binary
python3 scripts/scoreboard.py --binary target/release/examples/prove_hash_binary
```

`--format json` reports every field the example carries, including deserialization time
and the WHIR summary. `--quick` measures only the smallest height of each workload.

## Timings are not published here

Nothing in this directory records a proving time, and CI does not measure one.

A timing means something only together with the machine that produced it, and a shared CI
runner is too noisy to compare against itself a week later, let alone against a
contributor's desk. Where the project's official figures should come from, and on which
instance types, is a decision for the maintainers. Until it is made, run the command above
on a machine you can vouch for and attribute the table to it.

## What is gated

`baseline.json` holds proof size and security, and nothing else. Neither is a measurement:
the same workload at the same parameters emits the same proof on every machine and at
every thread count, which was checked across architectures before the file was written,
`aarch64-apple-darwin` against `x86_64-apple-darwin`, byte for byte.

```bash
python3 scripts/scoreboard.py --binary <prover> --quick --gate docs/scoreboards/baseline.json
```

That is what the weekly bench job runs. It is a determinism check rather than a benchmark,
so runner noise cannot move it: a failure means the proof or the security changed, which is
a change in the protocol.

The gate also fails a proof that changes with the thread count, which needs no baseline to
be wrong. A row gates on proving time only where someone adds `prove_seconds_budget` to it,
on a machine they control, and nothing committed here carries one.
