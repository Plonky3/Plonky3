# Binary backend scoreboards

A scoreboard is one run of the frozen workloads in `scripts/scoreboard.py`: the same
objectives, the same heights, single-threaded and on every thread the machine has, with
the security each run proved beside its timings.

## Where the numbers come from

Published numbers are produced by CI, on runners anyone can rent, so a reader can
reproduce them without trusting the machine they were measured on:

| Runner | Architecture |
| --- | --- |
| `ubuntu-latest` | x86-64 |
| `ubuntu-24.04-arm` | 64-bit ARM |

The weekly bench job runs both and writes the tables into its job summary. Nothing in
this directory records a timing, because a timing is only meaningful together with the
machine that produced it, and this repository cannot vouch for a machine it does not run.

Apple silicon is not covered there. A maintainer with an M-series machine can run the
same command locally and paste the table into an issue or a release note, where it is
attributed to that machine rather than presented as the project's figure.

## Running it yourself

```bash
cargo build --release -p p3-examples --features parallel --example prove_hash_binary
python3 scripts/scoreboard.py --binary target/release/examples/prove_hash_binary
```

`--format json` reports every field the example carries, including deserialization time
and the WHIR summary. `--quick` measures only the smallest height of each workload.

## What is gated

`baseline.json` carries proof size and security, and nothing else.

Both are identical on every machine and at every thread count at these parameters, so
they can be committed by one contributor and checked by another without anyone having to
trust the first one's hardware. A change to either is a change in the protocol.

That is a claim worth checking rather than asserting, so the committed baseline was
compared across architectures before it was written: the same workloads built for
`aarch64-apple-darwin` and for `x86_64-apple-darwin` emit byte-identical proofs at
identical security. The weekly job then checks it again on each runner, so a wrong
baseline fails the first time CI runs rather than being believed.

```bash
python3 scripts/scoreboard.py --binary <prover> --quick --gate docs/scoreboards/baseline.json
```

The gate also fails a proof that changes with the thread count, which needs no baseline
to be wrong.

Proving time is deliberately not gated here. A row gates on time only where someone adds
`prove_seconds_budget` to it, on a machine they control and can keep quiet.
