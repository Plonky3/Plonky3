# Binary backend scoreboard: Apple M2

Measured on 2026-09-24 with `scripts/scoreboard.py`, at the frozen workloads that script
carries. Every row is one run of the `prove_hash_binary` example at 96-bit security, and
the security column is what that run proved rather than what it asked for.

- **Machine**: Apple M2, 4 performance and 4 efficiency cores, 24 GiB, macOS 15.4
- **Toolchain**: rustc 1.98.1, `--release`, `--features parallel`
- **Command**: `python3 scripts/scoreboard.py --binary target/release/examples/prove_hash_binary`

`Hashes/s` counts hashes rather than rows: a Keccak-f permutation occupies 25 rows and a
compression occupies one, so it is the figure published comparisons quote.

| Objective | Rows | Width | Threads | Witness | Prove | Hashes/s | Verify | Serialize | Proof | Peak RSS | Security |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| keccak-f-permutations | 2^14 | 1625 | 1 | 0.009 s | 0.299 s | 2,188 | 0.011 s | 0.001 s | 470 KB | 76 MiB | 97.1 |
| keccak-f-permutations | 2^14 | 1625 | 8 | 0.002 s | 0.082 s | 7,975 | 0.010 s | 0.000 s | 470 KB | 93 MiB | 97.1 |
| keccak-f-permutations | 2^16 | 1625 | 1 | 0.025 s | 1.154 s | 2,271 | 0.011 s | 0.000 s | 571 KB | 265 MiB | 97.1 |
| keccak-f-permutations | 2^16 | 1625 | 8 | 0.007 s | 0.316 s | 8,287 | 0.011 s | 0.000 s | 571 KB | 304 MiB | 97.1 |
| blake-3-compressions | 2^10 | 11536 | 1 | 0.000 s | 0.100 s | 10,263 | 0.046 s | 0.006 s | 583 KB | 57 MiB | 97.1 |
| blake-3-compressions | 2^10 | 11536 | 8 | 0.000 s | 0.064 s | 16,092 | 0.044 s | 0.000 s | 583 KB | 84 MiB | 97.1 |
| blake-3-compressions | 2^12 | 11536 | 1 | 0.001 s | 0.228 s | 17,933 | 0.049 s | 0.000 s | 653 KB | 124 MiB | 97.1 |
| blake-3-compressions | 2^12 | 11536 | 8 | 0.001 s | 0.131 s | 31,177 | 0.046 s | 0.001 s | 653 KB | 152 MiB | 97.1 |
| sha-256-compressions | 2^10 | 23712 | 1 | 0.076 s | 0.210 s | 4,874 | 0.099 s | 0.001 s | 836 KB | 100 MiB | 97.1 |
| sha-256-compressions | 2^10 | 23712 | 8 | 0.014 s | 0.162 s | 6,334 | 0.106 s | 0.001 s | 836 KB | 155 MiB | 97.1 |
| sha-256-compressions | 2^12 | 23712 | 1 | 0.321 s | 0.503 s | 8,147 | 0.096 s | 0.001 s | 940 KB | 231 MiB | 97.1 |
| sha-256-compressions | 2^12 | 23712 | 8 | 0.056 s | 0.262 s | 15,617 | 0.095 s | 0.001 s | 940 KB | 280 MiB | 97.1 |

## Reading this

**Threads.** The single-threaded rows are the honest per-core figure. The 8-thread rows
use this machine's four performance and four efficiency cores together, and they scale by
1.7x to 3.7x rather than 8x, because four of those cores are efficiency cores and the
smaller traces run out of work. Keccak-f at 2^16 scales best at 3.7x, BLAKE3 at 2^12
worst at 1.7x.

**Peak memory** is the whole process, measured by wrapping the run rather than from
inside the prover, so it includes the trace, the commitment and the proof at once.

**Proof size and security do not depend on the thread count**, and `--gate` enforces
that: the same workload must emit the same proof however many threads produced it.

## Reproducing this

```bash
cargo build --release -p p3-examples --features parallel --example prove_hash_binary
python3 scripts/scoreboard.py --binary target/release/examples/prove_hash_binary
```

Add `--format json` for every field the example reports, including deserialization time
and the WHIR summary, and `--quick` to measure only the smallest height of each workload.

## Gating a change against this

```bash
python3 scripts/scoreboard.py --binary <prover> --quick --gate docs/scoreboards/baseline.json
```

The committed baseline carries proof size and security only, because those are identical
on every machine at these parameters. Proving time is not portable, so a row gates on time
only where someone adds `prove_seconds_budget` to it on a machine they control.

## What is missing

- **x86-64 rows.** This is Apple silicon only. The same command on an x86-64 machine
  produces the comparable half, and the two belong side by side.
- **BLAKE2s.** The objective lands with #2328; the script already measures it once the
  build offers it, and skips it with a note until then.
