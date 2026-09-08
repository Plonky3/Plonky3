# Adjacent-symbol Merkle grouping

Investigation on `robin/binary-pcs-grouped-leaves`, based on `66b07291`
([ARM tracing/arithmetic PR #2094](https://github.com/Plonky3/Plonky3/pull/2094)).
The prototype and measurements are separate from that PR.

## Result

Four symbols per leaf are the best measured balance for this workload:
**145.0 ms to 83.1 ms proving time (-42.7%)**, with **2.1% larger median
proofs**. Eight symbols halve proving time at a 16.2% proof-size cost.
Sixteen saves only another 3.6 ms while increasing proof size substantially.

Apple M4 Pro, 14 cores, macOS ARM64, Rust 1.98.0, 2026-09-08. Same nonlinear
recurrence as `prove_binary_field`: 2^18 rows, two GF(2^128) columns,
2^19 message elements, 2^21 encoded symbols, rate 1/4, 100-bit security,
zero PoW, Keccak-256, binary Merkle trees, 148 distinct base fold-pair queries.
The baseline already includes PR #2094's NEON basis-map improvement.

All entries are medians of 18 trials. Each trial uses a different public
transcript separator, shared by all configurations, to sample query-path
and proof-size variation. One warm-up per configuration is excluded; no
other samples are discarded. Every configuration occupies each of the six
execution positions three times. Runs are sequential, with tracing disabled
and no concurrent compilation or test jobs launched by this investigation.

| Symbols per leaf | Prove (ms) | Verify (ms) | Proof (decimal kB) | Prove reduction | Proof increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1, original MMCS | 145.005 | 4.626 | 487.064 | — | — |
| 1, adapter control | 143.748 | 4.933 | 487.083 | noise-level difference | 19 bytes |
| 2 | 103.881 | 3.957 | 488.589 | 28.4% | 0.3% |
| 4 | 83.063 | 3.641 | 497.248 | 42.7% | 2.1% |
| 8 | 71.934 | 3.380 | 566.203 | 50.4% | 16.2% |
| 16 | 68.337 | 3.472 | 739.617 | 52.9% | 51.9% |

[Raw CSV, including warm-ups](benchmarks/grouped-leaves-m4-pro.csv).
The CSV retains outliers: baseline proving times span 142.6–166.0 ms;
four-symbol times span 81.4–84.1 ms. These are host observations, not
confidence intervals or a native x86 performance claim. An earlier
nine-trial fixed-transcript run gave 143.7 / 103.4 / 83.6 / 71.8 / 69.2 ms
for original / 2 / 4 / 8 / 16, corroborating the ordering.

Proof sizes vary with the roots and sampled paths. Pair grouping sends no
additional field values, but changed roots change the transcript, so its
multiproof byte count can be above or below the original for a given trial.
The one-symbol control retains exactly the original transcript and adds one
empty-vector length byte per committed round (19 bytes total). Its verifier
pays adapter reconstruction overhead; the larger groups recover that cost
by hashing fewer leaves.

## Where the time goes

A separate traced fixed-transcript run shows the expected reduction in
Merkle work. These are illustrative phase timings, not medians of the CSV;
intermediate entries sum all 18 intermediate commitments. Parent and child
spans are not added together.

| Symbols per leaf | Initial Merkle (ms) | Intermediate Merkle total (ms) | All Merkle (ms) |
| ---: | ---: | ---: | ---: |
| 1, original | 41.0 | 49.9 | 90.9 |
| 2 | 20.9 | 28.0 | 48.9 |
| 4 | 11.4 | 17.1 | 28.5 |
| 8 | 6.72 | 10.8 | 17.5 |
| 16 | 4.94 | 8.75 | 13.7 |

Grouping four removes roughly 62 ms of Merkle work, explaining the full
prover improvement. Codeword folding still takes about 5.4–5.9 ms, residual
sumcheck rounds about 4.9–5.3 ms, and AIR zerocheck about 24–28 ms. Grouping
attacks hash overhead; it does not shrink the encoded 32 MiB or remove the
field-arithmetic work.

## Implementation and review scope

`GroupedCodewordMmcs` wraps the existing MMCS using a zero-copy matrix view.
The original width-one matrix remains available to the existing PCS fold
code. A group becomes one wider row for commitment; small codewords cap the
group size at their own length. The adapter intentionally accepts only the
single power-of-two-height column used by BinaryPcs, not general matrix
batches.

Queries still address the original symbols. Multiproofs authenticate sorted,
distinct groups and transmit only group members absent from the requested
symbol rows. Requested rows retain their original order and duplicate
handling. The verifier derives dimensions/group size from its configuration,
checks shapes and conflicting duplicates, reconstructs complete groups, and
passes them to the underlying MMCS verification.

The existing AIR, code, security/query calculator, fold arithmetic, query
sampling, and commitment schedule are unchanged. This does change roots and
proof encoding; applications must agree on grouping as part of their PCS
configuration. No existing example default is changed. Multi-round folding,
interleaving/head collapse, and rate changes are outside this prototype.

Validation: all 65 binary PCS library/integration tests passed (one existing
ignored test), including new commitment-equivalence, duplicate-query,
malformed-opening, supplemental-symbol tampering, tiny-round, and serialized
PCS lifecycle coverage. Every timed AIR proof is serialized, decoded, and
verified. Native Clippy with warnings denied passed for the PCS library/tests
and benchmark harness. x86_64 Linux and wasm32 library checks passed. A
read-only code review found no authentication/indexing issues; its execution
order observation was addressed before the final benchmark run.

This is a measured follow-up candidate for Codex review, not a merged
protocol change. Prefer an opt-in four-symbol configuration for the first
follow-up PR; expose eight as a proving-time/proof-size tradeoff.

## Reproduce

```sh
RUSTFLAGS='-C target-cpu=native -C opt-level=3' cargo build \
  --profile optimized --features parallel -p p3-multi-stark \
  --example bench_binary_grouping
RUST_LOG=off target/optimized/examples/bench_binary_grouping 18 18 vary > grouped.csv
# Fixed transcript, useful for phase inspection; iteration zero is the warm-up.
RUST_LOG=info target/optimized/examples/bench_binary_grouping 0 18
RUSTFLAGS='-C target-cpu=native -C opt-level=3' cargo test \
  --profile optimized --features parallel -p p3-binary-pcs --lib --tests
```

Group `0` in the CSV is the original MMCS, `1` the adapter control. Prove and
verify timers wrap the full calls, including their argument construction.
Trace generation, setup, serialization, and deserialization are outside
the timers. For medians, retain rows with `iteration > 0` and group by
`group_size`; the `seed` column records the public transcript separator suffix.
