# Binary PCS performance investigation

Measured on an Apple M4 Pro (14 cores), macOS ARM64, Rust 1.98.0, on
2026-09-08. Starting revision: `0aae8b57`. The workload is the nonlinear
recurrence in `p3-multi-stark`'s `prove_binary_field`, at 2^18 rows and two
GF(2^128) columns. The measured configuration retains rate 1/4, 100-bit
security, zero PCS PoW, Keccak-256, and a binary Merkle tree.

## Missing tracing time

The original outer `prove` span charged about 40% to its own work because
BinaryPcs did not instrument folding, intermediate commitments, or queries.
A representative instrumented baseline took 165 ms:

| Work | Time |
| --- | ---: |
| Witness layout | 0.286 ms |
| Initial PCS commitment, including encoding and allocation | 56.2 ms |
| AIR zerocheck | 28.6 ms |
| Opening claim evaluation | 8.23 ms |
| Residual sumcheck initialization | 6.30 ms |
| Intermediate Merkle commitments, 18 rounds total | 49.99 ms |
| Codeword folding, 19 rounds total | 6.75 ms |
| Residual sumcheck rounds | 7.04 ms |
| Query phase | 1.33 ms |

Rounding and small allocation/drop/transcript costs account for the remainder.
The previously missing portion is principally the last four rows, about
65 ms, rather than an unexplained field operation.

Tracing now covers commit, public `open`/`open_at` (including claim evaluation),
every fold's sumcheck/codeword/commit work, queries, and PCS verification.
The example installs `EnvFilter` and `ForestLayer` like `prove_prime_field_31`;
proof size remains printed, while tracing reports proving and verification time.
Nested inclusive times must not be added to their children a second time.
Some lines in the supplied terminal transcript are also visibly truncated;
that is separate from the missing spans.

## Retained optimization

On ARM with NEON, load each 128-bit basis-table entry into a vector register
and XOR the accumulator there. The tables, indices, and field representation
are unchanged. Other targets retain the scalar implementation. This improves
both tower-to-polynomial and polynomial-to-tower conversion, reaching AIR
arithmetic, opening evaluation, additive encoding, and codeword folding.

Existing conversion benchmarks (1,000 inputs per iteration, 10 samples,
1-second warm-up and measurement) measured:

| Conversion | Before | After |
| --- | ---: | ---: |
| Tower to GHASH | 3.1484 us | 1.6878 us |
| GHASH to tower | 3.1475 us | 1.6868 us |

Nine alternating full-prover runs, with separate executables built from the
instrumented baseline and optimized source, produced these rounded tracing
times in milliseconds:

- Before: 157, 158, 157, 158, 179, 157, 157, 157, 158.
- After: 152, 146, 145, 148, 145, 146, 144, 145, 148.

Median: **157 ms to 146 ms**, a **7.0% reduction**. This is an end-to-end
measurement, not a projection from the 46% conversion microbenchmark gain.
Tracing was enabled in both versions; benchmark runs were sequential.
A Keccak inlining experiment did not improve the median and was discarded.
No commitment layout, folding schedule, hash, or soundness parameters were
changed to obtain the retained speedup.

## Why WHIR does less work in the supplied runs

The user's warm WHIR runs are about 39-40 ms. They are external observations,
not a newly reproduced WHIR benchmark in this branch; the checked-out WHIR
test is the original BabyBear/Poseidon2 fixture configuration.

The logged commitment shapes nevertheless expose a large difference:

- Binary: 2^21 leaves, each containing one 16-byte element. The initial
  codeword is 32 MiB, and its tree costs about 46 ms in the baseline.
- WHIR: 2^18 leaves, each containing four 4-byte base-field elements. Its
  initial codeword is 4 MiB, and the supplied warm tree time is about 6.5 ms.

Both initial leaves contain 16 bytes, but binary hashes eight times as many
of them. Binary then commits after each single-variable fold. WHIR uses
interleaving, larger folding steps, and a different re-encoding schedule.
The binary scheme also uses unique decoding, while the WHIR test uses
`CapacityBound`; a common security-bit setting does not make their costs or
assumptions identical. PoW budgets and differing proof sizes in the supplied
runs further preclude treating them as a controlled field-only comparison.

BinaryField128's tower packing is scalar. Monty-31 has four-lane NEON packing;
the binary arithmetic additionally pays basis-conversion costs. Native ARM
already enables PMULL and the SHA3 Keccak path: software multiplication is
not the explanation here. Keccak batches two messages on ARM, versus four
or eight on applicable AVX2/AVX-512 builds. GHASH folding also has wider
packing on suitable x86 CPUs. These are code-level architectural differences,
not evidence of a measured x86/ARM speed ratio. No native x86 timings were taken.

## Reducing hashing volume: findings from other binary systems

Sources were inspected on 2026-09-08, including pinned source checkouts.

- **Binius64:** its FRI folder buffers challenges until designated commitment
  rounds, avoiding intermediate codewords. A commitment groups an entire
  queried coset into each leaf (`1 << log_coset_size`). The initial oracle
  also supports interleaved batches. See [FRI folder, revision 37e9cd64](https://github.com/binius-zk/binius64/blob/37e9cd64e82243cde0e79c7d8ac0dc319f1cbeb4/crates/iop-prover/src/fri/fold.rs)
  and [FRI parameters](https://github.com/binius-zk/binius64/blob/37e9cd64e82243cde0e79c7d8ac0dc319f1cbeb4/crates/iop/src/fri/common.rs).
- **Flock:** its Ligerito backend commits interleaved RS words: a leaf holds
  one position across all lanes, followed by multiple partial-evaluation
  or sumcheck rounds between recursive commitments. The source's usual
  initial interleave is 64, with configuration-specific exceptions. The
  inspected implementation keeps committed words in F128 and uses F256
  challenges. See [Ligerito source, revision a4a0b4a](https://github.com/succinctlabs/flock/blob/a4a0b4a9dd555f9b58853951c133d7c990af5dad/crates/flock-core/src/pcs/ligerito.rs).
  [The paper, Appendix C](https://arxiv.org/html/2607.27491v1) describes
  the interleaved construction. Its Boolean-witness packing is an additional
  benefit that cannot pack our already-full-width field elements further.
- **Bolt:** the published implementation benchmarks a 32-bit binary-field
  witness interleaved 128 ways. Its example codeword expands the message
  by 1.25x using a sparse-code syndrome plus RS encoding, instead of our 4x
  RS expansion. It commits entire rows. This is a different PCS, not a rate
  parameter we can copy into this scheme. See [bolt-rs](https://github.com/bcc-research/bolt-rs).

### Existing rate tradeoff, measured separately

Changing only `log_inv_rate` from 2 to 1 halves every codeword's size.
The existing unique-decoding calculator raises the query count from 148 to
241 at the same 100-bit target and zero PoW. Five alternating runs using
the optimized arithmetic measured:

- Rate 1/4: 146, 145, 144, 141, 147 ms; median **145 ms**.
- Rate 1/2: 109, 98.8, 106, 98.7, 98.4 ms; median **98.8 ms**.

All proofs verified. Serialized size rises from **490,739 to 640,157 bytes**
(about 30%). This is a roughly 32% proving-time reduction with a proof-size
tradeoff, rather than a free implementation speedup. The example retains
rate 1/4 so its before/after optimization measurement remains comparable;
to select this tradeoff, set `log_inv_rate: 1` in the example's `config`.

### Recommended next changes, in order

1. **Group adjacent symbols into leaves without skipping any folds.**
   Start with pairs, then benchmark 4/8/16-symbol groups. Preserve the sampled
   symbol indices and query count; map each query to its group and verify
   the required coordinates inside the authenticated group. Pairs already
   have to be opened together, so grouping two avoids redundant leaf hashes
   without increasing the field values revealed per query. Wider groups
   trade extra revealed values for fewer hashes and shorter paths.

   For the initial 2^21-symbol codeword, the theoretical Keccak permutation
   counts for a binary tree are:

   | Symbols per leaf | Leaves | Leaf + internal permutations |
   | ---: | ---: | ---: |
   | 1 | 2,097,152 | 4,194,303 |
   | 2 | 1,048,576 | 2,097,151 |
   | 4 | 524,288 | 1,048,575 |
   | 8 | 262,144 | 524,287 |
   | 16 | 131,072 | 393,215 |

   Each element is 16 bytes; Keccak-256's rate is 136 bytes. Up to eight
   elements fit in one padded permutation; sixteen require two. Internal
   binary nodes hash 64 bytes. These are operation counts, not predicted
   wall-clock speedups. Grouping changes roots/proof encoding, but need not
   change the polynomial, symbol-query distribution, or algebraic protocol.
   The verifier must bind grouping and dimensions and reject malformed groups.

2. **Commit after multiple folding challenges, as in Binius64.**
   This further reduces intermediate trees and can fuse fold passes. It
   changes the transcript and query relation: the verifier needs whole
   2^k-symbol cosets, and the soundness/error calculation must be re-derived
   for the schedule. Simply omitting roots from the current implementation
   is not a justified optimization.

3. **Add interleaved commitment/head folding.**
   This attacks initial tree height as Flock and WHIR do. Our `FOLDING = 0`
   currently enforces a width-one codeword, and its comments explicitly defer
   head collapse. An interleaved variant needs an equality-weighted column
   combination consistent with suffix binding, rather than changing that
   constant alone. Ligerito-style re-encoding is a larger PCS alternative.

4. **Evaluate tree arity and alternative codes separately.**
   Four 32-byte children fit into one Keccak rate block, reducing internal
   nodes, but leaves remain the dominant issue and query proofs change.
   Higher-rate/sketched codes can reduce total encoded bytes, as Bolt does,
   but need their own proximity analysis. Merkle caps/multiproofs mostly
   reduce authentication traffic, not the dominant initial leaf hashing.

The branch implements instrumentation and the measured NEON improvement.
The grouped-leaf and PCS redesigns above are recommendations, not implemented
or claimed secure by these measurements.

## Reproduce and validate

```sh
RUSTFLAGS='-C target-cpu=native -C opt-level=3' cargo run \
  --profile optimized --features parallel -p p3-multi-stark --example prove_binary_field
RUSTFLAGS='-C target-cpu=native -C opt-level=3' cargo bench \
  --profile optimized -p p3-binary-field --bench arithmetic -- \
  'representations/convert' --sample-size 10 --measurement-time 1 --warm-up-time 1
```

Validation performed: 316 native library tests across binary-field,
binary-dft, binary-pcs and multi-stark; 203 additional integration tests
across the three binary crates; eight binary AIR example tests. All passed;
four tests were intentionally ignored. The basis-map test checks every byte
value at every byte position against scalar table evaluation, alongside the
existing field/isomorphism and adversarial PCS tests. x86_64 Linux and wasm32
library cross-checks, targeted Clippy with warnings denied, and nightly
rustfmt/diff whitespace checks passed.
