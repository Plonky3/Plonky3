# Supported configurations

Optional, statically dispatched Plonky3 configurations. Enable `baby-bear`,
`koala-bear`, or `binary`; no backend is enabled by default. The crate uses
`no_std` plus `alloc`. The optional `parallel` feature enables Rayon-backed
parallelism in the selected backends.

| Feature / module | Supported stack | Entry point |
| --- | --- | --- |
| `baby-bear` / `baby_bear` | BabyBear, quartic binomial extension, two-adic FRI | `baby_bear::new(FriParameters<()>, cap_height)` |
| `koala-bear` / `koala_bear` | KoalaBear, quartic binomial extension, two-adic FRI | `koala_bear::new(FriParameters<()>, cap_height)` |
| `binary` / `binary` | GF(2^128), additive-domain binary PCS, Keccak-256 | `binary::Config::new(BinaryPcsConfig)` |

Each module exposes concrete `Val`, `Challenge`, `Challenger`, `Mmcs`, `Pcs`
and `Config` types. Underlying primitive crates remain generic; this facade
introduces no common prover trait or dynamic dispatch.

## Prime-field proofs

The prime stacks use the field crates' deterministic
`default_babybear_poseidon2_16/24` and `default_koalabear_poseidon2_16/24`
permutations. These use the checked-in, Grain-LFSR-generated round constants
published in `baby-bear/src/poseidon2.rs` and `koala-bear/src/poseidon2.rs`;
no seeded or runtime-random permutation parameters are selected. Both stacks
use a width-24/rate-16/output-8 sponge, width-16 compression of two eight-field
digests, a width-24/rate-16 duplex challenger, and `Radix2DitParallel` DFT.

Pass an explicit `FriParameters<()>`: blowup, final polynomial length, maximum
folding arity, query count and all three FRI grinding settings remain your
choices. The constructor replaces only `mmcs: ()` and uses the same cap height
for both Merkle trees. It returns a `StarkConfig`, so STARK-level out-of-domain
and lookup grinding remain available through its builder methods. Underlying
PCS parameter validation and trace-size constraints still apply.

The `uni_stark` reexport supplies `prove`, `verify`, `Proof`, and
`setup_preprocessed` / `prove_with_preprocessed` / `verify_with_preprocessed`.
Without preprocessed columns there is no setup phase. Every prove/verify call
clones the configuration's initial challenger; the example also constructs
a separate verifier configuration from the same parameters.

```sh
cargo run -p p3-configs --features baby-bear,koala-bear --example prove_prime_fields
```

The example proves an eight-row Fibonacci AIR over each field, serializes the
proof using postcard, verifies with fresh state, and rejects changed public
inputs. Its inexpensive parameters are for demonstration only.

## Binary multilinear proofs

Build `binary::BinaryPcsConfig::try_new(stacked_num_variables, params)` with
explicit `BinaryPcsParams`. The PCS arity includes *all* stacked trace columns
and tables: one two-column trace with `2^k` rows needs `k + 1` variables.
Mixed-height batches follow the PCS's suffix stacking/padding layout; the
caller must compute the arity of the whole batch. Individual trace tables
must contain at least two rows. The adapter supplies the suffix witness
layout and committed-table access used by the multilinear prover.

Use `Config::with_preprocessed` with a separately validated PCS configuration
before setup if the AIR declares preprocessed columns. Its stacked arity may
differ from the main trace. Omitting it for such an AIR causes setup to panic;
configurations do not infer AIR dimensions. The proof tests cover different
main/preprocessed widths, reusable keys, and tampered preprocessed openings.

The `multi_stark` reexport supplies `setup`, `prove`, `verify`, instance types,
and `MultiStarkProof`. Use a fresh `binary::challenger(application_domain)`
for each operation, with matching domain bytes and application context on
both sides. Multilinear sumcheck grinding is a separate explicit argument to
`prove` and `verify`; it is not the PCS's `pow_bits` setting.

```sh
cargo run -p p3-configs --features binary --example prove_binary_field
cargo test -p p3-configs --all-features --all-targets
```

The binary recurrence example moved from `p3-multi-stark` to this crate so
its public adapter can own the plumbing without a dependency cycle. It retains
the same field, hash, parameters and transcript domain. It serializes a real
PCS proof and verifies with a fresh transcript.

## Security responsibility and limitations

These configurations are **non-hiding** and do not provide zero knowledge.
Fixed permutation parameters do not establish an end-to-end security target.
Callers must assess their AIR, field/extension size, trace dimensions, batching,
commitment collision bounds, query count, grinding settings and applicable
soundness assumptions together. For prime proofs, `uni_stark` reexports
`StarkSecurityParams`, `AirLayout` and `OpeningShape`; use the selected FRI
parameters' `security_regime()` and `grinding_sites()` when doing that analysis.
Benchmark and testing presets are not production security recommendations.

The binary `security_level` prices PCS proximity queries and field width;
it excludes the surrounding STARK's error terms and commitment collision
resistance. The fixed Keccak-256 digest has at most a 128-bit generic collision
bound, even if a higher PCS target is requested. Binary PCS configuration construction checks supported parameter ranges,
not the complete protocol's soundness. Prime configuration constructors only
assemble the stack; invalid FRI parameters or incompatible trace dimensions
can panic during proving or fail verification. Characteristic-two LogUp
lookups are unsupported; do not apply the prime-field lookup argument to
the binary configuration.

This facade supports the listed backends only. Circle FRI, STIR, WHIR, hiding
PCS variants, alternate hash/DFT choices and custom transcript initialization
remain available through the primitive crates. Serialization uses the underlying
proof types; this crate adds no wire format or cross-version compatibility
promise. Applications must arrange matching AIRs, parameters, public inputs
and transcript context independently of the serialized proof.
