# Plonky3 binary backend: remaining work

This document replaces the original binary-field plan with a plan measured against the repository as it exists now.

It answers one question: what still has to land before Plonky3 is a complete proving backend for a separate RISC-V zkVM?

Recursion is outside the current execution scope. The non-recursive backend must remain compatible with a future upstreaming of Plonky3-recursion, but this plan does not move or modify recursive code.

The snapshot is 19 September 2026:

- Plonky3 `840779a03335`
- Flock `b684b1258e4b`
- leanVM `11807108dbfe`
- binius64 `b5191c6e51e4`
- worldfnd/whir `c03a4a512bd9`
- flock-challenge-multi `1b55c6ee2c9c`
- Plonky3-recursion `64884a20bab6`

Open Plonky3 work is described as open, not as landed. In particular, binary WHIR is [PR #2198](https://github.com/Plonky3/Plonky3/pull/2198).

## Executive conclusion

The binary backend no longer lacks its arithmetic substrate. It has binary fields, bit packing, additive transforms, Boolean commitments, optimized zerocheck rounds, a direction-aware bus representation, product GKR, and the first half of a word constraint system.

The remaining gap is composition.

Several cryptographic primitives now exist in isolation, but no released proof binds all of the following into one statement:

1. committed binary trace columns;
2. per-table constraints;
3. cross-table bus products;
4. offline-memory claims;
5. word-level constraints where a dense gadget needs them; and
6. a compact binary WHIR opening.

That distinction is security-critical. A product-GKR proof can be internally correct while its terminal leaf claims are unrelated to the committed trace. A word witness can be laid out efficiently while no proof checks its constraints. Neither primitive is an end-to-end backend until those claims are authenticated by the common opening protocol.

The shortest critical path is therefore:

```text
#2198 binary WHIR
        │
        ▼
binary multi-STARK PCS integration
        │
        ▼
bus products bound to committed tables
        │
        ▼
offline memory + unified table sumcheck
        │
        ▼
stable backend contract + tiny integration machine
```

The word proof protocol can proceed beside this path. It is necessary for competitive hash and big-integer precompiles, but it is not necessary to prove the first table-based machine.

## Completion criteria

The binary backend is complete when one public API can prove and verify a multi-table statement with all of these properties:

- trace values are committed at their natural bit or small-field width;
- table heights may differ without padding every table to the largest height;
- local constraints and bus shares are checked in one authenticated proof;
- read-write memory is checked by a reusable backend primitive;
- compact multi-rate binary WHIR is the opening scheme;
- every soundness contribution appears in one derived security report;
- proof parsing is bounded, versioned, and independent of prover-controlled shape;
- a downstream machine supplies tables and witnesses without importing protocol internals; and
- the repository publishes reproducible throughput, proof-size, verifier-time, and peak-memory results.

For the first release, the PCS may remain binding-only. Its public types and documentation must say so. Hiding must not be implied by the presence of zero-knowledge sumcheck code elsewhere in the workspace.

## What has changed since the original plan

The original plan correctly identified the major systems, but its remaining-work count is now misleading.

| Epic | Current state | Remaining gap |
| --- | --- | --- |
| E1 · packed arithmetic | Substantially landed | AArch64 GHASH and AES-field specialization remain platform gaps. |
| E2 · small-field commitment | Boolean packing and commitment landed | AIR execution still commonly stores full-width field cells; Basic Jagged is in review, while PCS composition and the assist remain open. |
| E3 · transforms | Landed or closed by measurement | Finish measured AArch64 work; GPU remains optional. |
| E4 · binary WHIR | In review in #2198 | Integrate it with the Boolean/multi-STARK path, freeze production profiles, and enforce budgets. |
| E5 · sumcheck kernels | Many optimizations landed | Multipoint ring-switch batching remains; other ideas require a new profile before code. |
| E6 · binary bus | Planning, leaf materialization, security accounting, and product GKR landed | Terminal claims are not connected to committed trace columns; mutable memory and unified table proving remain. |
| E7 · word constraints | IR and compiled packed witness layout landed | The shift reduction, nonlinear reductions, transcript, proof, and PCS discharge do not exist. |
| E8 · backend contract | Individual ingredients exist | There is no stable end-to-end machine-facing contract, proof envelope, cost report, or integration example. |
| E9 · recursion | A substantial external recursion repository exists | Its mature path is prime-field WHIR; it does not verify the new binary stack. |
| E10 · security and scoreboard | Strong local accounting exists | There is no common end-to-end benchmark, independent verifier, or complete cross-implementation vector suite. |

### Landed substrate

The following should be reused rather than rebuilt:

- `p3-binary-field`: tower, GHASH, `GF(2^64)`, its cubic challenge extension, AES field, packed bits, transpose, linear maps, and hardware kernels;
- `p3-binary-dft`: additive LCH transform and subfield-message specialization;
- `p3-binary-pcs`: field-generic BaseFold, Boolean trace packing, bit ring switching, and claim pooling;
- `p3-sumcheck`: pinned skip rounds, projective round messages, bit-sliced early rounds, representation-field weights, and in-place suffix handling;
- `p3-bus`: named direction-aware declarations, mixed-height layouts, fingerprint leaves, identity padding, product GKR, and bus security terms;
- `p3-word`: checked word relations and shift semantics;
- `p3-word-backend`: compiled shift keys and packed witness layout; and
- `p3-security`: explicit assumptions and named error terms.

### Work already in review

[PR #2198](https://github.com/Plonky3/Plonky3/pull/2198) changes the E4 starting point materially. It provides an additive-domain WHIR adapter, binary base/challenge fields, extension-codeword commitment, a multi-rate schedule, stratified queries, and transcript binding for the new domain and query shape.

It also preserves the prime-field path rather than routing prime fields through binary abstractions. This separation is the right performance boundary: shared protocol control flow, specialized domain arithmetic.

The PR should be treated as the implementation of most of E4.1, E4.2, E4.4, and E4.5. It is not yet the binary multi-STARK backend.

The security regimes must keep their names precise:

- unique decoding is theorem-backed and conservative;
- the Johnson regime uses the now-proven Reed–Solomon proximity-gap bounds already modeled in `p3-security`, while its complete WHIR composition must still cite every required correlated-agreement hypothesis accurately; and
- the capacity regime remains conjectural and must never be the silent production default.

The original 2024 WHIR analysis predates the newer proximity-gap result used by the present security crate. Documentation must distinguish the theorem added later from any assumption that remains in the composition.

Two other open PRs matter to the track without changing its critical path:

- [PR #2222](https://github.com/Plonky3/Plonky3/pull/2222) adds the one-byte NEON butterfly. It narrows the non-GFNI transform gap but does not close the separate AArch64 GHASH and AES-field work.
- [PR #2207](https://github.com/Plonky3/Plonky3/pull/2207) adds a binary-field SHA-256 AIR. It is a useful real workload for P1 and P2, but one hash AIR is not the backend contract or the word proof protocol.

## Disposition of the unfinished original items

This table prevents an old item from being implemented after its premise has changed.

| Original item | Disposition now |
| --- | --- |
| 1.3 · NEON packed GHASH | Still open. Requires real AArch64 execution, differential tests, and benchmarks. |
| 2.4 · small-field multi-STARK | Partial. Boolean commitment exists, but natural-width AIR execution and the WHIR integration in P1 remain. |
| 2.5 · jagged stacking | Basic Jagged is in review in #2225. PCS composition and the Jagged Assist remain in P11. |
| 3.6 · GPU | Deferred until the CPU protocol and ingestion contract stabilize. |
| 4.1 · WHIR domain abstraction | Implemented in #2198, pending review. |
| 4.2 · multi-rate schedule | Implemented in #2198, pending production-profile work in P3. |
| 4.3 · proven list decoding | Partly resolved by newer Reed–Solomon proximity-gap theorems and current `p3-security`; the complete WHIR assumptions still need an explicit audit in P0/P3. |
| 4.4 · grinding | Implemented in #2198 and security-accounted; keep transcript/config parity tests. |
| 4.5 · stratified queries | Implemented in #2198. |
| 4.6 · proof budget | Open; P2 and P3. |
| 4.7 · hiding hooks | The binding limitation is explicit, but a stable masking seam is still future work. It does not block the first release. |
| 5.2 · elide the C side | Not generally applicable to the AIR path. Reconsider only for a measured R1CS-shaped word reduction. |
| 5.3 · circuit walking | Not a current backend primitive. Keep it with a downstream circuit frontend unless a concrete Plonky3 lincheck needs it. |
| 5.4 · packed round kernels | Substantially advanced by recent sumcheck PRs. Profile before proposing deferred reduction or another representation crossover. |
| 5.5 · multipoint ring-switch batching | Open and important once jagged small-field claims or recursion make verifier work dominant; included in P11. |
| 5.6 · AG-code skip | Research only. |
| 6.1 · fingerprint product | Primitive landed. End-to-end authentication remains P4. |
| 6.2 · product GKR | Primitive landed, including radix-four batching and identity suffixes. End-to-end authentication remains P4. |
| 6.3 · counts in the exponent | Superseded for general memory by the ordered read-write trace in P6. Keep LogUp* for static indexed tables. |
| 6.4 · stacking and point recycling | Layout pieces landed; table authentication and safe point recycling remain P4/P5. |
| 6.5 · batched table sumcheck | Open; P5. |
| 6.6 · ergonomics and debugger | The balance debugger is in review in #2223; typed boundaries remain P7. |
| 6.7 · identity-padded dead cells | Landed in product GKR. It still needs end-to-end cost measurement. |
| 7.1–7.2 · word IR and relations | Landed. |
| 7.3 · shift reduction | Open; P8. |
| 7.4 · ordered reductions | Open; P9. |
| 7.7 · chip composition | Open; P10. |
| 8.1–8.3 · table, bus, and ingestion contract | Individual APIs exist, but no stable compiled backend statement exists; P11/P12. |
| 8.4 · offline memory | Static indexed reads use LogUp*; authenticated mutable memory remains P6. |
| 8.5 · read-write memory | Required for the RISC-V backend; P6 uses an execution-order/memory-order multiset argument. |
| 8.6 · segmentation | Public boundary values exist; a chaining contract and proof metadata remain P14. |
| 8.7 · proof format | Mature implementation exists in Plonky3-recursion; extract it in P13. |
| 8.8 · cost hooks | Open; P14. |
| 8.9 · tiny machine | Open; P15. |
| 9.1–9.4 · recursion | Deferred to the future Plonky3-recursion upstreaming effort. |
| 10.1–10.4 · accounting and tests | Strong partial coverage. Extend alongside every protocol PR rather than in one cleanup PR. |
| 10.5 · scoreboard | Open; P2. |
| 10.6 · independent verifier | Open; P19. |
| 10.7 · formal seams | Preserve named statements and hypotheses during P4–P15 and P19; formal proof itself remains a separate program. |

## What the reference systems actually teach us

### Flock

Flock is the closest performance reference for batch Boolean computation. Its decisive properties are not merely “binary fields”:

- the committed witness remains bit-valued;
- zerocheck and lincheck are specialized to the statement shape;
- Ligerito changes code rate across recursive levels;
- ring-switched claims use a Frobenius-aware multipoint assist;
- jagged inputs are charged by live data rather than maximum provisioned height; and
- the benchmark includes witness generation, proof construction, and serialization.

The Flock C-side elision and circuit-walking optimizations are specific to its batch-R1CS lincheck. Plonky3 should not acquire a second R1CS PIOP merely to copy those optimizations. They become applicable only if the word backend or a concrete downstream adapter produces the same algebraic shape and a profile shows the same bottleneck.

### leanVM

leanVM is the best reference for the full backend shape:

- a 64-bit column field and 192-bit challenge field;
- table constraints connected by a characteristic-two multiset bus;
- offline memory checking;
- WHIR/Ligerito commitments;
- a verifier whose proof shape is fixed by public parameters;
- recursion; and
- an independent Python verifier.

Its published M4 results are useful targets, not acceptance evidence for Plonky3. Different workloads, hashes, security budgets, and witness accounting must be normalized before comparing numbers.

### binius64

binius64 remains the design oracle for the word lane:

- shifts are linear views rather than materialized gates;
- only nonlinear relations enter the expensive reductions;
- reductions run in a transcript-critical order; and
- the native verifier, circuit emitter, and witness filler follow one shape.

The existing `p3-word` IR follows this direction. The next work is a proof protocol, not more relation syntax.

### worldfnd/whir

The worldfnd implementation is useful for checking prime-field WHIR structure and parameter derivation. It explicitly separates unique-decoding, provable-list, and conjectural-list choices, and it describes itself as a research implementation. It is a protocol reference, not a production API to transplant.

### Flock challenge

The challenge repository supplies a valuable benchmark contract: a fixed batch workload, a proof-size ceiling, and timing that includes witness generation and serialization while keeping verifier code fixed. Plonky3 should adopt that discipline, with its own checked security profile, rather than copying one headline throughput number.

### Plonky3-recursion

Plonky3-recursion already contains far more than the original plan assumed:

- circuit construction and witness filling;
- native and circuit challengers;
- Merkle and hash gadgets;
- prime-field WHIR verification;
- aggregation; and
- a bounded, versioned artifact format.

The binary work should generalize and upstream reusable parts of this implementation. Rewriting a second recursive verifier in the main repository would create verifier drift.

The current recursive configurations are nevertheless prime-field and Poseidon-oriented. A binary proof using `GF(2^64)`, a 192-bit challenge extension, an additive domain, and a byte-oriented standard hash needs new field, transcript, MMCS, and WHIR gadgets.

## Ordered PR plan

The numbering below describes dependency order. Independent lanes are marked explicitly.

### P0 · Finish binary WHIR

**Existing work:** #2198

Merge only after the following are true:

- prime-field proof bytes and benchmark distributions remain unchanged;
- additive-domain selector and fold identities have scalar-reference property tests;
- every rate, query count, grinding contribution, and assumption is represented in the security report;
- transcript tests fail if any new domain, codeword, OOD value, grinding result, or query schedule binding is removed;
- malformed proof lengths are rejected before allocation or transcript replay; and
- production examples do not default to the capacity conjecture.

This PR should not absorb multi-STARK integration. Its job is a field- and domain-generic WHIR PCS with no prime regression.

### P1 · Connect Boolean traces to binary WHIR

**Crates:** `p3-binary-pcs`, `p3-multi-stark`, `p3-whir`

Build the adapter that takes natural-width Boolean or small-field columns, preserves their binding constraint through ring switching, and discharges the resulting claims with the additive-domain WHIR PCS from P0.

The proof must not silently expand a bit trace into one `GF(2^128)` value per bit before commitment. Any unavoidable conversion must be explicit in the type and cost report.

Acceptance tests:

- a valid Boolean multi-STARK proves and verifies through WHIR;
- flipping a packed witness bit, ring-switch claim, final codeword symbol, or opening value fails;
- prime-field configurations instantiate the same public traits without taking binary conversion paths; and
- BaseFold remains available as a conservative independent configuration.

This PR establishes the compact PCS spine used by every later protocol.

### P2 · Add reproducible end-to-end scoreboards

**Crates:** benchmark crate or examples, CI metadata only

Create two fixed workloads:

1. the Flock challenge's batch of `2^18` BLAKE3 compressions, or an exactly documented equivalent if the frontend is not yet available; and
2. a Keccak-f batch matching the published Flock comparison closely enough to state every remaining difference.

Report:

- witness generation;
- constraint/witness packing;
- commitment and encoding;
- sumcheck and bus proving;
- opening proof;
- serialization;
- verification;
- proof bytes; and
- peak resident memory.

Publish single-thread and configured-parallel results on x86-64 and Apple silicon. CI should gate deterministic quantities such as proof shape and size. Wall-clock regression gates belong only on controlled runners.

This work happens early because it decides which later E5 optimizations are real.

### P3 · Freeze supported binary WHIR profiles

**Crates:** `p3-whir`, `p3-security`

Provide named, derived profiles rather than copied vectors of rates and queries:

- conservative unique decoding;
- theorem-backed Johnson-distance operation with its complete assumptions visible; and
- an explicitly experimental capacity profile.

Each profile owns its per-level rates, query allocation, proximity grinding, optional query grinding, and target error. Construction fails if the composed report misses its target.

Add deterministic proof-size and verifier-work budgets at representative sizes. The initial target is under 400 KiB at the agreed Boolean workload, but the test must bind an exact workload and security level rather than a slogan.

P3 may be folded into P0 if #2198 already exposes the required types cleanly. It should not become a second parameter system.

### P4 · Bind bus products to committed table columns

**Crates:** `p3-bus`, `p3-multi-stark`

This is the most urgent soundness PR after binary WHIR.

Add the bus proof to the multi-STARK transcript:

1. commit every table first;
2. sample tuple-fingerprint and product-offset challenges;
3. materialize push and pull factors according to the public `BusPlan`;
4. prove their products with `ProductGkr`;
5. expose the terminal point and per-block evaluations; and
6. authenticate every table-owned terminal evaluation against the committed trace.

The verifier must derive all block offsets, widths, heights, and tree shapes from public metadata. No proof field may choose a loop bound or allocation size.

Negative tests must independently corrupt roots, round polynomials, terminal values, direction metadata, domain separators, selectors, block offsets, and the committed trace.

The current `p3-bus` README correctly says that terminal claims are unauthenticated. Remove that warning only when this PR supplies the missing binding.

### P5 · Unify mixed-height table constraints and bus shares

**Crates:** `p3-multi-stark`, `p3-bus`, `p3-sumcheck`

Implement the back-loaded table sumcheck:

- a table joins when its own variables begin;
- unused leading variables contribute a factor whose Boolean-cube sum is one;
- local constraints and authenticated bus shares enter one composition; and
- the product-GKR terminal point is recycled as the zerocheck point only after proving that its sampling order meets zerocheck's independence requirement.

Benchmark this against one sumcheck per table. Retain the unified path only if proof size or prover time improves without increasing verifier risk or transcript ambiguity.

P4 and P5 may be reviewed as a stack. P4 must not expose an apparently complete bus verifier whose leaf claims are unauthenticated.

### P6 · Add offline read-write memory

**Crate:** `p3-bus`

Expose authenticated mutable memory as a reusable backend primitive, not as a VM chip.

The API describes initial state, ordered accesses, and final state. It does not know about program counters, registers, or bytecode.

The proof must bind two views of the same accesses:

- execution order, where the caller supplies each read or write;
- memory order, sorted by address and then timestamp;
- a bus product proving both views contain the same access multiset;
- strict address/timestamp ordering without field-order assumptions;
- read continuity and write updates within each address run; and
- authenticated initial and final boundary rows for segmentation.

Integer ordering must be proven through checked bit decomposition or an equivalent range argument; comparing field elements is not an ordering proof. All compression, range, and batching errors belong in `p3-security`.

Add adversarial tests for a missing access, wrong value, duplicate timestamp, timestamp wrap, read-before-write, invalid write transition, substituted boundary row, and a mismatch between the two trace orders.

Do not duplicate LogUp*. Static read-only arrays and bytecode tables should use that existing characteristic-independent indexed-lookup protocol. P6 exists for mutable state, whose temporal consistency obligation is different.

### P7 · Finish bus ergonomics and diagnostics

**Crates:** `p3-bus`, `p3-air`

Complete the machine-facing declaration layer:

- typed, caller-owned domain separators;
- first-class boundary pushes and pulls;
- widths large enough for realistic table tuples without heap work per row;
- source table and row retained in debug builds; and
- a balance debugger that prints the unmatched tuple and both emitting locations.

Keep proving types free of debug provenance. The debugger should reuse the same declaration and tuple evaluation logic as the prover so diagnostics cannot disagree with the proof.

### P8 · Prove the word shift reduction

**Crates:** `p3-word-backend`, `p3-sumcheck`

This begins the parallel word lane.

Reduce the compiled shifted operands to claims on the committed word trace. Use factorized tables for the two shift slots and bit indices rather than materializing the full tensor.

The verifier derives key order and reference order from `ConstraintSystem`; proof data never supplies them. Scalar reference tests cover every shift kind, boundary amount, 32-bit lane boundary, sign pattern, repeated reference, and composed shift.

Benchmarks must report both prover time and resident memory. The packed witness layout is valuable only if the reduction consumes it without expanding back into per-bit vectors.

### P9 · Complete the word proof protocol

**Crates:** `p3-word-backend`, `p3-binary-pcs`, `p3-security`

Implement the nonlinear reductions in the transcript order required by their dependencies:

1. integer multiplication;
2. binary multiplication;
3. bitwise AND;
4. zero checking;
5. shift reduction; and
6. public-value checking.

Commit every prover-chosen intermediate before sampling the challenge that combines it. Document the order beside the transcript type and lock it down with mutation tests.

Finish with claims on the same Boolean PCS used by the table path. A word proof must not introduce a second commitment stack.

### P10 · Add backend chip composition

**Crates:** `p3-word`, `p3-word-backend`, `p3-bus`

Represent a reusable constraint system plus an instance count as a backend chip. Calls contribute witness columns and bus entries without inlining the constraint graph for every instance.

This is the bridge for dense hash, big-integer, and signature precompiles. The circuit builder and gadget library remain downstream. Plonky3 owns only the proof statement and its composition.

### P11 · Define zero-copy trace ingestion

**Crates:** `p3-air`, `p3-matrix`, `p3-binary-pcs`, `p3-sumcheck`

Start with a short design PR measured against a real downstream witness generator.

The contract must accept:

- borrowed contiguous columns;
- packed bit-sliced columns;
- per-thread chunks; and
- columns with distinct logical heights.

It must state when the backend borrows, reinterprets, gathers, transposes, or allocates. A hidden full-trace conversion is not zero-copy.

Follow with capacity-free jagged stacking and its opening reduction. Authenticate public selectors and avoid padding each column or table to the largest height. Integrate the Frobenius-aware multipoint ring-switch assist here, because jagged small-field claims make verifier cost visible.

This is the highest-risk performance interface with the future VM repository. It should not be designed from synthetic matrices alone.

### P12 · Publish the backend contract

**Crates:** `p3-air`, `p3-multi-stark`, `p3-bus`

Introduce one stable descriptor for a proof statement. It contains only backend concepts:

- table and column schemas;
- degree bounds and local constraints;
- public values;
- bus interactions and boundaries;
- dynamic table heights;
- optional word chips; and
- the PCS/security configuration.

Compilation produces immutable prover and verifier keys with a shape digest. Runtime witness data is separate from the statement.

No VM term belongs in these types. Conversely, a downstream table must not reach into sumcheck, GKR, or PCS internals to obtain acceptable performance.

### P13 · Add a bounded, versioned proof envelope

**Crates:** `p3-multi-stark`, `p3-whir`

Adapt the proven frame design from Plonky3-recursion in the main workspace without moving recursive code or inventing a competing format.

The envelope binds:

- format version;
- protocol-suite identifier;
- statement-shape digest;
- field and hash suites;
- security profile; and
- proof payload length.

Decoding checks every length against the verifier key before allocation. Raw `serde` or `postcard` representation is an internal encoding detail, not the compatibility contract.

### P14 · Add segment claims and cost reports

**Crate:** `p3-multi-stark`

Expose public boundary claims that a downstream continuation policy can chain. The backend does not decide where a segment ends.

Return structured costs per table:

- rows and committed columns;
- live and padded symbols;
- bus pushes and pulls;
- sumcheck rounds;
- opening queries and bytes;
- prover phase times; and
- peak scratch allocation where measurable.

This data feeds the downstream instruction cost model and the common scoreboard.

### P15 · Add the tiny integration machine

**Location:** `p3-multi-stark/examples` or an integration-test crate

Build the smallest statement that exercises the complete contract:

- three or four abstract state transitions;
- two tables of different heights;
- one state bus;
- one mutable array with initial and final boundary rows;
- one boundary claim; and
- one batched word chip, preferably a small hash round rather than a VM instruction.

The example is not a RISC-V machine. It is an executable backend conformance test.

It must prove with binary WHIR, serialize through the versioned envelope, deserialize under explicit limits, verify, report costs, and fail under one corruption for every protocol layer.

This PR marks the first complete non-recursive binary backend.

### Future · Reconcile Plonky3-recursion with the main workspace

**Repositories:** Plonky3, Plonky3-recursion

Write a compatibility RFC before moving code.

Identify one owner for each reusable component:

- circuit IR and builder;
- native, circuit, and filler channels;
- artifact format;
- Merkle verifier;
- WHIR verifier logic; and
- aggregation driver.

Prefer upstreaming small generic crates or interfaces. Do not wholesale-copy the external repository, and do not preserve a vendored fork of the verifier after the APIs converge.

The success condition is one verifier algorithm exercised by native verification, circuit emission, and circuit witness filling.

### Future · Add binary recursion primitives

**Repository:** decided by P16

Implement circuit support for:

- `GF(2^64)` base values;
- the 192-bit cubic challenge field;
- additive-domain selector arithmetic;
- the byte challenger used by the native binary proof;
- the chosen standard hash and Merkle compression; and
- proof-of-work checks.

Every absorbed byte and domain separator must match the native transcript. No hash, field conversion, or Merkle node may be supplied as an unconstrained hint.

The hash choice must be made on recursive cost. Poseidon can remain available for prime configurations, but a binary proof must not silently switch transcript hashes when recursively verified.

### Future · Verify the complete binary proof in-circuit

**Repository:** decided by P16

Generalize the existing prime WHIR verifier to the additive binary domain, then add:

- stratified fixed-shape queries;
- multi-table zerocheck;
- product GKR;
- offline-memory claims;
- word reductions when present; and
- the versioned public boundary statement.

Run native and circuit verification from the same proof-shape description. A test compares every transcript event and rejects any divergence.

The backend emits the recursive circuit and its public inputs. Scheduling a workload-specific recursion tree remains downstream.

### P19 · Add independent verification and differential vectors

**Locations:** a dependency-free verifier outside the Rust workspace, plus vector generators in-tree

Publish vectors for:

- binary fields and basis conversions;
- additive NTT and multi-rate encoding;
- challenger events;
- Merkle caps and multiproofs;
- WHIR folds and queries;
- zerocheck and ring switch;
- bus fingerprints and product GKR;
- offline memory; and
- word reductions.

Consume compatible Flock vectors where representations match. State transformations explicitly where bases or transcript formats differ.

The independent verifier parses the bounded proof format and verifies the tiny integration statement. It must not call Rust through FFI.

### P20 · Release gates

**Scope:** CI and release process

Before calling the backend production-ready:

- run scalar-reference property tests for every specialized kernel;
- execute a native-feature test leg at the widest supported vector width;
- execute AArch64 tests on real hardware for every enabled specialization;
- fuzz proof decoding, transcript lengths, and malformed shapes;
- run transcript-absorb deletion tests;
- require an explicit security report for every named configuration;
- record proof-size and verifier-work regressions; and
- commission an external audit of the composed protocol, not only its crates.

## Dependency map

```text
P0 binary WHIR ──► P1 Boolean WHIR ────────────────┐
       │                    │                       │
       └────────► P2 scoreboard ─► P3 profiles     │
                                                    ▼
P4 bus binding ─► P5 unified table sumcheck ─► P6 offline memory
       │                    │                       │
       └────────────────────┴───────────────┐       │
                                            ▼       ▼
P8 shift reduction ─► P9 word proof ─► P10 chip composition
                                            │
P11 trace ingestion/jagged stacking ────────┤
                                            ▼
                              P12 backend contract
                                            │
                              P13 proof envelope
                                            │
                              P14 segments and costs
                                            │
                              P15 tiny integration machine
                                            │
                              P19 independent verifier
                                            │
                              P20 release gates
```

P8 through P10 are parallel to P4 through P7. P11 can begin as a measured design exercise after P1, but its protocol integration depends on the common claim and opening model.

## Work not justified yet

The following should not become implementation PRs until evidence changes.

### A second R1CS prover

Flock's C-side elision and circuit walker are excellent optimizations for its lincheck. Plonky3's primary composition is AIR tables plus a word backend. Add an R1CS-specific path only if a real consumer needs it and the common scoreboard shows that it beats both existing routes.

### More packed sumcheck kernels without a profile

Recent changes already added projective messages, bit-sliced early rounds, representation-field weights, and in-place suffix processing. Deferred reduction and a small-to-large switchover may still help, but they now require a phase profile and a field-specific primitive. A named epic is not benchmark evidence.

### AG-code skip on the critical path

Flock's algebraic-geometry skip is valuable frontier work. It brings new curves, generated constants, sampling rules, and audit surface. Keep it as a research branch until Reed–Solomon skip limits dominate an end-to-end Plonky3 profile.

### GPU support

GPU encoding, hashing, and dot products can follow a stable CPU proof and trace-ingestion contract. Building them earlier would optimize an interface that is still changing.

### A second static-memory protocol

LogUp* already proves static indexed reads in characteristic two. Do not add a parallel counts-in-the-exponent API unless a measured workload demonstrates a concrete advantage that justifies a second transcript, security argument, and integration path.

### Hiding before the binding backend closes

Keep masking seams visible, but do not mix a hiding retrofit into the first composed bus, word, and recursion proofs. The release documentation must continue to say that the binary PCS is binding-only.

## Performance policy

Every performance PR must contain:

- a fixed benchmark input;
- the exact security profile;
- before and after medians and dispersion;
- proof bytes and verifier work when protocol shape changes;
- peak memory for changes that alter layout;
- the target CPU and enabled features; and
- evidence that the prime-field path did not regress.

Do not merge an optimization merely because it is used by Flock, leanVM, or binius64. Their statement shapes and memory layouts differ.

The first end-to-end targets are:

- proof size below 400 KiB for the agreed Boolean batch workload;
- verifier time close to the Flock/leanVM class on the same machine and hash;
- no full-width expansion of the committed bit trace;
- no padding cost proportional to the largest table for every table; and
- no material prime-field regression.

These are engineering targets, not security parameters. Failure to hit one should produce a profile, not a weaker proof configuration.

## Security policy

Every protocol PR must answer five questions in code and review:

1. What exact statement is proved?
2. Which values are fixed before each challenge?
3. How are terminal claims authenticated?
4. Which theorem or explicit conjecture bounds each error event?
5. How does the error compose into the advertised bit target?

The verifier rejects malformed shape before replaying Fiat–Shamir. Loop counts, allocation lengths, table offsets, query strata, and field suites come from the verifier key, never from unauthenticated proof metadata.

Characteristic two requires direction to remain metadata. Push and pull must never be encoded as `+1` and `-1`, because they are equal in a binary field.

No zero-knowledge claim may be inferred from zero-knowledge sumcheck alone. Commitment hiding, final-codeword masking, and opening privacy are separate requirements.

## Repository boundary

Plonky3 owns:

- fields and commitments;
- AIR and word proof statements;
- sumcheck, GKR, bus, and memory reductions;
- proof serialization and verification;
- security and cost reports.

The downstream RISC-V repository owns:

- ISA semantics;
- ELF loading and transpilation;
- emulator and trace generation;
- instruction tables;
- memory and bytecode chips built on backend primitives;
- hash, big-integer, and signature gadget frontends;
- continuation policy;
- guest toolchain and host I/O; and
- recursion-tree scheduling and deployment wrappers.

OpenVM's adapter/core split is a useful consumer-side pattern. SP1 and ZisK demonstrate the importance of precompile batching, segmentation, and reproducible end-to-end measurements. None of those VM concepts should leak into the Plonky3 protocol types.

## Recommended execution order

### Milestone A · compact binary proof

Finish P0 through P3.

Exit when a Boolean multi-STARK proves through binary WHIR under an explicit security report and produces a reproducible proof-size result.

### Milestone B · composed table backend

Finish P4 through P7 while P8 through P10 proceed in parallel.

Exit when different-height tables prove local constraints, balance a bus, and check mutable memory with every terminal claim tied to committed columns.

### Milestone C · machine-facing backend

Finish P11 through P15.

Exit when the tiny integration machine uses only the public backend contract and round-trips a bounded, versioned proof.

At this point a separate RISC-V repository can build without reaching into protocol internals.

### Milestone D · independent verification and release

Finish P19 and P20.

Exit when the native and independent verifiers accept the same binary proof and the release gates pass.

Recursive verification remains a separate future project owned by the eventual Plonky3-recursion upstreaming effort.

## Sources

- Bünz, Rothblum, and Wang, [Flock: Fast Proving for Batch Boolean Computations](https://eprint.iacr.org/2026/1329)
- [succinctlabs/flock](https://github.com/succinctlabs/flock)
- [Layr-Labs/flock-challenge-multi](https://github.com/Layr-Labs/flock-challenge-multi)
- Arun, Chiesa, and Yogev, [WHIR](https://eprint.iacr.org/2024/1586)
- [worldfnd/whir](https://github.com/worldfnd/whir)
- [leanEthereum/leanVM](https://github.com/leanEthereum/leanVM)
- [binius-zk/binius64](https://github.com/binius-zk/binius64)
- [Plonky3/Plonky3-recursion](https://github.com/Plonky3/Plonky3-recursion)
- [OpenVM](https://github.com/openvm-org/openvm)
- [SP1](https://github.com/succinctlabs/sp1)
- [ZisK](https://github.com/0xPolygonHermez/zisk)
