# Caller obligations

A reduction in this workspace is sound only under some conditions. Most of those conditions are
checked: by a type, by a constructor, or by a runtime check that returns a typed error. This page
is for the rest — the conditions that cannot be checked from inside the crate that states them,
and that a caller therefore has to discharge itself.

The rule the workspace follows is, in order of preference:

```text
    a type carries it       ->  the compiler rejects the mistake
    a runtime check         ->  the code rejects the mistake, with a typed error
    this page               ->  neither is possible, and the reason is written down
```

Prose in a module doc is not one of the three. If you find an obligation stated only in a doc
comment, it belongs in one of the rows below, or it belongs in code.

## How the first two are done here

These are the patterns already in the tree. Reuse them rather than inventing a fourth.

| Shape | Pattern | Example |
| --- | --- | --- |
| A returned claim a caller could silently drop | `#[must_use]` on the **type**. The workspace denies `unused_must_use`, so dropping it is a compile error | `JaggedDenseClaim` (`sumcheck/src/jagged/mod.rs`), `Point` (`multilinear-util/src/point.rs`) |
| A value a prover could choose freely once some factor vanished | Reject the degenerate case. Absorbing the value afterwards pins nothing, because the absorb happens after the last challenge | `word-backend/src/shift/` |
| A security parameter supplied unchecked | Derive it from the object it has to match | `word-backend/src/proof/key.rs`, `SecurityTerm::over_candidates` |
| A statement dimension outside the transcript | Absorb it, as **length-delimited bytes, not field elements** — in characteristic two `from_usize(2)` and `from_usize(4)` are both zero | `bus/src/memory/`, `sumcheck/src/jagged/transcript.rs` |
| A public accessor that lets a caller drive the prover around its own checks | Make the internals private | `bus/src/memory/` |

## What a caller must discharge

Each row names the obligation, where it is stated, and **why** neither a type nor a runtime check
can carry it. The reason matters: a row without one is a row that should have been a fix.

### Transcript ordering

A crate can absorb a value. It cannot ask a challenger what it has already absorbed, or in what
order, because `CanObserve` / `FieldChallenger` expose no history. Every obligation of the form
"bind X before drawing Y" is therefore outside the reach of the crate that needs it, unless that
crate does the binding itself.

| Obligation | Stated at | Why not enforceable there |
| --- | --- | --- |
| The commitment must be absorbed, and the opening points bound, before `verify_at` or `open_at` is called | `binary-pcs/src/pcs.rs`, `binary-pcs/src/boolean.rs`, `binary-pcs/src/whir/boolean.rs`, `sumcheck/src/prescribed_pcs.rs` | The challenger is borrowed behind `FieldChallenger`; nothing can be asked of its history. Absorbing here instead would double-absorb for the callers that already did it right. A session token produced by `observe_commitment` and consumed by `verify_at` would close it, at the cost of a `PrescribedPointPcs` signature change |
| The column heights must be bound into the transcript, normally by the commitment, before the sparse point is drawn | `sumcheck/src/jagged/layout.rs` | `p3-sumcheck` has no PCS dependency. It seeds its own transcript from the full layout, which parts two jagged statements — but the point it is *handed* was drawn before that seed, by someone else |
| A caller-fixed opening point must already have been drawn from, or bound to, the same sponge | `sumcheck/src/layout/verifier.rs` (`add_claim_at`) | `PointSource::Given` deliberately emits no transcript step, so the point is invisible to Fiat-Shamir. Describing it as a message step would fix this and is the right follow-up |
| The columns a bus argument reads must already be committed | `bus/` | `p3-bus` has no PCS dependency and no way to query what a transcript absorbed |
| An opaque value's **width** is not bound by `observe_opaque`; bind it through the instance label | `challenger/src/fs/state/prover.rs` | The step records a scalar regardless of how many sponge units `C::observe` consumes. Binding the width here is a transcript-format change that invalidates every recorded fixture, so it is a coordinated change rather than a local one |
| Hint bytes ride the wire and are never absorbed, so only use hints the verifier re-derives or checks | `challenger/src/fs/state/prover.rs` (`add_hint`, `add_hint_bounded`) | That is what a hint is. The enforceable shape is a `Hint<T>` wrapper the verifier must validate before reading, which is a design change to the hint API |

### Properties of the committed witness

A reduction sees field elements. It cannot see which alphabet they came from, or what the AIR
around them constrains.

| Obligation | Stated at | Why not enforceable there |
| --- | --- | --- |
| Every witness the prover can commit to must keep the constraint prime-field-valued | `sumcheck/src/univariate_skip/pinned.rs` | The pinned fold has a large kernel over a wider alphabet, and any cell pattern inside it is invisible to the check. The premise has to come from the commitment alphabet — a prime-field-packed commitment or ring switching gives it; **a booleanity constraint in the same zerocheck does not** |
| The AIR must constrain every query count to the weight declared by `Count::bounded` | `lookup/src/count.rs` | The weight is the only input to the height bound `sum_i w_i * h_i < p`. A weight below the true per-row count lets a multiplicity wrap modulo `p`. The bound is over committed values the crate never sees; `lookup/src/debug_util.rs` walks a concrete trace and is the right place for a debug-build check |
| The AIR must constrain exclusive-branch flags to be boolean and to sum to at most one | `lookup/src/logup.rs`, `lookup/src/bus.rs`, `lookup/src/builder.rs` | Same reason. The trace-generation path does assert both on concrete values; the constraint system does not emit them |
| Entries past the message prefix must be zero in a padded encode | `commit/src/encoder.rs` (`encode_batch_padded`), `binary-dft/src/traits.rs` (`ntt_batch_padded`) | The default implementations evaluate the whole buffer and are safe under violation; an optimised implementation that skips the tail produces a different codeword. Only a `PaddedMessage` newtype built by a zeroing constructor closes it by type |

### Parameters the crate cannot see the other side of

| Obligation | Stated at | Why not enforceable there |
| --- | --- | --- |
| The digest must be at least `2 * security_level` bits wide | `binary-pcs/src/params.rs` | `p3_commit::Mmcs` declares `Commitment`, `Proof`, `ProverData` and `Error`, and nothing that reports a width — the digest width lives as a const generic on the concrete `MerkleTreeMmcs`. `BinaryPcsConfig` is also derived before any MMCS exists. Adding `fn collision_resistance_bits(&self) -> usize` to `Mmcs` and checking it in `BinaryPcs::new` would close it, and ripples to every MMCS implementor |
| `accept` must be able to admit `count` candidates, or `challenge_uniform_bits_rejecting` does not terminate | `challenger/src/fs/state/prover.rs` | `accept` is an opaque closure. A blanket `count <= 2^width` would be wrong for a predicate that admits repeats. The in-tree caller that uses a distinctness predicate, `binary-pcs`, derives its count through `num_distinct_queries`, which caps it at the domain, so the obligation is discharged by derivation there rather than by a check here |

## Minimum non-degenerate shapes

A reduction with no rounds samples no challenge. It still returns a point and a value, still reads
like a reduction, and separates nothing — which is how a cross-instance separation test can pass
against every transcript it is given.

Each reduction below either rejects its degenerate shape or says why the shape is harmless. The
two whose floor was previously unwritten — the generic-degree sumcheck and the jagged reduction —
also gained a test at their smallest legal shape that fails if no challenge is sampled.

| Reduction | Round count from | Minimum non-degenerate shape | Below it |
| --- | --- | --- | --- |
| `GenericDegreeProof::verify` (`sumcheck/src/generic_degree/proof.rs`) | caller's `num_rounds` | 1 round | **Rejected**, `GenericDegreeError::NoRounds`. At zero rounds the surviving value is the prover's own `claimed_sum`, echoed back at an empty point |
| `ZkVerifier::verify` (`sumcheck/src/zk/`) | `folding_factor` | 1 round | **Rejected**, `SumcheckError::NoRounds`, in `ZkSumcheckShape::validate` |
| `JaggedLayout::verify` (`sumcheck/src/jagged/layout.rs`) | `dense_variables()`, derived from the live area | live area 2 | **Allowed and harmless.** A one-cell dense multilinear has no interior to test; the terminal relation's two factors are public or pinned by the commitment, and the returned `JaggedDenseClaim` is `#[must_use]`, which is what rules out the vanishing-selector case |
| `BinaryPcs::verify_at` opening protocol (`binary-pcs/src/pcs.rs`) | claim count | 1 claim | **Allowed and harmless.** A run that claims nothing proves nothing and asserts nothing. The shape is load-bearing in `security_tests.rs`, where it is the only way to reach query sampling with a tampered final codeword |
| `BinaryWhirProfile::config` (`binary-pcs/src/whir/profile.rs`) | `security_level`, `folding_factor` | both positive | **Rejected**, `ProfileError::ZeroSecurityLevel` / `ZeroFoldingFactor`. A zero target is met by every schedule; a zero folding factor describes a schedule that never terminates |
| `SumcheckData::verify_rounds` (`sumcheck/src/data.rs`) | caller's `expected_rounds` | 1 round | **Allowed by design.** WHIR's closing fold legitimately runs zero rounds and plays no bracket, matching the prover exactly. The surrounding protocol owns the floor |

## Known gaps

These are recorded rather than fixed, and are tracked as follow-ups to
[#2271](https://github.com/Plonky3/Plonky3/issues/2271).

- `PrescribedPointPcs::verify_at` returns `Result<Vec<OpeningEvals<EF>>, _>`. `unused_must_use`
  does not recurse through `Vec`, so `pcs.verify_at(..)?;` discards the claimed column values and
  compiles. A `#[must_use]` newtype, or a consuming `verify_at_expecting(.., expected)`, closes it.
- `challenge_uniform_bits_rejecting` and `challenge_uniform_bits` record the same transcript step,
  so no seed parts two protocols with materially different challenge distributions. The same holds
  for `challenge_extensions` and `challenge_extensions_rejecting`.
- `sumcheck/src/layout/verifier.rs` lets `sum(alpha)` and `constraint(alpha)` take any field
  element, not only the one `batching_challenge` drew, and lets `add_claim` run after the challenge
  is drawn. Making `batching_challenge` consume `self` and hand back a batched-claims value closes
  both at once.
- `RingSwitchShape::sumcheck_rounds`, `BitRingSwitch::prefix_limit` and `SkipDomain::new` each admit
  a zero-round shape. Cross-instance separation survives in all three because the surrounding
  statement still draws a challenge, but the reductions themselves are empty.
