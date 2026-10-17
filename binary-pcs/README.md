# p3-binary-pcs

A multilinear polynomial commitment scheme over the binary tower, committing via an
additive-domain Reed–Solomon code (the Cantor domain from `p3-binary-dft`) and proving
proximity BaseFold-style, folding the codeword in lockstep with a multilinear sumcheck. See
Diamond, Posen, *Succinct Arguments over Towers of Binary Fields* (Binius),
<https://eprint.iacr.org/2023/1784>, and Diamond, Posen, *Polylogarithmic Proofs for
Multilinears over Binary Towers* (FRI-Binius, ring switching),
<https://eprint.iacr.org/2024/504>. Parameters are derived only in the unique-decoding regime.
The capacity bound is refuted over characteristic 2 with `F_2`-subspace domains, and the Cantor
domain is one. The classical Johnson list-decoding radius remains unconditional; using it in a
folding argument additionally needs the proven Reed--Solomon mutual-correlated-agreement bound.
This BaseFold implementation nevertheless stays in unique decoding by design; the WHIR adapter
below also accepts the Johnson regime.

The committed alphabet and the challenge field are separate choices. Columns and the base
codeword live in the alphabet; every challenge, every folded codeword and every claimed value
live in the challenge field. A narrower alphabet halves the bytes of the largest Merkle tree in
the proof without moving a single soundness term, because every error is charged against the
challenge field's width. Two constraints follow from the alphabet alone: the grinding witness is
one of its elements, so its width caps the difficulty, and the base codeword lives on its
additive domain, so `num_variables + log_inv_rate` must not exceed its bit width.

Two obligations the types do not carry:

- **Binding, not hiding.** The final codeword travels in the clear and every query opening is a
  raw codeword symbol, so this must not be used where zero-knowledge is required.
- **Collision resistance is the caller's.** The derived schedule prices the field's width and
  the query count, never the Merkle tree it is paired with.

`BinaryPcsConfig` delegates its algebraic budget to `p3_security::binary::BinaryPcsRegime`.
The target covers the sum of opening-claim batching, every fold and sumcheck round, and
query error. Query grinding applies only to queries; it runs after the alpha challenge,
every fold, and absorption of the entire final codeword. This final-codeword binding changes
the Fiat–Shamir transcript: proofs generated without it are not guaranteed to verify, although
the proof's serialization layout is unchanged.
When every base coset is queried, query error is zero and its reserved budget is released.
`PrescribedPointPcs::prescribed_security` supplies this composed bound and the unique-decoding
candidate bound to security-checked multi-STARK callers, which also need collision evidence.

This tightens previously accepted configurations and can change the derived query count
and proof transcript. Opening protocols must also fit `BinaryPcsConfig::max_opening_claims()`;
each current or successor-column evaluation consumes one claim. The limit is conservative,
not a tight attack threshold. Use `BinaryPcs::validate_opening_protocol` to preflight a
protocol, or `try_open` / `try_open_at` to receive typed errors without changing the challenger
on rejection. The existing infallible `open` / `open_at` traits panic on invalid or
over-budget protocols. Existing callers must handle this new rejection or choose a feasible
target and protocol before opening.

## WHIR over the additive domain

The `whir` module connects WHIR to the Cantor additive code. Commitments use
the 64-bit polynomial-basis field. Challenges use its 192-bit cubic extension.

Queries follow a transcript-bound stratified schedule. Use
`recommended_cap_height` to stop Merkle paths at its deepest stratum.

Unique decoding and the Johnson regime use proven bounds. The capacity regime
is unsupported because its assumption is refuted for this domain.

This adapter is binding, not hiding.

## Private current-row AIR reference protocol

The opt-in `zk` feature exposes `p3_binary_pcs::zk`, an independent
reference protocol over `BinaryField128`. It does not change `BinaryPcs`, the WHIR
adapter, or existing multilinear AIR paths into zero-knowledge protocols.

Run the native example with:

```sh
cargo run -p p3-binary-pcs --features zk --example zk_air --release
```

It proves that two private columns have the same public product at every row,
using `StdRng` freshly seeded from operating-system entropy for all private
randomness. It serializes the actual proof, deserializes it, and verifies the
round-trip. Output contains only proof byte length and public dimensions. The
native `zk_air` integration tests also verify distinct witnesses for one public
product; the statement does not reveal a unique factorization.

### Supported statements and lifecycle

`ReferenceZkAir::compile` accepts `Air<CurrentRowAirBuilder>`. Ordinary generic
`impl<AB: AirBuilder<F = Val>> Air<AB>` implementations work; implementations
specific to `SymbolicAirBuilder` must adapt to this restricted builder. It has
no permutation, interaction/bus, or runtime-context capabilities. Compilation
admits only current-row main variables, public values, constants and arithmetic.
Set `BaseAir::main_next_row_columns()` explicitly to an empty vector: its default
allows successor columns. Compilation rejects successor access, preprocessing,
periodic columns, row selectors, public boundary declarations, extension
constraints, and unsupported or out-of-range symbolic leaves. Leading Boolean
column declarations add explicit `x*(x+1)` constraints; inconsistent Boolean
metadata rejects. Lookup and bus arguments are unsupported.

Shape and constraints are public and fixed by the verifier's compiled AIR. The
version, canonical symbolic constraints, dimensions, public values and PCS
geometry are transcript-bound. Trace logarithms are 1 through 8, widths at most
16, augmented constraint counts at most 32, constraint degrees at most 4, and
public-value counts at most 1024. Expressions and PCS allocations/work also have
limits; a requested geometry can reject even within these individual limits.
The configured interactive target is 1 through 96 bits, subject to the degree
bound below. Unsupported AIR metadata and leaves reject during compilation,
before private randomness or transcript mutation.

The lower-level `ReferenceZkPcs` takes columns of **monomial coefficients**, not
multilinear tables or novel-basis coefficients. `commit` returns a public
commitment and private `ZkProverData`. This state is neither `Clone` nor
serializable and is consumed by `open`: each commitment permits one common-point
opening only. Configuration objects are reusable. Use fresh private CSPRNG
randomness for every proof, column mask and commitment; never seed it from the
public challenger. Never clone, reseed, reset, or replay the supplied `CryptoRng`,
including through VM rollback. Single-use commitment state cannot enforce RNG
freshness. Reusing the same `h` and `u` under distinct public `gamma` exposes
`h=(g1+g2)/(gamma1+gamma2)`, including every low message coefficient.
AIR proving manages single-use state internally; callers still own RNG freshness.
The `zk` feature is unaudited and does not certify production security.

### Reference protocol

Let `H` be the unshifted Cantor subspace of size `N`, and
`W(X) = product_{a in H}(X-a)`. For each trace column, interpolate `T0` over `H`
and choose an independent uniform field scalar `r` to form `T = T0 + r*W`.
For each constraint `C_i`, compute the separate, unsplit quotient
`Q_i = C_i(T,p)/W`, rejecting a nonzero remainder. Commit all `T` and `Q_i`
together before deriving one point `z` uniformly outside `H` in the interactive
model. For any fixed such point, the trace openings `t_j = T_j(z)` are jointly
uniform; quotient openings are determined by them and the public statement as
`q_i = C_i(t,p)/W(z)`. Verification checks every quotient identity and the PCS
opening.

For PCS message length `M` and `q` queries, put `kappa=q+1`, `L=M+kappa`, and
`n=next_power_of_two(4L)`. Each message `f` gets an independent uniform
`kappa`-coefficient tail `R`: `h=f+X^M*R`. Encode by Horner evaluation on the
shifted Cantor domain `S_log2(n) + s_log2(n)`, which excludes zero. Here `S_k`
is the span of the first `k` Cantor basis vectors and `s_k` is the next basis
vector. Commit the rows with Keccak256 `MerkleTreeHidingMmcs`, using two
independent full-field salts per leaf. No monomial coefficients are passed to
the novel-basis `AdditiveRsEncoder`.

For a claimed message evaluation `v=f(z)`, define
`lambda(a_low)=sum_{j=0}^{M-1} a_j*z^j`. After binding the configuration,
commitment, point and claims, independently sample a uniform `L`-coefficient
polynomial `u`, commit it with fresh leaf salts, and disclose
`sigma=lambda(u_low)`. Bind these before sampling a uniform nonzero `gamma`.
Disclose all coefficients of `g=gamma*h+u` and check
`lambda(g_low)=gamma*v+sigma`. Absorb the complete `g` before sampling `q`
uniform indices with replacement. Authenticate both committed rows and check
`g(x)=gamma*h(x)+u(x)` at every index, for every column. Columns share `gamma`
and indices, with independent masks. Duplicate indices retain transcript order
and repeat the same authenticated row, adding no privacy rank. The `q+1` tail
coefficients account for the joint observations, including conditioning on the
full blend and `sigma`; the salts protect unopened Merkle leaves.

### Protocol v2 framing and public coins

Leaf, node and Fiat–Shamir hashes respectively prepend the disjoint ASCII tags
`p3-binary-zk-v2/leaf`, `p3-binary-zk-v2/node` and
`p3-binary-zk-v2/fiat-shamir`, each followed by a NUL byte, on every scalar or
batched hash. A leaf has two independent field salts (256 bits total); roots and
path hashes remain 32 bytes. Canonical transport uses `P3ZKAIR2`, two salts per
row and fixed little-endian fields; the v1 wire protocol is rejected.

The protocol-owned challenger retains every public observation and draw output.
Records have distinct one-byte tags: field `1 || repr128LE`; root cap
`2 || count64LE || roots`; bytes `3 || length64LE || bytes`; size
`4 || value64LE`; draw
`5 || label_length64LE || label || output_length64LE || raw_output`.
All lengths, sizes and ordinals are checked fixed-width u64 values.

Each expansion block hashes the Fiat–Shamir tag followed by
`label_length64LE || label || transcript_length64LE || complete_transcript ||
draw_ordinal64LE || output_length64LE || block_counter32LE`. Counters start at
zero; concatenate 32-byte blocks and truncate to the requested length. All blocks
use the same prefix; the draw record is appended and ordinal incremented only
after the complete output is produced.

The `air-point` and `blend-scalar` draws each produce 256 bytes: sixteen
field candidates in tower representation. Select the first outside H or nonzero,
respectively, and return `ZkError::ChallengeExhausted` if all sixteen fail.
Unused public candidates add no prover messages or verifier rounds. All q indices
come from one `8*q`-byte draw whose label is `query-indices || q64LE || logn64LE`;
each little-endian u64 contributes its low logn bits. Generic trait sampling uses
separate `generic-field` and `generic-bits || bits64LE` labels for transcript
continuation, rather than these internal protocol rounds.

### Interactive bound and reference costs

Put `e=floor((n-L)/4)`. In the interactive oracle model the PCS error is at most

```text
1/(2^128-1) + (1-(e+1)/n)^q.
```

Outside at most one exceptional `gamma`, a false low-coefficient claim leaves
more than `e` inconsistent rows. Treating the columns as a vector-valued RS code
gives minimum row distance `n-L+1`, so the exceptional-challenge term does not
need a union bound over columns. Since `n>=4L`, choosing `q=4*(s+2)` makes the
query contribution less than `2^-(s+2)`.

The AIR adds at most `D/(2^128-N)`, where
`D=max(delta*(M-1), N+M-1)` and `delta` is the maximum augmented constraint
degree. This uses the committed message degree bound `M-1`, including malicious
messages, rather than the smaller honest degrees. Compilation chooses
`M=max(N+1, (delta-1)*N+1)` (with saturating subtraction for `delta=0`) and checks
`D*2^(s+2)<=2^128-N`. The target budgets these interactive algebra/query terms.
It excludes hash binding/hiding, salt guessing, and unproved Fiat–Shamir
compilation, random-oracle-query, grinding and simulator-programming losses.
There is no end-to-end production noninteractive security certification.

This implementation sends one full `L`-coefficient blend per committed
polynomial, plus sampled authenticated rows: proof size is polynomial, not
succinct. Horner encoding costs `O(n*L)` field operations per polynomial, and
the interpolation/quotient arithmetic is also a reference implementation.
An optimized hiding BaseFold replacement remains future work; this protocol
does not reuse the existing clear BaseFold reductions as a privacy argument.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
