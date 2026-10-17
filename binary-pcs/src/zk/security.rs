//! Security contract for the canonical current-row AIR reference protocol.
//!
//! This module gives protocol-specific derived lemmas, their byte-level
//! implementation boundary, and the checked public budget API. These are new
//! classical-ROM arguments, not published certification of this implementation.
//! Keccak256 instantiates the ideal oracle heuristically. Human review is
//! pending on the PR; no audit or standard-model guarantee is asserted.
//!
//! # Relation, experiments and lifecycle
//!
//! The relation is existence of `N_trace` rows, each satisfying the SAME
//! normalized `C_i(row,p)=0`. Rows may repeat. There is no progression,
//! distinctness or public-record binding unless explicitly constrained within
//! a row. Compilation admits only current-row arithmetic and public values,
//! with explicit Boolean constraints where declared. The restricted
//! [`super::CurrentRowAirBuilder`] accepts compatible generic AIRs; it does not
//! extend zero knowledge to existing clear provers, next-row AIRs, buses,
//! lookups, selectors, preprocessing or public boundary declarations.
//!
//! Soundness/knowledge below concern one final canonical AIR proof, with
//! adaptively chosen public statement and at most `Q_adv` primitive classical
//! RO calls, extended by the final verifier's calls. Privacy concerns adaptive
//! valid-witness proof requests and one GLOBAL cumulative leaf history. Every
//! invocation is atomic: no private intermediate state is exposed. Masks and
//! salts are fresh independent ideal values. Replacing the ENTIRE actual
//! master/child CSPRNG lifecycle by those values adds `Delta_rng`, which this
//! API does not quantify. `CryptoRng` alone cannot bound it or enforce freshness.
//! Never clone, reseed, reset or replay private RNG state, including rollback.
//! Each consumed commitment state permits one opening at one common point;
//! fresh re-proving is permitted. Two openings of a reused `h,u` under distinct
//! gamma reveal `h=(g1-g2)/(gamma1-gamma2)`, including all low coefficients.
//!
//! The scope excludes private-state compromise, side channels, concurrent
//! exposure of prover internals, RNG rollback, QROM, simulation extractability
//! and security after simulated proofs. Local simulation below is for the
//! prescribed paired query schedule, not a malicious-verifier IOP allowing
//! arbitrary additional unpaired word reads. Generic external PCS challenger
//! contexts are outside the canonical AIR compiler theorem.
//!
//! # Exact v2 bytes and canonical analysis grammar
//!
//! Every scalar AND batched hash uses one of the disjoint ASCII byte tags
//! `p3-binary-zk-v2/leaf\0`, `p3-binary-zk-v2/node\0`,
//! `p3-binary-zk-v2/fiat-shamir\0`; `\0` denotes one NUL byte. A leaf input is
//! `LEAF_TAG || repr128LE(row[0]) || ... || repr128LE(row[B-1]) ||
//! repr128LE(salt[0]) || repr128LE(salt[1])`. A node input is
//! `NODE_TAG || left_digest32 || right_digest32`. A batched message counts as
//! one primitive RO evaluation, even if its implementation hashes many in one
//! call. Field representation is the binary tower's `to_repr()` u128 LE.
//!
//! Concatenation of records uses these injective typed encodings (every u64
//! and u32 is little endian; every length is checked before encoding):
//!
//! ```text
//! F(x)   = 1 || repr128LE(x)
//! C(rs)  = 2 || u64(root_count) || digest32(root[0]) || ...
//! B(xs)  = 3 || u64(byte_length) || xs
//! U(x)   = 4 || u64(x)
//! D(l,r) = 5 || u64(label_length) || label || u64(output_length) || raw_output
//! ```
//!
//! Expansion block j hashes exactly
//! `FS_TAG || u64(label_length) || label || u64(transcript_length) ||
//! complete_prefix || u64(draw_ordinal) || u64(output_length) || u32(j)`.
//! Blocks start at zero, use the same unchanged complete prefix, concatenate,
//! and truncate to output length. Ordinals start at zero. Append the COMPLETE
//! fixed raw draw as D only after all blocks; increment the ordinal once.
//! There is no prover message between blocks. The 256-byte `air-point` and
//! `blend-scalar` streams each contain sixteen 128-bit LE candidates. Select
//! the FIRST outside the trace subspace or the FIRST nonzero, respectively;
//! abort if none exists. Keep unused candidates in D. Query label is exactly
//! `query-indices || u64(q) || u64(log2(n))`, output length `8q`; each LE u64
//! is masked to its low log2(n) bits. This is one ordered q-tuple with replacement.
//!
//! Here is the complete ordered canonical AIR grammar. Superscript repetition
//! means literal repetition of records, never hashing a replacement digest.
//! `AST` is postcard's serialization of the complete normalized constraint
//! vector, including Boolean augmentation, checked variable bounds and
//! recomputed degree annotations; pointer sharing and user caches are removed.
//! `P` is the verifier-owned public-value count, `J` the constraint count,
//! `Bwidth=width+J`, `k=q+1`, `L=M+k` and `n=next_power_of_two(4L)`.
//!
//! ```text
//! Header = B("p3-binary-reference-zk-air-v2")
//!          (U(0)                         // interactive policy
//!           OR U(1) U(target) U(b) U(a)) // classical-ROM policy
//!          U(log_trace_height) U(s) U(width) U(P) U(J)
//!          B(AST) U(P) F(public[0]) ... F(public[P-1])
//! Commit(h) = B("p3-binary-reference-zk-pcs-v2") B("commitment")
//!             U(M) U(q) U(k) U(L) U(n) U(M) U(q) U(Bwidth) U(1) C([h])
//! z-prefix = Header Commit(h)
//! gamma-prefix = z-prefix D("air-point", raw_z) Commit(h)
//!                B("opening-point-and-claims") F(z) U(Bwidth) F(v)^Bwidth
//!                B("uniform-mask-commitment-and-functional") U(1) C([u])
//!                U(Bwidth) F(sigma)^Bwidth
//! query-prefix = gamma-prefix D("blend-scalar", raw_gamma)
//!                B("complete-monomial-blend-before-queries") U(Bwidth)
//!                (U(L) F(g_column[0]) ... F(g_column[L-1]))^Bwidth
//! final-prefix = query-prefix D(query_label, raw_indices)
//! ```
//!
//! z/gamma/query draws have ordinals 0/1/2. z is the first accepted candidate
//! in raw_z; gamma is the first accepted candidate in raw_gamma. There is NO
//! extra F(gamma) record: its use is bound by the low-functional and queried
//! blend equations. h and its geometry are deliberately repeated IDENTICALLY;
//! there are three cap records, of which two name the same h root. u's sole
//! cap has exactly one root too. Neither omitted, alternative-order nor
//! inconsistent duplicate records designate valid protocol states.
//!
//! The proof's analysis parser recognizes only this reachable constant-record
//! grammar, admitted shapes, normalized AST, exact lengths, first-candidate
//! selection and identical duplicate geometry/h roots. At query stage it
//! derives gamma from raw_gamma and checks the low-functional equation with
//! that value. Earlier raw answers can be claimed in an out-of-order query;
//! only prefixes agreeing with the actual oracle can participate in a final
//! proof, and local bounds below hold even for arbitrary claimed earlier
//! accepted challenges. Malformed raw RO inputs do not designate alternate
//! states or tokens. This parser is part of analysis, not a public production
//! hash interface: the production verifier constructs the canonical prefix
//! itself. Test-only snapshots compare actual verifier draws with independently
//! constructed typed records and reject corrupted duplicate roots/selected
//! candidates; no unused production parser or configurable mock hash is needed.
//!
//! Canonical transport starts `P3ZKAIR2`, uses fixed LE geometry, fields, roots,
//! two salts per opened row and verifier-owned exact lengths. It rejects v1.
//! v2 is finalized within this unmerged PR: policy binding deliberately changed
//! its premerge vectors. The fixed-seed fixture's Keccak256 wire digest is
//! `d7b8f032065aae44fd96c625c5658775ddc995842b4a8604c7894e5d0fdb8e2f`;
//! its postcard full-proof digest is
//! `ccff3b673a79613c00be51de1a8e7c16797d103630d44b7d9b2b25babb9ac3e6`.
//! The independently framed raw multiblock vector and literal AST are also
//! pinned in the tests; interactive wire fixture length remains 7488 bytes.
//!
//! # Ideal RBR soundness and knowledge
//!
//! Write `K=2^128`, `N=N_trace`, `H` for the unshifted Cantor trace domain,
//! `W=product_(a in H)(X-a)`, and `delta` for maximum augmented AIR degree.
//! Honest traces are `T=T0+rW` with independent uniform r; each quotient is
//! `Q_i=C_i(T,p)/W`. Set `M=max(N+1,(delta-1)N+1)` with saturating delta-1.
//! Commit trace and quotient messages using `h_j=f_j+X^M R_j`, `deg R_j<q+1`.
//! The shifted code domain consists of n distinct NONZERO points. Independently
//! uniform L-coefficient u gives `sigma=lambda_z(u)`; after nonzero gamma the
//! prover discloses every coefficient of `g=gamma*h+u`. Here lambda evaluates
//! ONLY the low M coefficients. Claims v pass every AIR quotient equation,
//! g passes `lambda_z(g)=gamma*v+sigma`, and both authenticated words satisfy
//! `g(x)=gamma*h(x)+u(x)` at every selected row and in every column.
//!
//! Soundness permits arbitrary committed words h,u, not just codewords. The
//! vector-valued degree-below-L RS code has row minimum distance `n-L+1`:
//! two distinct matrices differ in a scalar nonzero polynomial in some column.
//! Set `e=floor((n-L)/4)`, so `4e<n-L+1`. Define the TOTAL decoder Dec(h) as the
//! unique coefficient matrix within 2e rows, or bottom if none exists. A
//! theoretical polynomial-time algorithm uniquely decodes each scalar column
//! to radius 2e and checks that the UNION of differing rows has size at most
//! 2e. This is an extraction algorithm in the proof, not an implemented
//! verifier or a promise of practical extraction performance.
//!
//! Fix arbitrary h,u,v,sigma,z. Call nonzero gamma good if some degree-below-L
//! g_gamma obeys the low-functional equality and is within e rows of
//! gamma*h+u. If a!=b are both good, `A=(g_a-g_b)/(a-b)` agrees with h outside
//! the union of their two error sets, hence within 2e rows. Uniqueness gives
//! `A=Dec(h)` and linearity gives `lambda_z(A)=v`. Thus when Dec(h) is absent
//! or has a different low functional, at most ONE nonzero gamma is good. This
//! is a joint vector argument; there is no union bound over columns.
//!
//! The RBR state argument is as follows. Initially every state is doomed for
//! knowledge. After h,z it is non-doomed when A exists and every residual
//! `R_i=C_i(Low(A)_trace,p)-W*Low(A)_quotient,i` vanishes at z. If any residual
//! is nonzero, escape probability is at most `D/(K-N)`, with
//! `D=max(delta*(M-1),N+M-1)` using MALICIOUS committed degree M-1. If all are
//! identically zero, evaluating Low(A)'s trace polynomials on H extracts a
//! valid witness, since W vanishes there. No candidate means zero escape.
//! From a doomed state, claims passing the AIR equations cannot be lambda(A).
//! After v,u,sigma,gamma label a state non-doomed if gamma is good; the preceding
//! lemma bounds escape by `1/(K-1)`. Retain non-doomed labels thereafter. From
//! a still-doomed state, any supplied g passing its low-functional check differs
//! on at least e+1 rows, and the entire query tuple misses them with probability
//! at most `(1-(e+1)/n)^q`. Acceptance is non-doomed; failed shape/equation checks
//! remain doomed. This gives RBR soundness AND knowledge error
//!
//! `epsilon_rbr=max(D/(K-N),1/(K-1),(1-(e+1)/n)^q)`.
//!
//! Ordinary interactive error is bounded by the sum of these three terms.
//! Because n>=4L, `(e+1)/n>3/16` and `(13/16)^4<1/2`; q=4(s+2) makes the
//! query term less than `2^-(s+2)`. The checked inequality
//! `D*2^(s+2)<=2^128-N` also bounds the gamma term since D>=1 and N>=2.
//! Both fixed sixteen-candidate samplers are exactly uniform on their accepted
//! sets conditioned on success. Honest completeness abort is at most
//! `(N/K)^16+(1/K)^16 <= 2^-1920+2^-2048`. Aborts decrease acceptance; full
//! finite raw streams, including unused candidates, are public verifier coins.
//!
//! # Direct full-prefix compiler and its own graph bound
//!
//! Extend the adversary's primitive query log by running the deterministic
//! final verifier. Count repeated queries conservatively. An upper bound is
//! `V_max=2*q*(log2(n)+1)+16+ceil(q/4)`: two salted leaves and their paths per
//! row query, eight z blocks, eight gamma blocks, and ceil(q/4) index blocks.
//! Let `T=Q_adv+V_max`. Each fresh valid node query registers its two child
//! digests BEFORE its answer. Each canonical challenge query registers at most
//! three cap-root tokens, including duplicate h. Register both final proof
//! roots at verification start. BadGraph is a fresh valid leaf/node answer
//! equal to an earlier distinct leaf/node answer OR any already registered
//! token, including its own input's tokens. Repeated queries are not fresh.
//! At primitive call t there are at most t-1 earlier answers and 3t+2 tokens;
//! a fresh leaf/node reply is conditionally uniform. Summing `(4t+1)/2^256`
//! for t<=T gives a bound below `8*(T+1)^2/2^256`. This intentionally loose
//! constant 8 is derived here; it is NOT the BCS theorem's constant 3.
//!
//! At a root's first designation traverse the PREVIOUS leaf/node graph at the
//! verifier-owned depth/width. Complete absent digests, wrong domains, or
//! wrong-width leaves by a fixed zero row. Outside BadGraph the map from
//! outputs to inputs is unambiguous. Every edge leads to an already recorded
//! answer or a token that can never later get a fresh preimage. Thus every
//! later accepted opening equals the completed word at that position; a
//! missing root cannot later acquire an accepted path. Repeated root uses
//! have the same completed word for that geometry. Role separation prevents
//! a leaf from masquerading as an internal node. This also covers openings
//! whose hashes were never queried by the adversary, because verifier calls
//! are included in T. Completing and uniquely decoding is polynomial time.
//!
//! For each canonical challenge prefix, sample/cache its entire finite coin
//! vector on the FIRST queried block, even if that block has a nonzero counter.
//! Independent blocks make this distribution identical to lazy sampling;
//! unqueried conceptual blocks are not extra primitive calls. Full literal
//! prefixes fix roots and messages before replies. Out-of-order prefixes have
//! arbitrary claimed earlier challenges but the same local lemmas; no parent
//! hash-chain guessing term is needed. In the pre-reply graph define escapes:
//! z when the completed h decodes to an invalid witness but all residuals
//! vanish; gamma when accepted AIR claims disagree with Dec(h), or decoding
//! fails, but sampled gamma is good; indices when a low-functional-valid g
//! differs on >e rows but all sampled indices miss them. Their conditional
//! probabilities are bounded by the three RBR terms, BEFORE conditioning on
//! absence of BadGraph. At most T fresh prefix trials occur across all stages,
//! so the adaptive union bound is T times their MAXIMUM, not their product
//! and not q separately grindable rounds.
//!
//! Outside BadGraph words remain stable. An accepting proof without an
//! extracted witness either has decoded invalid residuals vanishing at z,
//! disagrees with its decoded claims and has good gamma, or has nongood gamma
//! but all queried rows pass: one of the above escapes occurred. A straight-line
//! theoretical extractor records the RO log plus verification, completes h,
//! uniquely decodes and evaluates its low trace on H, returning the trace when
//! valid. This proves, for the canonical AIR grammar only,
//!
//! `Pr[accept AND extraction fails] <= T*epsilon_rbr+8*(T+1)^2/2^256`.
//!
//! False-statement soundness follows. Adaptive statement choice is covered
//! because each canonical prefix binds the entire statement literally and the
//! local estimates hold for every fixed admitted prefix. There is no practical
//! extractor-performance claim, proving-oracle security, or simulation
//! extractability assertion.
//!
//! # Joint privacy, simulator and global salt coupling
//!
//! For column j, condition on z,v,sigma and fixed nonzero gamma. The map from
//! `u_j` with `lambda_z(u_j)=sigma_j` to `g_j=gamma*h_j+u_j` is a bijection
//! onto `lambda_z(g_j)=gamma*v_j+sigma_j`, since lambda's constant coefficient
//! is one. Thus the ENTIRE g is uniform on a product of affine spaces and
//! independent of all tails R_j. Gamma may itself depend on sigma and public
//! variables independent of tails/u beyond sigma. Shared gamma introduces no
//! conditional correlation between independent per-column tails.
//!
//! At d distinct queried nonzero points x_a, the tail matrix entries are
//! `x_a^(M+r)`, 0<=r<q+1. Its first d columns are a nonsingular diagonal times
//! a d-by-d Vandermonde matrix. Hence it has rank d for d<=q+1. Conditional on
//! the entire g,sigma,gamma,v, queried h rows are jointly uniform over F^(dB),
//! irrespective of the witness; u rows are forced by g-gamma*h. Duplicates
//! reuse values AND salts. Since query positions depend only on public g and
//! the other conditioned data, this covers positions selected after g. The
//! q+1 tails leave one extra degree of freedom but the salt bound below does
//! not credit extra algebraic entropy. For AIR, ideal leaf digests make h's
//! root independent of trace masks, so even root-derived z outside H yields
//! jointly uniform T(z); Q(z)=C(T(z),p)/W(z) is determined by public data.
//!
//! A public-statement-only programmable-ROM simulator chooses n independent
//! h leaf digests, computes their node tree honestly and obtains z from the
//! ACTUAL FS oracle. It samples uniform trace claims and computes quotient
//! claims; chooses an independent random-digest u tree and uniform sigma;
//! derives actual gamma; chooses each full g uniformly subject to its low
//! equation (sample all coefficients except the constant and solve for it);
//! derives actual query indices; samples uniform h rows at distinct positions
//! and derives u rows; chooses private random salts and programs only the
//! opened leaf inputs to their preselected digests. Cache duplicates and
//! return actual stored authentication paths. Abort on inconsistent previous
//! leaf answers. FS is NEVER programmed; nodes are evaluated honestly.
//!
//! A hybrid can complete the unqueried witness-dependent rows with a uniform
//! solution of the full-rank tail system, leaving q+1-d free coefficients per
//! column, and set u=g-gamma*h. The affine bijection proves this completion
//! has exactly the honest conditional joint distribution, across both trees,
//! correlated quotients, all columns, full g, sigma and duplicates. The
//! simulator needs no witness or unqueried row completion; its work is
//! polynomial in the admitted instance size and number of proof requests.
//!
//! Let N_leaf be the GLOBAL cumulative number of honest salted slots and
//! Q_leaf all adversary leaf queries, including precomputation, between and
//! after proofs; Q_leaf<=Q_adv conservatively. Let S=2^256 and N_leaf+Q_leaf<S.
//! First couple IID salts to sampling without replacement across ALL slots;
//! distance is at most `C(N_leaf,2)/S`. This is a proof hybrid, not a runtime
//! salt registry. It prevents equal leaf inputs even with equal rows. Replace
//! leaf outputs by independent random digests at commitment time and program
//! opened inputs. Real and simulated games couple until an adversary queries
//! an unprogrammed committed input. Hidden salts never occur in node/FS
//! inputs; conditioned on disclosed salts and failed guesses, each hidden
//! salt has at least S-N_leaf-Q_leaf possibilities. One leaf query specifies
//! at most one candidate salt per slot. Union-bound over slot/query pairs,
//! including queries BEFORE slot creation, to obtain
//!
//! `Adv_ZK <= Delta_rng + C(N_leaf,2)/2^256
//!                     + N_leaf*Q_leaf/(2^256-N_leaf-Q_leaf)`.
//!
//! Already opened inputs return their programmed answers and are harmless.
//! This coupling covers programming failures and adaptive subsequent statements.
//! All honest salted slots sharing the leaf tag count, including standalone
//! commitments, other configurations/sessions and failed/aborted attempts
//! after salt creation. Exceeding the cap later invalidates the lifetime claim.
//! There is no additional digest-birthday term in THIS privacy argument:
//! node hashes are actual oracle computations in both games. The separate
//! binding/knowledge theorem accounts for graph collisions.
//!
//! # Checked parameter rule and review boundary
//!
//! [`super::ZkAirConfig::interactive`] accepts raw s in 1..=120, subject to
//! shape/degree/resources. [`super::ZkAirConfig::classical_rom`] takes
//! [`ClassicalRomBudget`]: target t in 1..=96, Q_adv<=2^b with b in 0..=64,
//! GLOBAL N_leaf<=2^a with a in 0..=120. These are external assumptions,
//! not observed enforcement. Begin c=b; choose s=t+c and q=4(s+2), check PCS
//! geometry and combined batch caps, compute exact u128 T=2^b+V_max, and replace
//! c by ceil(log2(T)) if larger. Repeat at most 128 times, rejecting overflow,
//! s>120, resource caps or failure to stabilize. Intermediate cap failures
//! reject because final q can only increase. ceil(log2(1))=0. Check
//! `D*2^(s+2)<=2^128-N` with checked arithmetic. Require `251-2c>=t+2`, since
//! T<=2^c and T+1<=2^(c+1) bound the graph term. Require `257-2a>=t+2` and
//! `255-a-b>=t+2` for salt collision and leaf-query terms: a<=120,b<=64 imply
//! N_leaf+Q_leaf<2^255, so the latter denominator exceeds 2^255. Finally require
//! 2^a>=2n for at least one complete proof. No float security decisions or
//! overflowing 256-bit arithmetic are used. All subtractions are guarded.
//!
//! Each soundness component is at most 2^-(t+2); ideal privacy has two separate
//! such components (the salt-collision bound is strict). They are DIFFERENT
//! probability guarantees, not generic target-bit work security. A real
//! target-bit privacy assertion requires separately bounding Delta_rng to fit
//! the remaining margin. For t/b/a=80/32/49, c=33,s=113,q=460; a 96-bit target
//! at b=32 fails this field's degree check. [`SecurityReport`] exposes geometry,
//! declared budgets, exact T/V_max and conservative component exponents.
//!
//! Review should cross-check assumptions here, exact grammar above and the
//! raw-draw/AST/full-proof/wire conformance tests together. Changes to roles,
//! roots, geometry, sampler lengths, batching or randomness lifecycle require
//! revisiting the proofs and call counts. The polynomial-size full g and
//! O(n*L*B) Horner encoder are reference costs; a hiding BaseFold replacement
//! would be a different protocol needing its own reduction.
//!
//! Primary context: [Ben-Sasson, Chiesa, Spooner, Interactive Oracle Proofs
//! (2016)](https://eprint.iacr.org/2016/116), Remark 2.1, Section 3.2 and Lemmas
//! 3.4/7.5 discuss counter expansion, salted commitments and compilation. Its
//! statistical salt parameters and transform do not certify this variant.
//! [Block, Garreta, Katz, Thaler, Tiwari, Zając, Fiat-Shamir Security of FRI and
//! Related SNARKs](https://eprint.iacr.org/2023/1071), Definitions 3.12–3.14 and
//! Theorem 3.15 give RBR/BCS machinery; its constant 3 is not invoked here.
//! [Holmgren, On Round-By-Round Soundness and State Restoration Attacks
//! (2019)](https://eprint.iacr.org/2019/1261) supplies RBR context. The decoder,
//! vector lemma, joint simulation/coupling and constant-8 direct graph proof
//! above are protocol-specific derivations rather than claims quoted from them.

use super::{Challenger, ReferencePcsConfig, ZkError};

/// Externally assumed classical adversary and deployment-wide leaf budgets.
///
/// `Q_adv <= 2^log_adversary_queries` counts primitive RO calls across every
/// hash role (each `hash_many` message is one call). `N_leaf <=
/// 2^log_total_salted_leaves` counts ALL cumulative honest salted leaf slots
/// sharing the v2 leaf domain: both trees, every configuration/session,
/// standalone commitment, and aborted/failed attempts after leaf creation.
/// The proof claim ceases to apply if that global history later exceeds the
/// cap. Neither budget is observable or enforced by this API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassicalRomBudget {
    target_bits: usize,
    log_adversary_queries: usize,
    log_total_salted_leaves: usize,
}

impl ClassicalRomBudget {
    /// Validate target 1..=96, adversary-call log 0..=64 and global-leaf log
    /// 0..=120. Shape-dependent feasibility is checked by AIR compilation.
    pub fn new(
        target_bits: usize,
        log_adversary_queries: usize,
        log_total_salted_leaves: usize,
    ) -> Result<Self, ZkError> {
        if !(1..=96).contains(&target_bits)
            || log_adversary_queries > 64
            || log_total_salted_leaves > 120
        {
            return Err(ZkError::InvalidConfig("classical ROM budget out of range"));
        }
        Ok(Self {
            target_bits,
            log_adversary_queries,
            log_total_salted_leaves,
        })
    }

    /// Separate acceptance/extraction-failure and ideal privacy targets, each
    /// at most `2^-target_bits` under this module's different experiments.
    /// `Delta_rng` must be separately bounded before a real privacy claim.
    pub const fn target_bits(&self) -> usize {
        self.target_bits
    }
    /// Base-two logarithm of the externally assumed adversarial primitive RO
    /// call bound, across leaf, node and Fiat–Shamir domains.
    pub const fn log_adversary_queries(&self) -> usize {
        self.log_adversary_queries
    }
    /// Base-two logarithm of the externally assumed GLOBAL cumulative honest
    /// salted-leaf-slot cap across all uses of the v2 leaf domain.
    pub const fn log_total_salted_leaves(&self) -> usize {
        self.log_total_salted_leaves
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum SecurityPolicy {
    Interactive(usize),
    ClassicalRom(ClassicalRomBudget),
}

impl SecurityPolicy {
    pub(super) fn observe(self, transcript: &mut Challenger) {
        match self {
            Self::Interactive(_) => transcript.observe_usize(0),
            Self::ClassicalRom(budget) => {
                transcript.observe_usize(1);
                transcript.observe_usize(budget.target_bits);
                transcript.observe_usize(budget.log_adversary_queries);
                transcript.observe_usize(budget.log_total_salted_leaves);
            }
        }
    }
}

/// Public checked geometry and interactive budget. No witness data or audit
/// assertion is included. The interactive error is at most `3*2^-(s+2)`;
/// the round-by-round maximum is at most `2^-(s+2)`, where `s=interactive_bits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SecurityReport {
    /// Number of trace rows, N_trace.
    pub trace_height: usize,
    /// Number of private trace columns.
    pub width: usize,
    /// Number of normalized constraints, including added Boolean constraints.
    pub constraints: usize,
    /// Number of committed trace and quotient columns, B.
    pub batch_width: usize,
    /// Low monomial message length, M.
    pub message_len: usize,
    /// Masked coefficient length, L=M+q+1.
    pub coefficient_len: usize,
    /// Nonzero shifted Cantor codeword domain size, n.
    pub codeword_len: usize,
    /// Independent row queries with replacement, q.
    pub num_queries: usize,
    /// Committed-message residual degree bound, D.
    pub degree_bound: usize,
    /// Requested or derived interactive algebra/query target, s.
    pub interactive_bits: usize,
    /// Independent ideal salt bits per leaf (actual RNG adds Delta_rng).
    pub salt_bits: usize,
    /// Ideal random-oracle output bits; Keccak is its concrete instantiation.
    pub hash_bits: usize,
    /// Checked classical-ROM budget and exponents, if that policy was selected.
    pub classical_rom: Option<ClassicalRomReport>,
}

/// Separate conservative component bounds `<= 2^-exponent` in the ideal
/// classical ROM. The salt-collision inequality is strict. Acceptance with
/// failed extraction is bounded by RBR plus graph; ideal privacy advantage
/// by salt collision plus leaf query. Real privacy adds `Delta_rng` for the
/// entire fresh master/child CSPRNG lifecycle. See the module contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassicalRomReport {
    /// Declared external budgets, bound in the canonical AIR transcript.
    pub budget: ClassicalRomBudget,
    /// Q_adv = 2^b primitive adversary calls, across all roles.
    pub adversary_queries: u128,
    /// V_max = 2*q*(log2(n)+1)+16+ceil(q/4) final-verifier calls.
    pub verifier_queries: u128,
    /// T = Q_adv + V_max (not just the adversary's query budget).
    pub total_queries: u128,
    /// c = ceil(log2(T)), computed by monotone fixed-point iteration.
    pub log_total_queries: usize,
    /// GLOBAL cumulative salted-leaf-slot budget, 2^a.
    pub total_salted_leaves: u128,
    /// Exponent for T*epsilon_rbr, s+2-c.
    pub rbr_exponent: usize,
    /// Exponent for 8*(T+1)^2/2^256, conservatively 251-2c.
    pub graph_exponent: usize,
    /// Exponent for C(N_leaf,2)/2^256, conservatively 257-2a.
    pub salt_collision_exponent: usize,
    /// Exponent for N_leaf*Q_leaf/(2^256-N_leaf-Q_leaf), 255-a-b.
    pub leaf_query_exponent: usize,
}

fn field_check(n: usize, d: usize, s: usize) -> Result<(), ZkError> {
    let denominator = u128::MAX - (n as u128 - 1); // 2^128 - N_trace
    let factor = u32::try_from(s + 2)
        .ok()
        .and_then(|shift| 1u128.checked_shl(shift));
    if factor
        .and_then(|v| (d as u128).checked_mul(v))
        .is_none_or(|v| v > denominator)
    {
        return Err(ZkError::InvalidConfig(
            "committed-degree AIR soundness bound exceeds target",
        ));
    }
    Ok(())
}

const fn ceil_log2(value: u128) -> usize {
    // Callers supply T >= 1; ceil_log2(1) = 0.
    (u128::BITS - (value - 1).leading_zeros()) as usize
}

pub(super) fn compile_security(
    policy: SecurityPolicy,
    n_trace: usize,
    width: usize,
    constraints: usize,
    message_len: usize,
    d: usize,
) -> Result<(ReferencePcsConfig, SecurityReport), ZkError> {
    let batch_width = width + constraints;
    let make_pcs = |s| {
        field_check(n_trace, d, s)?;
        if s > 120 {
            return Err(ZkError::InvalidConfig(
                "derived interactive target out of range",
            ));
        }
        let pcs = ReferencePcsConfig::new(message_len, 4 * (s + 2))?;
        pcs.check_width(batch_width)?;
        Ok(pcs)
    };
    let (pcs, s, rom) = match policy {
        SecurityPolicy::Interactive(s) => (make_pcs(s)?, s, None),
        SecurityPolicy::ClassicalRom(budget) => {
            let t = budget.target_bits;
            let b = budget.log_adversary_queries;
            let a = budget.log_total_salted_leaves;
            let adversary_queries = 1u128 << b;
            let mut c = b;
            let mut fixed = None;
            for _ in 0..128 {
                let s = t + c;
                let pcs = make_pcs(s)?;
                let q = pcs.num_queries() as u128;
                let verifier_queries = q
                    .checked_mul(2 * (u128::from(pcs.codeword_len().ilog2()) + 1))
                    .and_then(|v| v.checked_add(16 + q.div_ceil(4)))
                    .ok_or(ZkError::InvalidConfig("verifier query budget overflow"))?;
                let total_queries = adversary_queries
                    .checked_add(verifier_queries)
                    .ok_or(ZkError::InvalidConfig("total query budget overflow"))?;
                let new_c = ceil_log2(total_queries);
                if new_c > c {
                    c = new_c;
                    continue;
                }
                let graph_exponent = 251usize
                    .checked_sub(2 * c)
                    .ok_or(ZkError::InvalidConfig("graph budget exceeds hash size"))?;
                let salt_collision_exponent = 257usize.checked_sub(2 * a).ok_or(
                    ZkError::InvalidConfig("salt collision budget exceeds salt size"),
                )?;
                let leaf_query_exponent = 255usize.checked_sub(a + b).ok_or(
                    ZkError::InvalidConfig("leaf query budget exceeds salt size"),
                )?;
                if [graph_exponent, salt_collision_exponent, leaf_query_exponent]
                    .into_iter()
                    .any(|e| e < t + 2)
                {
                    return Err(ZkError::InvalidConfig(
                        "classical ROM component exceeds target",
                    ));
                }
                let total_salted_leaves = 1u128 << a;
                if total_salted_leaves < 2 * pcs.codeword_len() as u128 {
                    return Err(ZkError::InvalidConfig(
                        "global leaf budget cannot cover one proof",
                    ));
                }
                fixed = Some((
                    pcs,
                    s,
                    Some(ClassicalRomReport {
                        budget,
                        adversary_queries,
                        verifier_queries,
                        total_queries,
                        log_total_queries: c,
                        total_salted_leaves,
                        rbr_exponent: s + 2 - c,
                        graph_exponent,
                        salt_collision_exponent,
                        leaf_query_exponent,
                    }),
                ));
                break;
            }
            fixed.ok_or(ZkError::InvalidConfig(
                "classical ROM budget has no fixed point",
            ))?
        }
    };
    let report = SecurityReport {
        trace_height: n_trace,
        width,
        constraints,
        batch_width,
        message_len,
        coefficient_len: pcs.coefficient_len(),
        codeword_len: pcs.codeword_len(),
        num_queries: pcs.num_queries(),
        degree_bound: d,
        interactive_bits: s,
        salt_bits: 256,
        hash_bits: 256,
        classical_rom: rom,
    };
    Ok((pcs, report))
}
