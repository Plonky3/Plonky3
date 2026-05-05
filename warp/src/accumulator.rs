//! Accumulator and proof types.
//!
//! After every WARP step the prover holds an [`Accumulator`] consisting
//! of a verifier-visible [`AccumulatorInstance`] and a prover-only
//! [`AccumulatorWitness`]. The next step takes some number of fresh
//! PESAT witnesses + zero or more prior accumulators, and produces
//! a new `Accumulator` along with a [`WarpProof`] checkable by the
//! verifier (without any access to the witness side).

use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::sumcheck::SumcheckProof;

/// Verifier-visible accumulator state, mirroring `acc.x = (rt, α, µ, β, η)`
/// from Construction 10.4.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    bound = "EF: Serialize + serde::de::DeserializeOwned, Comm: Serialize + serde::de::DeserializeOwned"
)]
pub struct AccumulatorInstance<EF, Comm> {
    /// Merkle commitment of the merged codeword `f` over `EF`.
    pub rt: Comm,
    /// §8.2 sumcheck challenges (length `log n`).
    pub alpha: Vec<EF>,
    /// `µ = f̂(α)`.
    pub mu: EF,
    /// §6.3 final folded `β = B̂(γ)` (length `log M + κ`).
    pub beta: Vec<EF>,
    /// `η = Pb(β, w)`.
    pub eta: EF,
}

/// Prover-only accumulator state, mirroring `acc.w = (td, f, w)`.
///
/// `td` is the Merkle prover data needed to reopen the merged `f` at
/// shift-query positions in subsequent accumulation steps.
pub struct AccumulatorWitness<EF, ProverData> {
    pub td: ProverData,
    /// The merged codeword `f ∈ EF^n`.
    pub f: Vec<EF>,
    /// The merged witness `w ∈ EF^k`.
    pub w: Vec<EF>,
}

/// Bundled accumulator state: instance + witness.
pub struct Accumulator<EF, Comm, ProverData> {
    pub instance: AccumulatorInstance<EF, Comm>,
    pub witness: AccumulatorWitness<EF, ProverData>,
}

/// One-step WARP proof.
///
/// The naming mirrors the Construction 10.4 transcript:
///
/// - `rt_0`: Merkle root of the stacked fresh codewords (length `ℓ_1`).
/// - `mu_fresh`: `µ_i = f̂_i(0^{log n})` for the `ℓ_1` fresh instances.
/// - `twin_constraint_sumcheck`: §6.3 sumcheck (length `log ℓ`, degree
///   `1 + max{log n + 1, log M + d}`).
/// - `nu_0`: `f̂(ζ_0)` after the §6.3 fold, where `ζ_0 = Â(γ)`.
/// - `eta`: `η = Pb(β, w)` after the §6.3 fold.
/// - `nu_ood`: out-of-domain answers `f̂(ζ_k)` for `k ∈ [s]`.
/// - `batching_sumcheck`: §8.2 sumcheck (length `log n`, degree 2).
/// - `mu_final`: final folded value `µ = f̂(α)`.
/// - `fresh_shift_answers[k]`: the `ℓ_1` base-field values at shift index
///   `k` (one per stacked fresh codeword).
/// - `fresh_merkle_proofs[k]`: the Merkle proof for the `k`-th shift index
///   on `rt_0`.
/// - `acc_shift_answers[j][k]`: the EF value of prior accumulator `j`'s
///   merged codeword at shift index `k` (a single-element `Vec<EF>` per
///   opening, since the prior acc commitment has width 1).
/// - `acc_merkle_proofs[j][k]`: the Merkle proof for the `k`-th shift
///   index on prior accumulator `j`'s `rt`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned,
                  Comm: Serialize + serde::de::DeserializeOwned,
                  Proof: Serialize + serde::de::DeserializeOwned")]
pub struct WarpProof<F, EF, Comm, Proof> {
    pub rt_0: Comm,
    pub mu_fresh: Vec<EF>,
    pub twin_constraint_sumcheck: SumcheckProof<EF>,
    pub nu_0: EF,
    pub eta: EF,
    pub nu_ood: Vec<EF>,
    pub batching_sumcheck: SumcheckProof<EF>,
    pub mu_final: EF,
    pub fresh_shift_answers: Vec<Vec<F>>,
    pub fresh_merkle_proofs: Vec<Proof>,
    pub acc_shift_answers: Vec<Vec<Vec<EF>>>,
    pub acc_merkle_proofs: Vec<Vec<Proof>>,
}

/// One-step WARP proof for the **alphabet-`F` variant** of Construction
/// 10.4 (WARP paper, Theorem 10.3 proof, lines 3911-3915).
///
/// In this variant the `ℓ_1` fresh codewords are committed
/// **individually** rather than via one stacked alphabet-`F^{ℓ_1}` tree.
/// The verifier consumes the `ℓ_1` external commitments separately
/// (e.g., precommitted segment roots), so this proof
/// **omits `rt_0`** and instead carries one Merkle path **per `(shift,
/// fresh)` pair**.
///
/// Compared to [`WarpProof`]:
/// - `rt_0` removed (verifier holds the commitments).
/// - `fresh_merkle_proofs: Vec<Vec<Proof>>` instead of `Vec<Proof>` —
///   indexed `[shift_idx][fresh_idx]`.
/// - All other fields are identical.
///
/// The corresponding prover entry point is
/// [`WarpProver::prove_with_committed`](crate::protocol::WarpProver::prove_with_committed)
/// and verifier entry point is
/// [`WarpVerifier::verify_with_committed`](crate::protocol::WarpVerifier::verify_with_committed).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned,
                  Comm: Serialize + serde::de::DeserializeOwned,
                  Proof: Serialize + serde::de::DeserializeOwned")]
pub struct WarpProofCommitted<F, EF, Comm, Proof> {
    /// `µ_i = f̂_i(0^{log n})` for the `ℓ_1` fresh instances.
    pub mu_fresh: Vec<EF>,
    pub twin_constraint_sumcheck: SumcheckProof<EF>,
    pub nu_0: EF,
    pub eta: EF,
    pub nu_ood: Vec<EF>,
    pub batching_sumcheck: SumcheckProof<EF>,
    pub mu_final: EF,
    /// Per `(shift index, fresh index)`: one base-field value, opened
    /// against the corresponding fresh codeword's external commitment.
    /// Indexed as `[shift_idx][fresh_idx]`.
    pub fresh_shift_answers: Vec<Vec<F>>,
    /// Per `(shift index, fresh index)`: one Merkle authentication path
    /// against the corresponding fresh codeword's commitment.
    /// Indexed as `[shift_idx][fresh_idx]`.
    pub fresh_merkle_proofs: Vec<Vec<Proof>>,
    /// Same as `WarpProof`: prior-accumulator shift answers and paths.
    pub acc_shift_answers: Vec<Vec<Vec<EF>>>,
    pub acc_merkle_proofs: Vec<Vec<Proof>>,
    /// Phantom needed because `Comm` only appears via the absence of
    /// `rt_0` (the type parameter is preserved for symmetry with
    /// [`WarpProof`]).
    #[serde(skip)]
    pub _ph: core::marker::PhantomData<Comm>,
}

/// One-step WARP proof for fresh inputs committed by an external PCS.
///
/// This is the generic form of [`WarpProofCommitted`]. It keeps the WARP
/// accumulator commitment/proofs on the local Plonky3 `Mmcs`, but lets fresh
/// inputs use a different opening proof type. This is needed for precommitted
/// PCS inputs:
/// its initial commitment authenticates row groups of an RS codeword matrix,
/// not a single Plonky3 `Mmcs` row proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned,
                  AccComm: Serialize + serde::de::DeserializeOwned,
                  FreshProof: Serialize + serde::de::DeserializeOwned,
                  AccProof: Serialize + serde::de::DeserializeOwned")]
pub struct WarpProofExternal<F, EF, AccComm, FreshProof, AccProof> {
    /// `µ_i = f̂_i(0^{log n})` for the `ℓ_1` fresh instances.
    pub mu_fresh: Vec<EF>,
    pub twin_constraint_sumcheck: SumcheckProof<EF>,
    pub nu_0: EF,
    pub eta: EF,
    pub nu_ood: Vec<EF>,
    pub batching_sumcheck: SumcheckProof<EF>,
    pub mu_final: EF,
    /// Per `(shift index, fresh index)`: one base-field value, opened
    /// against the corresponding external fresh commitment.
    pub fresh_shift_answers: Vec<Vec<F>>,
    /// Per `(shift index, fresh index)`: one external PCS proof.
    pub fresh_opening_proofs: Vec<Vec<FreshProof>>,
    /// Prior-accumulator shift answers and local Plonky3 `Mmcs` paths.
    pub acc_shift_answers: Vec<Vec<Vec<EF>>>,
    pub acc_merkle_proofs: Vec<Vec<AccProof>>,
    /// Phantom needed because the accumulator commitment appears through the
    /// verifier-visible accumulator, not directly in this proof body.
    #[serde(skip)]
    pub _ph: core::marker::PhantomData<AccComm>,
}

/// One-step WARP proof for externally committed fresh inputs with batched
/// shift openings.
///
/// The algebraic transcript is identical to [`WarpProofExternal`]. The only
/// difference is the PCS opening layout: after WARP samples the `t` shift
/// indices, each fresh codeword is opened once at all `t` Boolean points, and
/// each prior accumulator is opened once at all `t` Boolean points. This is
/// the natural shape for WHIR, whose PCS API can prove many point openings for
/// one committed polynomial in a single proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned,
                  AccComm: Serialize + serde::de::DeserializeOwned,
                  FreshProof: Serialize + serde::de::DeserializeOwned,
                  AccProof: Serialize + serde::de::DeserializeOwned")]
pub struct WarpProofExternalBatched<F, EF, AccComm, FreshProof, AccProof> {
    /// `µ_i = f̂_i(0^{log n})` for the `ℓ_1` fresh instances.
    pub mu_fresh: Vec<EF>,
    pub twin_constraint_sumcheck: SumcheckProof<EF>,
    pub nu_0: EF,
    pub eta: EF,
    pub nu_ood: Vec<EF>,
    pub batching_sumcheck: SumcheckProof<EF>,
    pub mu_final: EF,
    /// Per `(shift index, fresh index)`: one base-field value, opened
    /// against the corresponding external fresh commitment.
    pub fresh_shift_answers: Vec<Vec<F>>,
    /// Per fresh codeword: one external PCS proof for all sampled shifts.
    pub fresh_opening_proofs: Vec<FreshProof>,
    /// Prior-accumulator shift answers, indexed `[prior][shift][0]`.
    pub acc_shift_answers: Vec<Vec<Vec<EF>>>,
    /// Per prior accumulator: one accumulator PCS proof for all sampled shifts.
    pub acc_merkle_proofs: Vec<AccProof>,
    /// Phantom needed because the accumulator commitment appears through the
    /// verifier-visible accumulator, not directly in this proof body.
    #[serde(skip)]
    pub _ph: core::marker::PhantomData<AccComm>,
}
