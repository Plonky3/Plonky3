use super::*;

/// PCS proof for the accumulator codeword opening `f_hat(alpha) = mu`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "PcsProof: Serialize + serde::de::DeserializeOwned")]
pub struct WhirAccumulatorOpeningProof<PcsProof> {
    /// Opening proof produced by the underlying multilinear PCS.
    pub pcs_proof: PcsProof,
}

/// WHIR-facing proof of the final PESAT decider claim.
///
/// The final decider equation `Pb(beta, C^{-1}(f)) = eta` is reduced by a
/// sumcheck to one terminal witness claim. In systematic RS mode, `C^{-1}(f)`
/// is the message subspace of the committed codeword.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    bound = "EF: Serialize + serde::de::DeserializeOwned, PcsProof: Serialize + serde::de::DeserializeOwned"
)]
pub struct WhirPesatProof<EF, PcsProof> {
    /// Sumcheck over the PESAT witness hypercube.
    pub decider_sumcheck: SumcheckProof<EF>,
    /// Claimed terminal witness value at the sampled point.
    pub terminal_values: Vec<EF>,
    /// PCS opening proof for terminal values on the systematic RS oracle.
    pub pcs_proof: PcsProof,
}

/// WHIR-facing final WARP proof.
///
/// This is the reusable assembly point for a WHIR-native WARP finalizer. The
/// two subproofs certify the two non-local decider equations against the same
/// accumulator commitment:
///
/// ```text
///     f_hat(alpha) = mu
///     Pb(beta, C^{-1}(f)) = eta
/// ```
///
/// Soundness depends on the `Pcs` commitment being the same public commitment
/// layout as the accumulator's `rt`. A PCS that commits to a fresh unrelated
/// oracle must not be used here.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    bound = "EF: Serialize + serde::de::DeserializeOwned, PcsProof: Serialize + serde::de::DeserializeOwned"
)]
pub struct WhirWarpFinalizerProof<EF, PcsProof> {
    /// Opening proof for `f_hat(alpha) = mu`.
    pub accumulator_opening: WhirAccumulatorOpeningProof<PcsProof>,
    /// PESAT decider proof for `Pb(beta, C^{-1}(f)) = eta`.
    pub pesat: WhirPesatProof<EF, PcsProof>,
}
