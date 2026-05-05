//! Transmissible (but linear-size) finalizer that bundles `acc.w` into
//! the proof so any third party can re-run the four [`WarpDecider`] checks.
//!
//! # Trade-off vs. a full WHIR wrap
//!
//! - **Pro**: zero new cryptography, no SNARG dependency, byte-for-byte
//!   re-uses the existing [`WarpDecider`]. The verifier sees the entire
//!   witness (`f`, `w`) in the clear and runs exactly the four decider
//!   identities. Soundness is the same as the local decider's.
//! - **Con**: proof size is `O(n + k)` field elements (the merged
//!   codeword and witness). Not succinct. For `n ≈ 2²⁰` and EF ≈ 16 bytes,
//!   the proof is ≈ 16 MB — fine for offline verification, prohibitive
//!   on-chain.
//!
//! For succinct on-chain verification, this should eventually be replaced by
//! a precommitted WHIR finalizer that proves the same checks against the
//! accumulator's existing Merkle root.
//!
//! # Why we don't include `td` (the Mmcs prover-data)
//!
//! `td` is purely a Merkle authentication-path index for re-opening the
//! committed codeword. The verifier rebuilds the Merkle tree from `f`
//! during the decider's check 1 (`rt = MT.Commit(f)`); `td` is therefore
//! redundant in this finalizer's proof.

use core::marker::PhantomData;

use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::{Deserialize, Serialize};

use crate::accumulator::{AccumulatorInstance, AccumulatorWitness};
use crate::code::ReedSolomonCode;
use crate::error::{DeciderError, FinalizerError};
use crate::protocol::prover::ExtProverData;
use crate::relation::BundledPesat;

use super::Finalizer;

/// Transmissible proof carrying the witness side `acc.w` of a WARP
/// accumulator (sans the Mmcs prover-data, which is redundant).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "EF: Serialize + serde::de::DeserializeOwned")]
pub struct WitnessProof<EF> {
    /// The merged codeword `f ∈ EF^n`.
    pub f: Vec<EF>,
    /// The merged witness `w ∈ EF^k`.
    pub w: Vec<EF>,
}

/// Finalizer that bundles `acc.w` into the proof.
pub struct WitnessFinalizer<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
{
    mmcs: &'a MT,
    code: &'a ReedSolomonCode<F, Dft>,
    pesat: &'a Pesat,
    _ph: PhantomData<EF>,
}

impl<'a, F, EF, MT, Dft, Pesat> WitnessFinalizer<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
{
    pub fn new(mmcs: &'a MT, code: &'a ReedSolomonCode<F, Dft>, pesat: &'a Pesat) -> Self {
        assert_eq!(
            pesat.shape().explicit_len,
            0,
            "p3-warp v1 supports instance-free PESAT only (κ = 0)"
        );
        assert_eq!(
            code.msg_len(),
            pesat.shape().witness_len(),
            "RS message length must equal PESAT witness length"
        );
        Self {
            mmcs,
            code,
            pesat,
            _ph: PhantomData,
        }
    }
}

impl<'a, F, EF, MT, Dft, Pesat> Finalizer<F, EF, MT, ExtProverData<F, EF, MT>>
    for WitnessFinalizer<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
    MT::Commitment: PartialEq,
{
    type Proof = WitnessProof<EF>;

    fn finalize(
        &self,
        _instance: &AccumulatorInstance<EF, MT::Commitment>,
        witness: &AccumulatorWitness<EF, ExtProverData<F, EF, MT>>,
    ) -> Result<Self::Proof, FinalizerError> {
        // Just bundle the witness side. The decider's checks aren't run by
        // the prover here — soundness comes from the verifier re-running
        // them in `verify`.
        Ok(WitnessProof {
            f: witness.f.clone(),
            w: witness.w.clone(),
        })
    }

    fn verify(
        &self,
        instance: &AccumulatorInstance<EF, MT::Commitment>,
        proof: &Self::Proof,
    ) -> Result<(), FinalizerError> {
        // Re-run the four decider checks on the (instance, claimed witness)
        // pair. Identical logic to `WarpDecider::decide`, but acting on the
        // witness from the proof rather than from a held `Accumulator`.
        let n = self.code.codeword_len();
        let k = self.code.msg_len();
        let shape = self.pesat.shape();
        let log_m = shape.log_constraints;

        if proof.f.len() != n || proof.w.len() != k {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }

        // 1. Re-Merkleise f and check the root matches.
        let ext_mmcs = p3_commit::ExtensionMmcs::<F, EF, MT>::new(self.mmcs.clone());
        let f_matrix = RowMajorMatrix::new(proof.f.clone(), 1);
        let (rt_recomputed, _td) = ext_mmcs.commit_matrix(f_matrix);
        if rt_recomputed != instance.rt {
            return Err(FinalizerError::Decider(DeciderError::MerkleRoot));
        }

        // 2. Multilinear extension f̂(α) == µ.
        let f_poly = Poly::<EF>::new(proof.f.clone());
        let alpha_pt = Point::<EF>::new(instance.alpha.clone());
        if f_poly.eval_ext::<F>(&alpha_pt) != instance.mu {
            return Err(FinalizerError::Decider(DeciderError::MlEval));
        }

        // 3. Bundled PESAT Pb(β, w) == η.
        if instance.beta.len() != shape.beta_len() {
            return Err(FinalizerError::Decider(DeciderError::BundledPesat));
        }
        let beta_tau = &instance.beta[..log_m];
        let beta_x = &instance.beta[log_m..];
        let mut z = Vec::with_capacity(beta_x.len() + proof.w.len());
        z.extend_from_slice(beta_x);
        z.extend_from_slice(&proof.w);
        let beta_tau_eq = Poly::<EF>::new_from_point(beta_tau, EF::ONE);
        let eta_recomputed = self.pesat.evaluate_bundled(beta_tau_eq.as_slice(), &z);
        if eta_recomputed != instance.eta {
            return Err(FinalizerError::Decider(DeciderError::BundledPesat));
        }

        // 4. Codeword consistency: C(w) == f.
        let f_recomputed = self.code.encode_algebra::<EF>(&proof.w);
        if f_recomputed != proof.f {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }

        Ok(())
    }
}
