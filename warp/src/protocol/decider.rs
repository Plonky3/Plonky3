//! WARP decider: re-checks the accumulator against the underlying witness.
//!
//! The decider is the final-stage check that converts the running
//! accumulator into a verdict on the entire history of accumulated
//! PESAT instances. It takes both halves of the accumulator (`acc.x`
//! and `acc.w`) and verifies four identities (Construction 10.4):
//!
//! - `(rt, td) == MT.Commit(f)` — the accumulator's commitment matches
//!   a fresh re-Merkleisation of `f`.
//! - `f̂(α) == µ` — the multilinear extension matches the claimed value.
//! - `Pb(β, w) == η` — the bundled PESAT identity holds.
//! - `f == C(w)` — the encoded witness matches the codeword.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use crate::accumulator::{AccumulatorInstance, AccumulatorWitness};
use crate::code::ReedSolomonCode;
use crate::error::DeciderError;
use crate::relation::BundledPesat;

use super::prover::ExtProverData;

/// WARP decider bound to a specific PESAT, RS code, and Mmcs.
pub struct WarpDecider<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
{
    pub mmcs: &'a MT,
    pub code: &'a ReedSolomonCode<F, Dft>,
    pub pesat: &'a Pesat,
    _ph: PhantomData<EF>,
}

impl<'a, F, EF, MT, Dft, Pesat> WarpDecider<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
    MT::Commitment: PartialEq,
{
    /// Create a decider.
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

    /// Run all four decider checks. Returns `Ok(())` iff every check passes.
    pub fn decide(
        &self,
        instance: &AccumulatorInstance<EF, MT::Commitment>,
        witness: &AccumulatorWitness<EF, ExtProverData<F, EF, MT>>,
    ) -> Result<(), DeciderError> {
        let n = self.code.codeword_len();
        let k = self.code.msg_len();
        let shape = self.pesat.shape();
        let log_m = shape.log_constraints;

        if witness.f.len() != n {
            return Err(DeciderError::EncodingMismatch);
        }
        if witness.w.len() != k {
            return Err(DeciderError::EncodingMismatch);
        }

        // 1. Re-Merkleise f and check the root matches.
        let ext_mmcs = ExtensionMmcs::<F, EF, MT>::new(self.mmcs.clone());
        let f_matrix = RowMajorMatrix::new(witness.f.clone(), 1);
        let (rt_recomputed, _td_recomputed) = ext_mmcs.commit_matrix(f_matrix);
        if rt_recomputed != instance.rt {
            return Err(DeciderError::MerkleRoot);
        }

        // 2. Multilinear extension f̂(α) == µ.
        let f_poly = Poly::<EF>::new(witness.f.clone());
        let alpha_pt = Point::<EF>::new(instance.alpha.clone());
        let mu_recomputed = f_poly.eval_ext::<F>(&alpha_pt);
        if mu_recomputed != instance.mu {
            return Err(DeciderError::MlEval);
        }

        // 3. Bundled PESAT Pb(β, w) == η.
        if instance.beta.len() != shape.beta_len() {
            return Err(DeciderError::BundledPesat);
        }
        let beta_tau = &instance.beta[..log_m];
        let beta_x = &instance.beta[log_m..];
        let mut z = Vec::with_capacity(beta_x.len() + witness.w.len());
        z.extend_from_slice(beta_x);
        z.extend_from_slice(&witness.w);
        let beta_tau_eq = Poly::<EF>::new_from_point(beta_tau, EF::ONE);
        let eta_recomputed = self.pesat.evaluate_bundled(beta_tau_eq.as_slice(), &z);
        if eta_recomputed != instance.eta {
            return Err(DeciderError::BundledPesat);
        }

        // 4. Codeword consistency: C(w) == f. Encoding lifts F → EF
        //    transparently because RS encoding (DFT + zero-padding) commutes
        //    with the field embedding.
        let f_recomputed = self.code.encode_algebra::<EF>(&witness.w);
        if f_recomputed != witness.f {
            return Err(DeciderError::EncodingMismatch);
        }

        Ok(())
    }
}
