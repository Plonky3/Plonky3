//! Direct Boolean PESAT relation.
//!
//! This is the PESAT relation used by the native kernel benchmarks:
//!
//! ```text
//!     Pb(beta, w) = sum_i eq(beta, i) * w_i * (w_i - 1)
//! ```
//!
//! It implements the WARP paper's Claim 6.5 directly for the §6.3
//! constraint-side round polynomial. This avoids routing the same simple
//! relation through a generic constraint interpreter in prover hot loops.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing};

use super::claim_6_5::{
    Claim65Scratch, fold_claim_6_5_packed_round, fold_claim_6_5_scalar_round,
    packed_ext_scalar_with_scratch, unpack_packed_coeffs_to_scalar,
};
use super::{BundledPesat, PesatShape};

/// PESAT relation for one Boolean constraint per witness coordinate.
#[derive(Clone, Debug)]
pub struct BooleanPesat<F, EF> {
    log_witness: usize,
    description: Vec<u8>,
    _ph: PhantomData<fn() -> (F, EF)>,
}

impl<F, EF> BooleanPesat<F, EF> {
    /// Create the direct Boolean PESAT relation over `2^log_witness` values.
    pub fn new(log_witness: usize, description: Vec<u8>) -> Self {
        assert!(log_witness > 0, "BooleanPesat needs at least two values");
        Self {
            log_witness,
            description,
            _ph: PhantomData,
        }
    }

    /// Number of witness values.
    #[inline]
    pub const fn witness_len(&self) -> usize {
        1 << self.log_witness
    }
}

impl<F, EF> BundledPesat<F, EF> for BooleanPesat<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn shape(&self) -> PesatShape {
        PesatShape {
            log_constraints: self.log_witness,
            log_witness: self.log_witness,
            explicit_len: 0,
            max_degree: 2,
        }
    }

    fn evaluate_bundled(&self, tau_eq: &[EF], z: &[EF]) -> EF {
        assert_eq!(tau_eq.len(), self.witness_len(), "tau_eq length");
        assert_eq!(z.len(), self.witness_len(), "z length");
        tau_eq
            .iter()
            .zip(z)
            .map(|(&weight, &value)| weight * value * (value - EF::ONE))
            .sum()
    }

    fn iter_constraint_polys_at_lerp(
        &self,
        b_x_lo: &[EF],
        b_x_hi: &[EF],
        w_lo: &[EF],
        w_hi: &[EF],
    ) -> Vec<Vec<EF>> {
        assert!(b_x_lo.is_empty(), "BooleanPesat has no explicit vars");
        assert!(b_x_hi.is_empty(), "BooleanPesat has no explicit vars");
        assert_eq!(w_lo.len(), self.witness_len(), "w_lo length");
        assert_eq!(w_hi.len(), self.witness_len(), "w_hi length");
        w_lo.iter()
            .zip(w_hi)
            .map(|(&lo, &hi)| boolean_lerp_poly(lo, hi).to_vec())
            .collect()
    }

    fn bundled_round_poly(&self, b_lo: &[EF], b_hi: &[EF], w_lo: &[EF], w_hi: &[EF]) -> Vec<EF> {
        assert_eq!(b_lo.len(), self.log_witness, "b_lo length");
        assert_eq!(b_hi.len(), self.log_witness, "b_hi length");
        assert_eq!(w_lo.len(), self.witness_len(), "w_lo length");
        assert_eq!(w_hi.len(), self.witness_len(), "w_hi length");
        let mut out = Vec::new();
        let mut scratch = Claim65Scratch::<F, EF>::new();
        boolean_claim_6_5_coeffs_into::<F, EF>(b_lo, b_hi, w_lo, w_hi, &mut out, &mut scratch);
        out
    }

    fn bundled_round_poly_into(
        &self,
        b_lo: &[EF],
        b_hi: &[EF],
        w_lo: &[EF],
        w_hi: &[EF],
        out: &mut Vec<EF>,
        scratch: &mut Claim65Scratch<F, EF>,
    ) {
        assert_eq!(b_lo.len(), self.log_witness, "b_lo length");
        assert_eq!(b_hi.len(), self.log_witness, "b_hi length");
        assert_eq!(w_lo.len(), self.witness_len(), "w_lo length");
        assert_eq!(w_hi.len(), self.witness_len(), "w_hi length");
        boolean_claim_6_5_coeffs_into::<F, EF>(b_lo, b_hi, w_lo, w_hi, out, scratch);
    }

    fn description(&self) -> Vec<u8> {
        self.description.clone()
    }
}

#[inline]
fn boolean_lerp_poly<EF: Field>(lo: EF, hi: EF) -> [EF; 3] {
    let diff = hi - lo;
    [lo * (lo - EF::ONE), diff * (lo + lo - EF::ONE), diff * diff]
}

#[cfg(test)]
fn boolean_claim_6_5_coeffs<F, EF>(b_lo: &[EF], b_hi: &[EF], w_lo: &[EF], w_hi: &[EF]) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut out = Vec::new();
    let mut scratch = Claim65Scratch::<F, EF>::new();
    boolean_claim_6_5_coeffs_into::<F, EF>(b_lo, b_hi, w_lo, w_hi, &mut out, &mut scratch);
    out
}

fn boolean_claim_6_5_coeffs_into<F, EF>(
    b_lo: &[EF],
    b_hi: &[EF],
    w_lo: &[EF],
    w_hi: &[EF],
    out: &mut Vec<EF>,
    scratch: &mut Claim65Scratch<F, EF>,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    let pack_w = F::Packing::WIDTH;
    if w_lo.len() >= pack_w * 2 && w_lo.len().is_multiple_of(pack_w) {
        boolean_claim_6_5_coeffs_packed_prefix_into::<F, EF>(b_lo, b_hi, w_lo, w_hi, out, scratch);
    } else {
        boolean_claim_6_5_coeffs_scalar_into::<F, EF>(b_lo, b_hi, w_lo, w_hi, out, scratch);
    }
}

#[cfg(test)]
fn boolean_claim_6_5_coeffs_scalar<F, EF>(
    b_lo: &[EF],
    b_hi: &[EF],
    w_lo: &[EF],
    w_hi: &[EF],
) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut out = Vec::new();
    let mut scratch = Claim65Scratch::<F, EF>::new();
    boolean_claim_6_5_coeffs_scalar_into::<F, EF>(b_lo, b_hi, w_lo, w_hi, &mut out, &mut scratch);
    out
}

fn boolean_claim_6_5_coeffs_scalar_into<F, EF>(
    b_lo: &[EF],
    b_hi: &[EF],
    w_lo: &[EF],
    w_hi: &[EF],
    out: &mut Vec<EF>,
    scratch: &mut Claim65Scratch<F, EF>,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut width = 3;
    let mut len = w_lo.len();
    scratch.scalar_current.clear();
    for (&lo, &hi) in w_lo.iter().zip(w_hi) {
        scratch
            .scalar_current
            .extend_from_slice(&boolean_lerp_poly(lo, hi));
    }

    for (&c0, &c_at_one) in b_lo.iter().zip(b_hi) {
        let c1 = c_at_one - c0;
        let next_len = len / 2;
        let next_width = width + 1;
        scratch.scalar_next.clear();
        scratch.scalar_next.resize(next_len * next_width, EF::ZERO);
        fold_claim_6_5_scalar_round(
            &scratch.scalar_current,
            width,
            len,
            c0,
            c1,
            &mut scratch.scalar_next,
        );
        core::mem::swap(&mut scratch.scalar_current, &mut scratch.scalar_next);
        width = next_width;
        len = next_len;
    }

    debug_assert_eq!(len, 1);
    out.clear();
    out.extend_from_slice(&scratch.scalar_current);
}

fn boolean_claim_6_5_coeffs_packed_prefix_into<F, EF>(
    b_lo: &[EF],
    b_hi: &[EF],
    w_lo: &[EF],
    w_hi: &[EF],
    out: &mut Vec<EF>,
    scratch: &mut Claim65Scratch<F, EF>,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    let pack_w = F::Packing::WIDTH;
    let log_pack = pack_w.trailing_zeros() as usize;
    let log_n = w_lo.len().trailing_zeros() as usize;
    let packed_rounds = log_n.saturating_sub(log_pack);

    let mut width = 3;
    let mut len = w_lo.len() / pack_w;
    scratch.packed_current.clear();
    scratch.lane_buf.clear();
    scratch.lane_buf.resize(3 * pack_w, EF::ZERO);
    let (q0, rest) = scratch.lane_buf.split_at_mut(pack_w);
    let (q1, q2) = rest.split_at_mut(pack_w);
    for (lo_chunk, hi_chunk) in w_lo.chunks_exact(pack_w).zip(w_hi.chunks_exact(pack_w)) {
        for lane in 0..pack_w {
            let coeffs = boolean_lerp_poly(lo_chunk[lane], hi_chunk[lane]);
            q0[lane] = coeffs[0];
            q1[lane] = coeffs[1];
            q2[lane] = coeffs[2];
        }
        scratch
            .packed_current
            .push(<EF::ExtensionPacking as PackedFieldExtension<F, EF>>::from_ext_slice(q0));
        scratch
            .packed_current
            .push(<EF::ExtensionPacking as PackedFieldExtension<F, EF>>::from_ext_slice(q1));
        scratch
            .packed_current
            .push(<EF::ExtensionPacking as PackedFieldExtension<F, EF>>::from_ext_slice(q2));
    }

    for (&c0, &c_at_one) in b_lo.iter().zip(b_hi).take(packed_rounds) {
        let c1 = c_at_one - c0;
        let c0_packed = packed_ext_scalar_with_scratch::<F, EF>(c0, &mut scratch.broadcast_buf);
        let c1_packed = packed_ext_scalar_with_scratch::<F, EF>(c1, &mut scratch.broadcast_buf);
        let next_len = len / 2;
        let next_width = width + 1;
        scratch.packed_next.clear();
        scratch
            .packed_next
            .resize(next_len * next_width, EF::ExtensionPacking::ZERO);
        fold_claim_6_5_packed_round::<F, EF>(
            &scratch.packed_current,
            width,
            len,
            c0_packed,
            c1_packed,
            &mut scratch.packed_next,
        );
        core::mem::swap(&mut scratch.packed_current, &mut scratch.packed_next);
        width = next_width;
        len = next_len;
    }
    debug_assert_eq!(len, 1);

    unpack_packed_coeffs_to_scalar::<F, EF>(
        &scratch.packed_current,
        width,
        &mut scratch.scalar_current,
    );
    let mut scalar_width = width;
    let mut scalar_len = pack_w;
    for (&c0, &c_at_one) in b_lo.iter().zip(b_hi).skip(packed_rounds) {
        let c1 = c_at_one - c0;
        let next_len = scalar_len / 2;
        let next_width = scalar_width + 1;
        scratch.scalar_next.clear();
        scratch.scalar_next.resize(next_len * next_width, EF::ZERO);
        fold_claim_6_5_scalar_round(
            &scratch.scalar_current,
            scalar_width,
            scalar_len,
            c0,
            c1,
            &mut scratch.scalar_next,
        );
        core::mem::swap(&mut scratch.scalar_current, &mut scratch.scalar_next);
        scalar_width = next_width;
        scalar_len = next_len;
    }

    debug_assert_eq!(scalar_len, 1);
    debug_assert_eq!(scalar_width, log_n + 3);
    out.clear();
    out.extend_from_slice(&scratch.scalar_current);
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PackedValue, PrimeCharacteristicRing};
    use p3_multilinear_util::poly::Poly;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::relation::eq_dot_q_recursive;

    type TestF = BabyBear;
    type TestEF = BinomialExtensionField<TestF, 4>;

    fn reference(
        b_lo: &[TestEF],
        b_hi: &[TestEF],
        w_lo: &[TestEF],
        w_hi: &[TestEF],
    ) -> Vec<TestEF> {
        let c = b_lo
            .iter()
            .zip(b_hi)
            .map(|(&lo, &hi)| [lo, hi - lo])
            .collect::<Vec<_>>();
        let q = w_lo
            .iter()
            .zip(w_hi)
            .map(|(&lo, &hi)| boolean_lerp_poly(lo, hi).to_vec())
            .collect::<Vec<_>>();
        eq_dot_q_recursive(&c, q)
    }

    fn deterministic_vec(len: usize, seed: u64) -> Vec<TestEF> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..len).map(|_| rng.random::<TestEF>()).collect()
    }

    #[test]
    fn boolean_claim_6_5_scalar_matches_reference() {
        let n = 8usize;
        let b_lo = deterministic_vec(n.ilog2() as usize, 1);
        let b_hi = deterministic_vec(n.ilog2() as usize, 2);
        let w_lo = deterministic_vec(n, 3);
        let w_hi = deterministic_vec(n, 4);
        let expected = reference(&b_lo, &b_hi, &w_lo, &w_hi);
        let actual = boolean_claim_6_5_coeffs_scalar::<TestF, TestEF>(&b_lo, &b_hi, &w_lo, &w_hi);
        assert_eq!(actual, expected);
    }

    #[test]
    fn boolean_claim_6_5_packed_prefix_matches_reference() {
        let pack_w = <TestF as Field>::Packing::WIDTH;
        let n = pack_w * 16;
        let b_lo = deterministic_vec(n.ilog2() as usize, 5);
        let b_hi = deterministic_vec(n.ilog2() as usize, 6);
        let w_lo = deterministic_vec(n, 7);
        let w_hi = deterministic_vec(n, 8);
        let expected = reference(&b_lo, &b_hi, &w_lo, &w_hi);
        let actual = boolean_claim_6_5_coeffs::<TestF, TestEF>(&b_lo, &b_hi, &w_lo, &w_hi);
        assert_eq!(actual, expected);
    }

    #[test]
    fn boolean_pesat_evaluate_bundled_matches_definition() {
        let log_n = 4;
        let n = 1 << log_n;
        let pesat = BooleanPesat::<TestF, TestEF>::new(log_n, b"test".to_vec());
        let beta = deterministic_vec(log_n, 9);
        let tau_eq = Poly::<TestEF>::new_from_point(&beta, TestEF::ONE);
        let witness = deterministic_vec(n, 10);
        let expected = tau_eq
            .as_slice()
            .iter()
            .zip(&witness)
            .map(|(&weight, &value)| weight * value * (value - TestEF::ONE))
            .sum::<TestEF>();
        let actual = <BooleanPesat<TestF, TestEF> as BundledPesat<TestF, TestEF>>::evaluate_bundled(
            &pesat,
            tau_eq.as_slice(),
            &witness,
        );
        assert_eq!(actual, expected);
    }
}
