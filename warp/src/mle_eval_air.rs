//! AIR for multilinear-evaluation checks over Plonky3 binomial extensions.
//!
//! This is a reusable arithmetic core for the final WARP decider claim
//! `f_hat(alpha) = mu`. It proves the weighted sum
//!
//! ```text
//!     sum_i values[i] * eq(alpha, binary(i)) = claim
//! ```
//!
//! where `binary(i)` uses the same MSB-first convention as WARP shift
//! queries. The committed-value binding is intentionally outside this AIR:
//! compose this with Merkle/WHIR opening constraints that bind `values[i]` to
//! the accumulator commitment.

use alloc::vec::Vec;
use core::array;

#[cfg(feature = "stark-backend")]
use openvm_stark_backend::{
    AirRef, PartitionedBaseAir, StarkProtocolConfig,
    prover::{AirProvingContext, ColMajorMatrix, CpuColMajorBackend, ProvingContext},
};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::extension::{BinomialExtensionField, BinomiallyExtendable};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

/// AIR checking one multilinear evaluation over a binomial extension.
#[derive(Clone, Debug)]
pub struct BinomialMleEvalAir<F, const D: usize> {
    log_values_len: usize,
    extension_w: F,
}

impl<F, const D: usize> BinomialMleEvalAir<F, D>
where
    F: Copy,
{
    /// Create an AIR using the binomial extension relation `X^D = extension_w`.
    pub fn new(log_values_len: usize, extension_w: F) -> Self {
        assert!(D > 1, "extension degree must be greater than one");
        assert!(
            log_values_len > 0,
            "MLE evaluation AIR needs at least two values"
        );
        Self {
            log_values_len,
            extension_w,
        }
    }

    /// Create an AIR for Plonky3's canonical binomial extension of `F`.
    pub fn for_binomial_extension(log_values_len: usize) -> Self
    where
        F: BinomiallyExtendable<D>,
    {
        Self::new(log_values_len, F::W)
    }

    pub const fn log_values_len(&self) -> usize {
        self.log_values_len
    }

    const fn index_offset(&self) -> usize {
        0
    }

    const fn bits_offset(&self) -> usize {
        1
    }

    const fn prefix_offset(&self) -> usize {
        self.bits_offset() + self.log_values_len
    }

    const fn value_offset(&self) -> usize {
        self.prefix_offset() + (self.log_values_len + 1) * D
    }

    const fn acc_in_offset(&self) -> usize {
        self.value_offset() + D
    }

    const fn acc_out_offset(&self) -> usize {
        self.acc_in_offset() + D
    }

    fn row_width(&self) -> usize {
        self.acc_out_offset() + D
    }
}

impl<F, const D: usize> BaseAir<F> for BinomialMleEvalAir<F, D>
where
    F: Copy + Sync,
{
    fn width(&self) -> usize {
        self.row_width()
    }

    fn num_public_values(&self) -> usize {
        D * (self.log_values_len + 1)
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(3)
    }
}

#[cfg(feature = "stark-backend")]
impl<F, const D: usize> PartitionedBaseAir<F> for BinomialMleEvalAir<F, D> where F: Copy + Sync {}

impl<AB, F, const D: usize> Air<AB> for BinomialMleEvalAir<F, D>
where
    AB: AirBuilder<F = F>,
    F: Field + PrimeCharacteristicRing,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice().to_vec();
        let next = main.next_slice().to_vec();
        let public_values = builder.public_values().to_vec();

        let index: AB::Expr = local[self.index_offset()].into();
        let next_index: AB::Expr = next[self.index_offset()].into();
        let bits = local[self.bits_offset()..self.bits_offset() + self.log_values_len]
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<AB::Expr>>();
        for bit in bits.iter().cloned() {
            builder.assert_bool(bit);
        }

        let recomposed = bits
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (bit_idx, bit)| {
                let shift = self.log_values_len - 1 - bit_idx;
                acc + bit.clone() * AB::Expr::from_usize(1usize << shift)
            });
        builder.assert_eq(index.clone(), recomposed);

        builder.when_first_row().assert_zero(index.clone());
        builder.when_last_row().assert_eq(
            index.clone(),
            AB::Expr::from_usize((1usize << self.log_values_len) - 1),
        );
        builder
            .when_transition()
            .assert_eq(next_index, index + AB::Expr::ONE);

        let acc_in = read_ext::<AB, D>(&local, self.acc_in_offset());
        let acc_out = read_ext::<AB, D>(&local, self.acc_out_offset());
        let next_acc_in = read_ext::<AB, D>(&next, self.acc_in_offset());
        let value = read_ext::<AB, D>(&local, self.value_offset());

        builder
            .when_first_row()
            .assert_zeros(ext_sub::<AB, D>(acc_in.clone(), ext_zero::<AB, D>()));
        builder
            .when_transition()
            .assert_zeros(ext_sub::<AB, D>(next_acc_in, acc_out.clone()));

        let mut prefix = read_ext_at::<AB, D>(&local, self.prefix_offset());
        builder.assert_zeros(ext_sub::<AB, D>(prefix.clone(), ext_one::<AB, D>()));
        for (bit_idx, bit) in bits.iter().enumerate() {
            let point_limb = read_public_ext::<AB, D>(&public_values, bit_idx * D);
            let factor = ext_add::<AB, D>(
                ext_scalar_mul::<AB, D>(
                    ext_sub::<AB, D>(ext_one::<AB, D>(), point_limb.clone()),
                    AB::Expr::ONE - bit.clone(),
                ),
                ext_scalar_mul::<AB, D>(point_limb, bit.clone()),
            );
            let next_prefix =
                read_ext_at::<AB, D>(&local, self.prefix_offset() + (bit_idx + 1) * D);
            let expected = ext_mul::<AB, F, D>(prefix, factor, self.extension_w);
            builder.assert_zeros(ext_sub::<AB, D>(next_prefix.clone(), expected));
            prefix = next_prefix;
        }

        let weighted = ext_mul::<AB, F, D>(value, prefix, self.extension_w);
        builder.assert_zeros(ext_sub::<AB, D>(
            acc_out.clone(),
            ext_add::<AB, D>(acc_in, weighted),
        ));

        let public_claim = read_public_ext::<AB, D>(&public_values, self.log_values_len * D);
        builder
            .when_last_row()
            .assert_zeros(ext_sub::<AB, D>(acc_out, public_claim));
    }
}

/// Build the AIR object used for one MLE evaluation proof.
#[cfg(feature = "stark-backend")]
pub fn binomial_mle_eval_air<F, SC, const D: usize>(log_values_len: usize) -> AirRef<SC>
where
    F: Field + BinomiallyExtendable<D> + Sync + 'static,
    SC: StarkProtocolConfig<F = F>,
{
    alloc::sync::Arc::new(BinomialMleEvalAir::<F, D>::for_binomial_extension(
        log_values_len,
    )) as AirRef<SC>
}

/// Build a trace and public values for one MLE evaluation.
pub fn binomial_mle_eval_air_trace<F, const D: usize>(
    values: &[BinomialExtensionField<F, D>],
    point: &[BinomialExtensionField<F, D>],
) -> (RowMajorMatrix<F>, Vec<F>, BinomialExtensionField<F, D>)
where
    F: Field + BinomiallyExtendable<D> + PrimeCharacteristicRing,
{
    assert!(
        values.len().is_power_of_two() && values.len() >= 2,
        "values length must be a power of two >= 2"
    );
    let log_values_len = values.len().trailing_zeros() as usize;
    assert_eq!(
        point.len(),
        log_values_len,
        "point dimension must be log(values.len())"
    );
    let air = BinomialMleEvalAir::<F, D>::for_binomial_extension(log_values_len);
    let width = air.width();
    let mut rows = Vec::with_capacity(values.len() * width);
    let mut acc = BinomialExtensionField::<F, D>::ZERO;
    for (index, &value) in values.iter().enumerate() {
        let bits = boolean_point_bits::<F>(index, log_values_len);
        let mut prefix = BinomialExtensionField::<F, D>::ONE;
        let mut prefixes = Vec::with_capacity(log_values_len + 1);
        prefixes.push(prefix);
        for (bit, &challenge) in bits.iter().zip(point.iter()) {
            let factor = if *bit == F::ONE {
                challenge
            } else {
                BinomialExtensionField::<F, D>::ONE - challenge
            };
            prefix *= factor;
            prefixes.push(prefix);
        }
        let acc_out = acc + value * prefix;

        rows.push(F::from_usize(index));
        rows.extend_from_slice(&bits);
        for prefix in prefixes {
            push_ext(&mut rows, prefix);
        }
        push_ext(&mut rows, value);
        push_ext(&mut rows, acc);
        push_ext(&mut rows, acc_out);
        acc = acc_out;
    }

    let mut public_values = Vec::with_capacity(D * (log_values_len + 1));
    for &challenge in point {
        push_ext(&mut public_values, challenge);
    }
    push_ext(&mut public_values, acc);

    (RowMajorMatrix::new(rows, width), public_values, acc)
}

/// Build a `stark-backend` proving context for one MLE evaluation AIR.
#[cfg(feature = "stark-backend")]
pub fn binomial_mle_eval_air_context<SC, const D: usize>(
    values: &[BinomialExtensionField<SC::F, D>],
    point: &[BinomialExtensionField<SC::F, D>],
) -> (
    AirProvingContext<CpuColMajorBackend<SC>>,
    BinomialExtensionField<SC::F, D>,
)
where
    SC: StarkProtocolConfig,
    SC::F: Field + BinomiallyExtendable<D> + PrimeCharacteristicRing,
{
    let (trace, public_values, claim) = binomial_mle_eval_air_trace::<SC::F, D>(values, point);
    (
        AirProvingContext::simple(ColMajorMatrix::from_row_major(&trace), public_values),
        claim,
    )
}

/// Build a `stark-backend` proving context for one MLE evaluation AIR.
#[cfg(feature = "stark-backend")]
pub fn binomial_mle_eval_proving_context<SC, const D: usize>(
    air_id: usize,
    values: &[BinomialExtensionField<SC::F, D>],
    point: &[BinomialExtensionField<SC::F, D>],
) -> (
    ProvingContext<CpuColMajorBackend<SC>>,
    BinomialExtensionField<SC::F, D>,
)
where
    SC: StarkProtocolConfig,
    SC::F: Field + BinomiallyExtendable<D> + PrimeCharacteristicRing,
{
    let (ctx, claim) = binomial_mle_eval_air_context::<SC, D>(values, point);
    (ProvingContext::new(alloc::vec![(air_id, ctx)]), claim)
}

fn boolean_point_bits<F>(x: usize, log_n: usize) -> Vec<F>
where
    F: Field,
{
    (0..log_n)
        .map(|i| {
            if (x >> (log_n - 1 - i)) & 1 == 1 {
                F::ONE
            } else {
                F::ZERO
            }
        })
        .collect()
}

fn push_ext<F, const D: usize>(out: &mut Vec<F>, value: BinomialExtensionField<F, D>)
where
    F: Field + BinomiallyExtendable<D>,
{
    out.extend_from_slice(value.as_basis_coefficients_slice());
}

fn read_ext<AB, const D: usize>(row: &[AB::Var], offset: usize) -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    array::from_fn(|i| row[offset + i].into())
}

fn read_ext_at<AB, const D: usize>(row: &[AB::Var], offset: usize) -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    read_ext::<AB, D>(row, offset)
}

fn read_public_ext<AB, const D: usize>(
    public_values: &[AB::PublicVar],
    offset: usize,
) -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    array::from_fn(|i| public_values[offset + i].into())
}

fn ext_zero<AB, const D: usize>() -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    array::from_fn(|_| AB::Expr::ZERO)
}

fn ext_one<AB, const D: usize>() -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    let mut out = ext_zero::<AB, D>();
    out[0] = AB::Expr::ONE;
    out
}

fn ext_add<AB, const D: usize>(lhs: [AB::Expr; D], rhs: [AB::Expr; D]) -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    array::from_fn(|i| lhs[i].clone() + rhs[i].clone())
}

fn ext_sub<AB, const D: usize>(lhs: [AB::Expr; D], rhs: [AB::Expr; D]) -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    array::from_fn(|i| lhs[i].clone() - rhs[i].clone())
}

fn ext_scalar_mul<AB, const D: usize>(value: [AB::Expr; D], scalar: AB::Expr) -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    array::from_fn(|i| value[i].clone() * scalar.clone())
}

fn ext_mul<AB, F, const D: usize>(
    lhs: [AB::Expr; D],
    rhs: [AB::Expr; D],
    extension_w: F,
) -> [AB::Expr; D]
where
    AB: AirBuilder<F = F>,
    F: Field,
{
    let mut coeffs = array::from_fn(|_| AB::Expr::ZERO);
    for (i, lhs_i) in lhs.iter().enumerate() {
        for (j, rhs_j) in rhs.iter().enumerate() {
            let term = lhs_i.clone() * rhs_j.clone();
            let degree = i + j;
            if degree < D {
                coeffs[degree] = coeffs[degree].clone() + term;
            } else {
                coeffs[degree - D] = coeffs[degree - D].clone() + term * extension_w;
            }
        }
    }
    coeffs
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_air::check_constraints;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn honest_trace() -> (BinomialMleEvalAir<F, 4>, RowMajorMatrix<F>, Vec<F>) {
        let values = (0..8)
            .map(|i| EF::new([F::from_u64(i + 3), F::from_u64(2 * i + 1), F::ZERO, F::ONE]))
            .collect::<Vec<_>>();
        let point = vec![
            EF::new([F::from_u64(5), F::ONE, F::ZERO, F::ZERO]),
            EF::new([F::from_u64(7), F::ZERO, F::ONE, F::ZERO]),
            EF::new([F::from_u64(11), F::ZERO, F::ZERO, F::ONE]),
        ];
        let air = BinomialMleEvalAir::<F, 4>::for_binomial_extension(3);
        let (trace, public_values, _claim) = binomial_mle_eval_air_trace::<F, 4>(&values, &point);
        (air, trace, public_values)
    }

    #[test]
    fn binomial_mle_eval_air_accepts_honest_trace() {
        let (air, trace, public_values) = honest_trace();
        check_constraints(&air, &trace, &public_values);
    }

    #[test]
    #[should_panic(expected = "constraints not satisfied")]
    fn binomial_mle_eval_air_rejects_bad_value() {
        let (air, mut trace, public_values) = honest_trace();
        trace.values[air.value_offset()] += F::ONE;
        check_constraints(&air, &trace, &public_values);
    }

    #[test]
    #[should_panic(expected = "constraints not satisfied")]
    fn binomial_mle_eval_air_rejects_bad_public_claim() {
        let (air, trace, mut public_values) = honest_trace();
        let claim_offset = air.log_values_len() * 4;
        public_values[claim_offset] += F::ONE;
        check_constraints(&air, &trace, &public_values);
    }
}
