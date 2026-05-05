//! AIR for a random-point Reed-Solomon encoding check.
//!
//! WARP's final decider checks `C(w) == f`. For coefficient-form RS codes,
//! this means:
//!
//! - `w = (w_0, ..., w_{k-1})` are coefficients of a degree-`< k`
//!   polynomial `W(X)`.
//! - `f_i = W(g^i)` for the size-`n` two-adic generator `g`.
//!
//! A succinct way to prove this relation is to sample `z` outside the RS
//! domain and prove
//!
//! ```text
//!     Interp_H(f)(z) = W(z)
//! ```
//!
//! where `H = {g^i}`. If `f` is not the RS encoding of `w`, the difference
//! polynomial has degree `< n` and a transcript-sampled `z` catches it except
//! with probability at most `(n - 1) / |F_ext|`.
//!
//! This AIR proves the arithmetic of that random check. It does not itself
//! derive `z`; callers must bind `z` through Fiat-Shamir/public inputs.

use alloc::vec::Vec;
use core::array;

#[cfg(feature = "stark-backend")]
use openvm_stark_backend::{
    AirRef, PartitionedBaseAir, StarkProtocolConfig,
    prover::{AirProvingContext, ColMajorMatrix, CpuColMajorBackend, ProvingContext},
};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::extension::{BinomialExtensionField, BinomiallyExtendable};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

/// AIR checking coefficient-form RS consistency at one random point.
#[derive(Clone, Debug)]
pub struct BinomialRsEncodingAir<F, const D: usize> {
    log_message_len: usize,
    log_codeword_len: usize,
    root: F,
    inv_n: F,
    extension_w: F,
}

impl<F, const D: usize> BinomialRsEncodingAir<F, D>
where
    F: TwoAdicField,
{
    /// Create an AIR for a coefficient-form RS code of message length
    /// `2^log_message_len` and codeword length `2^log_codeword_len`.
    pub fn new(log_message_len: usize, log_codeword_len: usize, extension_w: F) -> Self {
        assert!(D > 1, "extension degree must be greater than one");
        assert!(
            log_message_len > 0,
            "RS encoding AIR needs at least two message values",
        );
        assert!(
            log_message_len <= log_codeword_len,
            "message length cannot exceed codeword length",
        );
        let n = 1usize << log_codeword_len;
        Self {
            log_message_len,
            log_codeword_len,
            root: F::two_adic_generator(log_codeword_len),
            inv_n: F::from_usize(n).inverse(),
            extension_w,
        }
    }

    /// Create an AIR for Plonky3's canonical binomial extension of `F`.
    pub fn for_binomial_extension(log_message_len: usize, log_codeword_len: usize) -> Self
    where
        F: BinomiallyExtendable<D>,
    {
        Self::new(log_message_len, log_codeword_len, F::W)
    }

    pub const fn log_message_len(&self) -> usize {
        self.log_message_len
    }

    pub const fn log_codeword_len(&self) -> usize {
        self.log_codeword_len
    }

    const fn index_offset(&self) -> usize {
        0
    }

    const fn active_w_offset(&self) -> usize {
        1
    }

    const fn active_count_offset(&self) -> usize {
        2
    }

    const fn domain_x_offset(&self) -> usize {
        3
    }

    const fn denom_inv_offset(&self) -> usize {
        4
    }

    const fn factor_offset(&self) -> usize {
        self.denom_inv_offset() + D
    }

    const fn z_power_offset(&self) -> usize {
        self.factor_offset() + D
    }

    const fn f_value_offset(&self) -> usize {
        self.z_power_offset() + D
    }

    const fn w_value_offset(&self) -> usize {
        self.f_value_offset() + D
    }

    const fn acc_f_in_offset(&self) -> usize {
        self.w_value_offset() + D
    }

    const fn acc_f_out_offset(&self) -> usize {
        self.acc_f_in_offset() + D
    }

    const fn acc_w_in_offset(&self) -> usize {
        self.acc_f_out_offset() + D
    }

    const fn acc_w_out_offset(&self) -> usize {
        self.acc_w_in_offset() + D
    }

    fn row_width(&self) -> usize {
        self.acc_w_out_offset() + D
    }
}

impl<F, const D: usize> BaseAir<F> for BinomialRsEncodingAir<F, D>
where
    F: TwoAdicField + Sync,
{
    fn width(&self) -> usize {
        self.row_width()
    }

    fn num_public_values(&self) -> usize {
        D
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(3)
    }
}

#[cfg(feature = "stark-backend")]
impl<F, const D: usize> PartitionedBaseAir<F> for BinomialRsEncodingAir<F, D> where
    F: TwoAdicField + Sync
{
}

impl<AB, F, const D: usize> Air<AB> for BinomialRsEncodingAir<F, D>
where
    AB: AirBuilder<F = F>,
    F: TwoAdicField + PrimeCharacteristicRing,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice().to_vec();
        let next = main.next_slice().to_vec();
        let public_values = builder.public_values().to_vec();

        let index: AB::Expr = local[self.index_offset()].into();
        let next_index: AB::Expr = next[self.index_offset()].into();
        let active_w: AB::Expr = local[self.active_w_offset()].into();
        let next_active_w: AB::Expr = next[self.active_w_offset()].into();
        let active_count: AB::Expr = local[self.active_count_offset()].into();
        let next_active_count: AB::Expr = next[self.active_count_offset()].into();
        let domain_x: AB::Expr = local[self.domain_x_offset()].into();
        let next_domain_x: AB::Expr = next[self.domain_x_offset()].into();

        builder.assert_bool(active_w.clone());
        builder.when_first_row().assert_one(active_w.clone());
        builder
            .when_transition()
            .assert_zero(next_active_w.clone() * (AB::Expr::ONE - active_w.clone()));
        builder.when_first_row().assert_zero(index.clone());
        builder.when_last_row().assert_eq(
            index.clone(),
            AB::Expr::from_usize((1usize << self.log_codeword_len) - 1),
        );
        builder
            .when_transition()
            .assert_eq(next_index, index + AB::Expr::ONE);

        builder.when_first_row().assert_zero(active_count.clone());
        builder
            .when_transition()
            .assert_eq(next_active_count, active_count.clone() + active_w.clone());
        builder.when_last_row().assert_eq(
            active_count + active_w.clone(),
            AB::Expr::from_usize(1usize << self.log_message_len),
        );

        builder.when_first_row().assert_one(domain_x.clone());
        builder
            .when_transition()
            .assert_eq(next_domain_x, domain_x.clone() * self.root);

        let z = read_public_ext::<AB, D>(&public_values, 0);
        let denom_inv = read_ext::<AB, D>(&local, self.denom_inv_offset());
        let factor = read_ext::<AB, D>(&local, self.factor_offset());
        let next_factor = read_ext::<AB, D>(&next, self.factor_offset());
        let z_power = read_ext::<AB, D>(&local, self.z_power_offset());
        let next_z_power = read_ext::<AB, D>(&next, self.z_power_offset());
        let f_value = read_ext::<AB, D>(&local, self.f_value_offset());
        let w_value = read_ext::<AB, D>(&local, self.w_value_offset());
        let acc_f_in = read_ext::<AB, D>(&local, self.acc_f_in_offset());
        let acc_f_out = read_ext::<AB, D>(&local, self.acc_f_out_offset());
        let next_acc_f_in = read_ext::<AB, D>(&next, self.acc_f_in_offset());
        let acc_w_in = read_ext::<AB, D>(&local, self.acc_w_in_offset());
        let acc_w_out = read_ext::<AB, D>(&local, self.acc_w_out_offset());
        let next_acc_w_in = read_ext::<AB, D>(&next, self.acc_w_in_offset());

        builder
            .when_first_row()
            .assert_zeros(ext_sub::<AB, D>(z_power.clone(), ext_one::<AB, D>()));
        builder.when_transition().assert_zeros(ext_sub::<AB, D>(
            next_z_power,
            ext_mul::<AB, F, D>(z_power.clone(), z.clone(), self.extension_w),
        ));
        builder
            .when_transition()
            .assert_zeros(ext_sub::<AB, D>(next_factor, factor.clone()));
        builder.when_last_row().assert_zeros(ext_sub::<AB, D>(
            ext_scalar_mul::<AB, D>(factor.clone(), AB::Expr::from(self.inv_n.inverse())),
            ext_sub::<AB, D>(
                ext_mul::<AB, F, D>(z_power.clone(), z.clone(), self.extension_w),
                ext_one::<AB, D>(),
            ),
        ));

        let domain_x_ext = ext_scalar::<AB, D>(domain_x.clone());
        builder.assert_zeros(ext_sub::<AB, D>(
            ext_mul::<AB, F, D>(
                ext_sub::<AB, D>(z.clone(), domain_x_ext),
                denom_inv.clone(),
                self.extension_w,
            ),
            ext_one::<AB, D>(),
        ));

        builder
            .when(AB::Expr::ONE - active_w.clone())
            .assert_zeros(w_value.clone());

        let barycentric_weight = ext_mul::<AB, F, D>(
            ext_scalar_mul::<AB, D>(factor, domain_x),
            denom_inv,
            self.extension_w,
        );
        let f_contribution = ext_mul::<AB, F, D>(f_value, barycentric_weight, self.extension_w);
        let w_contribution = ext_mul::<AB, F, D>(w_value, z_power, self.extension_w);

        builder
            .when_first_row()
            .assert_zeros(ext_sub::<AB, D>(acc_f_in.clone(), ext_zero::<AB, D>()));
        builder
            .when_first_row()
            .assert_zeros(ext_sub::<AB, D>(acc_w_in.clone(), ext_zero::<AB, D>()));
        builder
            .when_transition()
            .assert_zeros(ext_sub::<AB, D>(next_acc_f_in, acc_f_out.clone()));
        builder
            .when_transition()
            .assert_zeros(ext_sub::<AB, D>(next_acc_w_in, acc_w_out.clone()));
        builder.assert_zeros(ext_sub::<AB, D>(
            acc_f_out.clone(),
            ext_add::<AB, D>(acc_f_in, f_contribution),
        ));
        builder.assert_zeros(ext_sub::<AB, D>(
            acc_w_out.clone(),
            ext_add::<AB, D>(acc_w_in, w_contribution),
        ));
        builder
            .when_last_row()
            .assert_zeros(ext_sub::<AB, D>(acc_f_out, acc_w_out));
    }
}

/// Build the AIR object used for one RS encoding random check.
#[cfg(feature = "stark-backend")]
pub fn binomial_rs_encoding_air<F, SC, const D: usize>(
    log_message_len: usize,
    log_codeword_len: usize,
) -> AirRef<SC>
where
    F: TwoAdicField + BinomiallyExtendable<D> + Sync + 'static,
    SC: StarkProtocolConfig<F = F>,
{
    alloc::sync::Arc::new(BinomialRsEncodingAir::<F, D>::for_binomial_extension(
        log_message_len,
        log_codeword_len,
    )) as AirRef<SC>
}

/// Build a trace and public values for one coefficient-form RS encoding
/// random check.
pub fn binomial_rs_encoding_air_trace<F, const D: usize>(
    message: &[BinomialExtensionField<F, D>],
    codeword: &[BinomialExtensionField<F, D>],
    z: BinomialExtensionField<F, D>,
) -> RowMajorMatrix<F>
where
    F: TwoAdicField + BinomiallyExtendable<D> + PrimeCharacteristicRing,
{
    assert!(
        message.len().is_power_of_two() && message.len() >= 2,
        "message length must be a power of two >= 2",
    );
    assert!(
        codeword.len().is_power_of_two() && codeword.len() >= message.len(),
        "codeword length must be a power of two >= message length",
    );
    let log_message_len = message.len().trailing_zeros() as usize;
    let log_codeword_len = codeword.len().trailing_zeros() as usize;
    let air =
        BinomialRsEncodingAir::<F, D>::for_binomial_extension(log_message_len, log_codeword_len);
    let width = air.width();
    let n = codeword.len();
    let k = message.len();
    let n_f = F::from_usize(n);
    let root = F::two_adic_generator(log_codeword_len);
    let factor = (z.exp_power_of_2(log_codeword_len) - BinomialExtensionField::<F, D>::ONE)
        * BinomialExtensionField::<F, D>::from(n_f.inverse());

    let mut rows = Vec::with_capacity(n * width);
    let mut domain_x = F::ONE;
    let mut z_power = BinomialExtensionField::<F, D>::ONE;
    let mut acc_f = BinomialExtensionField::<F, D>::ZERO;
    let mut acc_w = BinomialExtensionField::<F, D>::ZERO;
    for (index, &f_value) in codeword.iter().enumerate() {
        let active_w = index < k;
        let w_value = if active_w {
            message[index]
        } else {
            BinomialExtensionField::<F, D>::ZERO
        };
        let denom_inv = (z - BinomialExtensionField::<F, D>::from(domain_x)).inverse();
        let weight = factor * BinomialExtensionField::<F, D>::from(domain_x) * denom_inv;
        let acc_f_out = acc_f + f_value * weight;
        let acc_w_out = acc_w + w_value * z_power;

        rows.push(F::from_usize(index));
        rows.push(F::from_bool(active_w));
        rows.push(F::from_usize(index.min(k)));
        rows.push(domain_x);
        push_ext(&mut rows, denom_inv);
        push_ext(&mut rows, factor);
        push_ext(&mut rows, z_power);
        push_ext(&mut rows, f_value);
        push_ext(&mut rows, w_value);
        push_ext(&mut rows, acc_f);
        push_ext(&mut rows, acc_f_out);
        push_ext(&mut rows, acc_w);
        push_ext(&mut rows, acc_w_out);

        domain_x *= root;
        z_power *= z;
        acc_f = acc_f_out;
        acc_w = acc_w_out;
    }
    RowMajorMatrix::new(rows, width)
}

/// Build a `stark-backend` proving context for one RS encoding AIR.
#[cfg(feature = "stark-backend")]
pub fn binomial_rs_encoding_air_context<SC, const D: usize>(
    message: &[BinomialExtensionField<SC::F, D>],
    codeword: &[BinomialExtensionField<SC::F, D>],
    z: BinomialExtensionField<SC::F, D>,
) -> AirProvingContext<CpuColMajorBackend<SC>>
where
    SC: StarkProtocolConfig,
    SC::F: TwoAdicField + BinomiallyExtendable<D> + PrimeCharacteristicRing,
{
    let trace = binomial_rs_encoding_air_trace::<SC::F, D>(message, codeword, z);
    let public_values = z.as_basis_coefficients_slice().to_vec();
    AirProvingContext::simple(ColMajorMatrix::from_row_major(&trace), public_values)
}

/// Build a `stark-backend` proving context for one RS encoding AIR.
#[cfg(feature = "stark-backend")]
pub fn binomial_rs_encoding_proving_context<SC, const D: usize>(
    air_id: usize,
    message: &[BinomialExtensionField<SC::F, D>],
    codeword: &[BinomialExtensionField<SC::F, D>],
    z: BinomialExtensionField<SC::F, D>,
) -> ProvingContext<CpuColMajorBackend<SC>>
where
    SC: StarkProtocolConfig,
    SC::F: TwoAdicField + BinomiallyExtendable<D> + PrimeCharacteristicRing,
{
    ProvingContext::new(alloc::vec![(
        air_id,
        binomial_rs_encoding_air_context::<SC, D>(message, codeword, z),
    )])
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

fn ext_scalar<AB, const D: usize>(value: AB::Expr) -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    let mut out = ext_zero::<AB, D>();
    out[0] = value;
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
    use alloc::vec::Vec;

    use p3_air::check_constraints;
    use p3_baby_bear::BabyBear;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::ReedSolomonCode;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn honest_trace() -> (BinomialRsEncodingAir<F, 4>, RowMajorMatrix<F>, Vec<F>) {
        let code = ReedSolomonCode::<F, Radix2DFTSmallBatch<F>>::new_coefficient(
            3,
            1,
            Radix2DFTSmallBatch::default(),
        );
        let message = (0..code.msg_len())
            .map(|i| EF::new([F::from_u64(i as u64 + 2), F::ONE, F::ZERO, F::ZERO]))
            .collect::<Vec<_>>();
        let codeword = code.encode_algebra(&message);
        let z = EF::new([F::from_u64(7), F::from_u64(3), F::ONE, F::ZERO]);
        let air = BinomialRsEncodingAir::<F, 4>::for_binomial_extension(
            code.log_msg_len(),
            code.log_codeword_len(),
        );
        let trace = binomial_rs_encoding_air_trace::<F, 4>(&message, &codeword, z);
        let public_values = z.as_basis_coefficients_slice().to_vec();
        (air, trace, public_values)
    }

    #[test]
    fn binomial_rs_encoding_air_accepts_honest_trace() {
        let (air, trace, public_values) = honest_trace();
        check_constraints(&air, &trace, &public_values);
    }

    #[test]
    #[should_panic(expected = "constraints not satisfied")]
    fn binomial_rs_encoding_air_rejects_bad_codeword_value() {
        let (air, mut trace, public_values) = honest_trace();
        trace.values[air.f_value_offset()] += F::ONE;
        check_constraints(&air, &trace, &public_values);
    }

    #[test]
    #[should_panic(expected = "constraints not satisfied")]
    fn binomial_rs_encoding_air_rejects_bad_message_value() {
        let (air, mut trace, public_values) = honest_trace();
        trace.values[air.w_value_offset()] += F::ONE;
        check_constraints(&air, &trace, &public_values);
    }
}
