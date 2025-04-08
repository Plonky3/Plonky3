use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{
    Field, PackedField, TwoAdicField, batch_multiplicative_inverse,
    cyclic_subgroup_coset_known_order,
};

/// Precomputations of the evaluation of `Z_H(X) = X^n - 1` on a coset `s K` with `H <= K`.
#[derive(Debug)]
pub struct VanishingPolynomialOnCoset<F: Field> {
    /// `n = |H|`.
    log_n: usize,
    /// `rate = |K|/|H|`.
    rate_bits: usize,
    coset_shift: F,
    /// Holds `g^n * (w^n)^i - 1 = g^n * v^i - 1` for `i in 0..rate`, with `w` a generator of `K` and `v` a
    /// `rate`-primitive root of unity.
    evals: Vec<F>,
    /// Holds the multiplicative inverses of `evals`.
    inverses: Vec<F>,
}

impl<F: TwoAdicField> VanishingPolynomialOnCoset<F> {
    pub fn new(log_n: usize, rate_bits: usize, coset_shift: F) -> Self {
        let s_pow_n = coset_shift.exp_power_of_2(log_n);
        let evals = F::two_adic_generator(rate_bits)
            .powers()
            .take(1 << rate_bits)
            .map(|x| s_pow_n * x - F::ONE)
            .collect::<Vec<_>>();
        let inverses = batch_multiplicative_inverse(&evals);
        Self {
            log_n,
            rate_bits,
            coset_shift,
            evals,
            inverses,
        }
    }

    /// Returns `Z_H(g * w^i)`.
    pub fn eval(&self, i: usize) -> F {
        self.evals[i & ((1 << self.rate_bits) - 1)]
    }

    /// Returns `1 / Z_H(g * w^i)`.
    pub fn eval_inverse(&self, i: usize) -> F {
        self.inverses[i & ((1 << self.rate_bits) - 1)]
    }

    /// Like `eval_inverse`, but for a range of indices starting with `i_start`.
    pub fn eval_inverse_packed<P: PackedField<Scalar = F>>(&self, i_start: usize) -> P {
        let mut packed = P::ZERO;
        packed
            .as_slice_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(j, packed_j)| *packed_j = self.eval_inverse(i_start + j));
        packed
    }

    /// Evaluate the Langrange basis polynomial, `L_i(x) = Z_H(x) / (x - g_H^i)`, on our coset `s K`.
    /// Here `L_i(x)` is unnormalized in the sense that it evaluates to some nonzero value at `g_H^i`,
    /// not necessarily 1.
    pub fn lagrange_basis_unnormalized(&self, i: usize) -> Vec<F> {
        let log_coset_size = self.log_n + self.rate_bits;
        let coset_size = 1 << log_coset_size;
        let g_h = F::two_adic_generator(self.log_n);
        let g_k = F::two_adic_generator(log_coset_size);

        let target_point = g_h.exp_u64(i as u64);
        let denominators = cyclic_subgroup_coset_known_order(g_k, self.coset_shift, coset_size)
            .map(|x| x - target_point)
            .collect_vec();
        let inverses = batch_multiplicative_inverse(&denominators);

        self.evals
            .iter()
            .cycle()
            .zip(inverses)
            .map(|(&z_h, inv)| z_h * inv)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{PackedValue, PrimeCharacteristicRing};

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_eval_and_inverse() {
        let log_n = 1; // |H| = 2
        let rate_bits = 2; // |K| = 8, rate = 4
        let coset_shift = F::from_u8(3);

        let vanishing = VanishingPolynomialOnCoset::<F>::new(log_n, rate_bits, coset_shift);

        // Compute s^n = 3^2 = 9
        let s_pow_n = F::from_u8(9);

        // Get a primitive 4-th root of unity w4 = g_K where K = size 4
        let w4 = F::two_adic_generator(rate_bits);

        // Compute v^i for i=0..3, where v = w4
        let v0 = F::ONE; // w4^0 = 1
        let v1 = w4; // w4^1
        let v2 = w4 * w4; // w4^2
        let v3 = v2 * w4; // w4^3

        // Compute evals: s^n * v^i - 1
        let z0 = s_pow_n * v0 - F::ONE;
        let z1 = s_pow_n * v1 - F::ONE;
        let z2 = s_pow_n * v2 - F::ONE;
        let z3 = s_pow_n * v3 - F::ONE;

        let expected_evals = vec![z0, z1, z2, z3];
        assert_eq!(vanishing.evals, expected_evals);

        // Compute inverses manually
        let expected_inverses = expected_evals
            .iter()
            .map(|z| z.inverse())
            .collect::<Vec<_>>();
        assert_eq!(vanishing.inverses, expected_inverses);

        // Sanity check: eval(i) * eval_inverse(i) == 1
        for i in 0..4 {
            assert_eq!(
                vanishing.eval(i) * vanishing.eval_inverse(i),
                F::ONE,
                "Z_H * Z_H^-1 mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_eval_index_wraparound() {
        let vanishing = VanishingPolynomialOnCoset::<F>::new(1, 2, F::from_u8(2));
        let rate = 1 << 2; // rate = 4

        // Ensure that indexing wraps modulo rate.
        // For example, eval(0) and eval(4) should access the same slot.
        assert_eq!(vanishing.eval(0), vanishing.eval(rate));
        assert_eq!(vanishing.eval(1), vanishing.eval(rate + 1));
        assert_eq!(vanishing.eval_inverse(3), vanishing.eval_inverse(rate + 3));
    }

    #[test]
    fn test_eval_inverse_packed() {
        let vanishing = VanishingPolynomialOnCoset::<F>::new(1, 2, F::from_u8(7));

        // Get a packed field of inverses starting from index 0
        let packed = vanishing.eval_inverse_packed::<<F as Field>::Packing>(0);
        let unpacked = packed.as_slice();

        // Each entry of the packed field should match the scalar inverse
        for (i, &unpack) in unpacked
            .iter()
            .enumerate()
            .take(<F as Field>::Packing::WIDTH)
        {
            let expected = vanishing.eval_inverse(i);
            assert_eq!(unpack, expected);
        }
    }

    #[test]
    fn test_lagrange_basis_unnormalized_i0() {
        let log_n = 1; // H has size 2
        let rate_bits = 1; // K has size 4, so coset size is 4
        let coset_shift = F::TWO;

        let vanishing = VanishingPolynomialOnCoset::<F>::new(log_n, rate_bits, coset_shift);

        // Compute generators
        let g_h = F::two_adic_generator(log_n); // g_H of order 2 (i.e., g_H = -1)
        let g_k = F::two_adic_generator(log_n + rate_bits); // g_K of order 4

        let target_point = g_h.exp_u64(0); // g_H^0 = 1

        // Compute coset: s * g_K^i for i = 0..3
        let w0 = coset_shift * F::ONE;
        let w1 = coset_shift * g_k;
        let w2 = w1 * g_k;
        let w3 = w2 * g_k;

        let coset = [w0, w1, w2, w3];

        // Compute denominators for L_0(x): x - 1
        let denoms = coset.iter().map(|x| *x - target_point).collect::<Vec<_>>();
        let invs = denoms.iter().map(|d| d.inverse()).collect::<Vec<_>>();

        // Cycle through evals as needed
        let evals = &vanishing.evals;

        // Since rate = 2, evals = [z0, z1] and repeat
        let expected = vec![
            evals[0] * invs[0],
            evals[1] * invs[1],
            evals[0] * invs[2],
            evals[1] * invs[3],
        ];

        // Evaluate the unnormalized L_0(x) over the coset
        let actual = vanishing.lagrange_basis_unnormalized(0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_lagrange_basis_unnormalized_i1() {
        let log_n = 1;
        let rate_bits = 1;
        let coset_shift = F::from_u8(2); // Changed from 1 to 2 to avoid hitting -1

        let vanishing = VanishingPolynomialOnCoset::<F>::new(log_n, rate_bits, coset_shift);

        // generator of K (order 4)
        let g_k = F::two_adic_generator(log_n + rate_bits);

        // Build the coset: s * g_k^i for i = 0..3
        let w0 = coset_shift * F::ONE;
        let w1 = coset_shift * g_k;
        let w2 = w1 * g_k;
        let w3 = w2 * g_k;
        let coset = [w0, w1, w2, w3];

        // Denominators for L_1(x) = Z_H(x) / (x - g_H^1) = Z_H(x) / (x + 1)
        // Since s ≠ 1, -1 is not in the coset => safe
        let denoms = coset.iter().map(|x| *x + F::ONE).collect::<Vec<_>>();
        let invs = denoms.iter().map(|d| d.inverse()).collect::<Vec<_>>();

        let evals = &vanishing.evals;

        // Cycle through evals: [z0, z1], then repeat
        let expected = vec![
            evals[0] * invs[0],
            evals[1] * invs[1],
            evals[0] * invs[2],
            evals[1] * invs[3],
        ];

        let actual = vanishing.lagrange_basis_unnormalized(1);
        assert_eq!(actual, expected);
    }
}
