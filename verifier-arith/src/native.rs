use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{ExtensionField, Field, HornerIter, TwoAdicField, batch_multiplicative_inverse};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::reverse_bits_len;

use crate::{Arith, Lift};

/// Native field arithmetic: values are the field elements themselves.
#[derive(Clone, Copy, Debug, Default)]
pub struct Native<Val, Challenge>(PhantomData<(Val, Challenge)>);

impl<Val, Challenge> Native<Val, Challenge> {
    /// The native arithmetic.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Val: Field, Challenge: ExtensionField<Val>> Arith for Native<Val, Challenge> {
    type Val = Val;
    type Challenge = Challenge;
    type F = Val;
    type EF = Challenge;
    type Bit = bool;
    type Index = usize;

    #[inline]
    fn bit(&self, bit: &bool) -> Val {
        Val::from_bool(*bit)
    }

    fn index_bits(&mut self, index: &usize, count: usize) -> Vec<bool> {
        (0..count).map(|i| (index >> i) & 1 == 1).collect()
    }

    #[inline]
    fn index_shr(&self, index: &usize, shift: usize) -> usize {
        index >> shift
    }

    #[inline]
    fn reversed_index_power(&mut self, generator: Val, index: &usize, bits: usize) -> Val {
        generator.exp_u64(reverse_bits_len(*index, bits) as u64)
    }

    fn fold_coset_row(
        &mut self,
        index: &usize,
        log_height: usize,
        log_arity: usize,
        point: Challenge,
        evals: &[Challenge],
    ) -> Challenge
    where
        Val: TwoAdicField,
    {
        assert_eq!(evals.len(), 1 << log_arity, "one evaluation per row entry");
        let generator = Val::two_adic_generator(log_height + log_arity);
        let power = reverse_bits_len(*index, log_height) as u64;
        match log_arity {
            1 => {
                // The points are `s` and `-s`, so `p(beta) = (y0 + y1) / 2 + (y0 - y1) * beta / (2s)`.
                let (lo, hi) = (evals[0], evals[1]);
                let half_inv_start = generator.exp_u64(power).double().inverse();
                (lo + hi).halve() + (lo - hi) * point * half_inv_start
            }
            2 | 3 => {
                // The inverse FFT gives the coefficients of `arity * p(s X)`.
                //
                // Horner at `beta / s`, then division by the arity, gives `p(beta)` with no extension inversion.
                let start_inv = 1u64
                    .checked_shl((log_height + log_arity) as u32)
                    .map_or_else(
                        || generator.exp_u64(power).inverse(),
                        |order| generator.exp_u64(order - power),
                    );
                let x = point * start_inv;
                if log_arity == 2 {
                    let [c0, c1, c2, c3] = ifft4(
                        [evals[0], evals[1], evals[2], evals[3]],
                        -Val::two_adic_generator(2),
                    );
                    [c0, c1, c2].into_iter().horner_acc(c3, x) * Val::ONE.halve().halve()
                } else {
                    let row: [Challenge; 8] = core::array::from_fn(|i| evals[i]);
                    let [c0, c1, c2, c3, c4, c5, c6, c7] = ifft8::<Val, Challenge>(row);
                    [c0, c1, c2, c3, c4, c5, c6].into_iter().horner_acc(c7, x)
                        * Val::ONE.halve().halve().halve()
                }
            }
            _ => crate::two_adic::fold_coset_row_by_halving(
                self, index, log_height, log_arity, point, evals,
            ),
        }
    }

    #[inline]
    fn index_power(&mut self, generator: Val, index: &usize, _bits: usize) -> Val {
        generator.exp_u64(*index as u64)
    }

    #[inline]
    fn constant_index(&mut self, value: usize) -> usize {
        value
    }

    #[inline]
    fn index_with_high(&mut self, low: &usize, low_bits: usize, high: usize) -> usize {
        (high << low_bits) | low
    }

    #[inline]
    fn eval_multilinear(&mut self, evals: &[Val], point: &Point<Challenge>) -> Challenge {
        Poly::new(evals).eval_base(point)
    }

    #[inline]
    fn eval_multilinear_ext(&mut self, evals: &[Challenge], point: &Point<Challenge>) -> Challenge {
        Poly::new(evals).eval_ext::<Val>(point)
    }

    fn insert_at(
        &mut self,
        index: &usize,
        log_len: usize,
        value: Challenge,
        siblings: &[Challenge],
    ) -> Vec<Challenge> {
        let mut row = Vec::with_capacity(siblings.len() + 1);
        row.extend_from_slice(siblings);
        row.insert(index & ((1 << log_len) - 1), value);
        row
    }

    #[inline]
    fn select(&mut self, bit: &bool, then: Challenge, otherwise: Challenge) -> Challenge {
        if *bit { then } else { otherwise }
    }

    #[inline]
    fn select_base(&mut self, bit: &bool, then: Val, otherwise: Val) -> Val {
        if *bit { then } else { otherwise }
    }

    fn batch_inverse<E>(
        &mut self,
        values: &[Challenge],
        zero: impl FnOnce(usize) -> E,
    ) -> Result<Vec<Challenge>, E> {
        if let Some(position) = values.iter().position(Field::is_zero) {
            return Err(zero(position));
        }
        Ok(batch_multiplicative_inverse(values))
    }

    #[inline]
    fn assert_zero<E>(&mut self, value: Challenge, err: impl FnOnce() -> E) -> Result<(), E> {
        if value.is_zero() { Ok(()) } else { Err(err()) }
    }

    fn coefficients(&mut self, value: &Challenge) -> Vec<Val> {
        value.as_basis_coefficients_slice().to_vec()
    }
}

impl<Val: Field, Challenge: ExtensionField<Val>> Lift for Native<Val, Challenge> {
    #[inline]
    fn load(&mut self, value: Val) -> Val {
        value
    }

    #[inline]
    fn load_ext(&mut self, value: Challenge) -> Challenge {
        value
    }

    #[inline]
    fn load_slice<'a>(&mut self, values: &'a [Val]) -> Cow<'a, [Val]> {
        Cow::Borrowed(values)
    }

    #[inline]
    fn load_ext_slice<'a>(&mut self, values: &'a [Challenge]) -> Cow<'a, [Challenge]> {
        Cow::Borrowed(values)
    }

    #[inline]
    fn value(&self, value: &Val) -> Val {
        *value
    }

    #[inline]
    fn value_ext(&self, value: &Challenge) -> Challenge {
        *value
    }

    #[inline]
    fn will_observe(&mut self, _units: Vec<Val>) {}

    #[inline]
    fn will_observe_ext(&mut self, _value: &Challenge) {}

    #[inline]
    fn sampled_ext(&mut self, value: Challenge) -> Challenge {
        value
    }

    #[inline]
    fn sampled_index(&mut self, value: usize, _bits: usize) -> usize {
        value
    }
}

/// Unnormalized inverse FFT of four evaluations in bit-reversed order.
///
/// The inverse fourth root must belong to this transform's domain.
#[inline]
fn ifft4<F: TwoAdicField, EF: ExtensionField<F>>(
    [y0, y2, y1, y3]: [EF; 4],
    root_inv: F,
) -> [EF; 4] {
    let a0 = y0 + y2;
    let a1 = y0 - y2;
    let a2 = y1 + y3;
    let a3 = (y1 - y3) * root_inv;
    [a0 + a2, a1 + a3, a0 - a2, a1 - a3]
}

/// Unnormalized inverse FFT of eight evaluations in bit-reversed order.
#[inline]
fn ifft8<F: TwoAdicField, EF: ExtensionField<F>>(
    [y0, y4, y2, y6, y1, y5, y3, y7]: [EF; 8],
) -> [EF; 8] {
    let root = F::two_adic_generator(3);
    let root_squared = root.square();
    // Every twiddle comes from this domain's root.
    //
    // A field need not choose its fourth root to be the square of its eighth root.
    let fourth_root_inv = -root_squared;
    let [b0, b1, b2, b3] = ifft4([y0, y4, y2, y6], fourth_root_inv);
    let [b4, b5, b6, b7] = ifft4([y1, y5, y3, y7], fourth_root_inv);
    let t1 = b5 * -(root_squared * root);
    let t2 = b6 * fourth_root_inv;
    let t3 = b7 * -root;
    [
        b0 + b4,
        b1 + t1,
        b2 + t2,
        b3 + t3,
        b0 - b4,
        b1 - t1,
        b2 - t2,
        b3 - t3,
    ]
}
