use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{ExtensionField, Field, batch_multiplicative_inverse};
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
