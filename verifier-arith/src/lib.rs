//! The arithmetic a verifier is written over.
//!
//! A verifier generic over [`Arith`] runs natively on field elements through [`Native`].
//!
//! The same code also runs inside a circuit, where every value is a wire.

#![no_std]

extern crate alloc;

mod native;
pub mod two_adic;

use alloc::borrow::Cow;
use alloc::vec::Vec;

use p3_field::{Algebra, BasedVectorSpace, Dup, ExtensionField, Field};

pub use crate::native::Native;
pub use crate::two_adic::LagrangeSelectors;

/// Values and operations a verifier computes with.
///
/// Ring operations come from the value types.
///
/// Everything else a verifier needs is a method here.
pub trait Arith {
    /// Base field of the verified proofs.
    type Val: Field;

    /// Challenge field of the verified proofs.
    type Challenge: ExtensionField<Self::Val>;

    /// A base-field value.
    type F: Algebra<Self::Val>;

    /// A challenge-field value.
    type EF: Algebra<Self::F> + Algebra<Self::Challenge>;

    /// A value known to be zero or one.
    type Bit;

    /// A query index into a two-adic domain.
    type Index: Clone;

    /// Embed a bit as zero or one.
    fn bit(&self, bit: &Self::Bit) -> Self::F;

    /// The low `count` bits of an index, least significant first.
    fn index_bits(&mut self, index: &Self::Index, count: usize) -> Vec<Self::Bit>;

    /// An index shifted right by `shift` bits.
    fn index_shr(&self, index: &Self::Index, shift: usize) -> Self::Index;

    /// `then` when the bit is one, `otherwise` when it is zero.
    fn select(&mut self, bit: &Self::Bit, then: Self::EF, otherwise: Self::EF) -> Self::EF {
        let bit = self.bit(bit);
        otherwise.dup() + (then - otherwise) * bit
    }

    /// `then` when the bit is one, `otherwise` when it is zero.
    fn select_base(&mut self, bit: &Self::Bit, then: Self::F, otherwise: Self::F) -> Self::F {
        let bit = self.bit(bit);
        otherwise.dup() + (then - otherwise) * bit
    }

    /// `generator^rev(index)`, where `rev` reverses the low `bits` bits of the index.
    fn reversed_index_power(
        &mut self,
        generator: Self::Val,
        index: &Self::Index,
        bits: usize,
    ) -> Self::F {
        two_adic::reversed_index_power_by_bits(self, generator, index, bits)
    }

    /// The row holding `value` at the position the low `log_len` index bits name.
    ///
    /// The siblings fill the other positions, in order.
    fn insert_at(
        &mut self,
        index: &Self::Index,
        log_len: usize,
        value: Self::EF,
        siblings: &[Self::EF],
    ) -> Vec<Self::EF>
    where
        Self::EF: Clone,
    {
        let bits = self.index_bits(index, log_len);
        insert_by_bits(self, &bits, value, siblings)
    }

    /// The inverse of every value.
    ///
    /// # Errors
    ///
    /// The position of a zero value, mapped through `zero`.
    fn batch_inverse<E>(
        &mut self,
        values: &[Self::EF],
        zero: impl FnOnce(usize) -> E,
    ) -> Result<Vec<Self::EF>, E>;

    /// Require a value to be zero.
    ///
    /// # Errors
    ///
    /// `err()` when the value is not zero.
    fn assert_zero<E>(&mut self, value: Self::EF, err: impl FnOnce() -> E) -> Result<(), E>;

    /// The basis coefficients of a challenge-field value.
    fn coefficients(&mut self, value: &Self::EF) -> Vec<Self::F>;
}

/// Moves native proof data and native transcript outputs into an arithmetic.
///
/// The native transcript stays the single description of absorb and squeeze order.
///
/// A circuit drives it on a challenger that mirrors every step onto wires.
pub trait Lift: Arith {
    /// A proof value.
    fn load(&mut self, value: Self::Val) -> Self::F;

    /// A proof value in the challenge field.
    fn load_ext(&mut self, value: Self::Challenge) -> Self::EF;

    /// Proof values, borrowed when no lifting is needed.
    fn load_slice<'a>(&mut self, values: &'a [Self::Val]) -> Cow<'a, [Self::F]>
    where
        Self::F: Clone,
    {
        Cow::Owned(values.iter().map(|&value| self.load(value)).collect())
    }

    /// Proof values in the challenge field, borrowed when no lifting is needed.
    fn load_ext_slice<'a>(&mut self, values: &'a [Self::Challenge]) -> Cow<'a, [Self::EF]>
    where
        Self::EF: Clone,
    {
        Cow::Owned(values.iter().map(|&value| self.load_ext(value)).collect())
    }

    /// The native value behind a base-field value.
    fn value(&self, value: &Self::F) -> Self::Val;

    /// The native value behind a challenge-field value.
    fn value_ext(&self, value: &Self::EF) -> Self::Challenge;

    /// Construct a native transcript, whose seed the arithmetic absorbs as constants.
    ///
    /// A transcript seeds itself from its shape alone, so its seed is never a proof value.
    fn seeding<T>(&mut self, construct: impl FnOnce() -> T) -> T {
        construct()
    }

    /// Announce the values the native transcript absorbs next, in absorb order.
    fn will_observe(&mut self, units: Vec<Self::F>);

    /// A challenge-field sample the native transcript just drew.
    fn sampled_ext(&mut self, value: Self::Challenge) -> Self::EF;

    /// A query index of `bits` bits the native transcript just drew.
    fn sampled_index(&mut self, value: usize, bits: usize) -> Self::Index;

    /// Announce a challenge-field value the native transcript absorbs next.
    fn will_observe_ext(&mut self, value: &Self::EF) {
        let units = self.coefficients(value);
        self.will_observe(units);
    }
}

/// The value whose basis coefficients are the given challenge-field values.
///
/// An extension column committed coefficient by coefficient opens as one value per coefficient.
///
/// # Panics
///
/// When the slice does not hold exactly one value per basis element.
pub fn recompose_ext<A: Arith>(coefficients: &[A::EF]) -> A::EF {
    assert_eq!(coefficients.len(), A::Challenge::DIMENSION);
    coefficients
        .iter()
        .enumerate()
        .map(|(i, c)| c.dup() * A::Challenge::ith_basis_element(i).expect("i < DIMENSION"))
        .sum()
}

/// The row holding `value` at the position `bits` name, and the siblings elsewhere in order.
///
/// The most significant bit decides which half the value lands in.
///
/// The other half is a run of siblings, and the value's half is placed recursively.
pub fn insert_by_bits<A: Arith + ?Sized>(
    arith: &mut A,
    bits: &[A::Bit],
    value: A::EF,
    siblings: &[A::EF],
) -> Vec<A::EF>
where
    A::EF: Clone,
{
    let Some((top, low)) = bits.split_last() else {
        return alloc::vec![value];
    };
    let half = 1 << low.len();

    //     top = 1:  siblings[..half]              ++ place(siblings[half..])
    //     top = 0:  place(siblings[..half - 1])   ++ siblings[half - 1..]
    let mut upper_half = siblings[..half].to_vec();
    upper_half.extend(insert_by_bits(arith, low, value.clone(), &siblings[half..]));
    let mut lower_half = insert_by_bits(arith, low, value, &siblings[..half - 1]);
    lower_half.extend_from_slice(&siblings[half - 1..]);

    upper_half
        .into_iter()
        .zip(lower_half)
        .map(|(then, otherwise)| arith.select(top, then, otherwise))
        .collect()
}
