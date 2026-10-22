//! The tables a circuit is proven with.
//!
//! Every table speaks on [`WIRE_BUS`], whose messages are `(wire, limb_0, .., limb_{D-1})`.
//!
//! Each wire has exactly one writer row, which provides the message as many times as it is read.
//!
//! A base-field wire's message carries zero in every limb after the first.

mod alu;
mod permute;
mod public;
mod recompose;
mod value;

use alloc::vec::Vec;

use p3_air::{Air, BaseAir, PermutationAirBuilder};
use p3_field::extension::{BinomialExtensionField, BinomiallyExtendable, binomial_mul};
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};
use p3_lookup::{Count, InteractionBuilder};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::SubAirBuilder;

pub use self::alu::AluAir;
pub use self::permute::PermuteAir;
pub use self::public::PublicAir;
pub use self::recompose::RecomposeAir;
pub use self::value::ValueAir;
use crate::PermutationChip;

/// Name of the bus every wire travels on.
pub const WIRE_BUS: &str = "wire";

/// A challenge field whose products a table constrains limb by limb.
pub trait CircuitExtension<F: Field>: ExtensionField<F> {
    /// The product of two values given by their basis coefficients, over any algebra of `F`.
    fn mul_limbs<R: Algebra<F>>(a: &[R], b: &[R]) -> Vec<R>;
}

impl<F: BinomiallyExtendable<D>, const D: usize> CircuitExtension<F>
    for BinomialExtensionField<F, D>
{
    fn mul_limbs<R: Algebra<F>>(a: &[R], b: &[R]) -> Vec<R> {
        let a: &[R; D] = a.try_into().expect("one limb per coefficient");
        let b: &[R; D] = b.try_into().expect("one limb per coefficient");
        let mut product = core::array::from_fn(|_| R::ZERO);
        binomial_mul::<F, R, R, D>(a, b, &mut product, F::W);
        product.to_vec()
    }
}

/// Send one wire message on the bus.
///
/// A base-field value is padded with zero limbs up to the challenge-field dimension.
pub(crate) fn send<AB: InteractionBuilder>(
    builder: &mut AB,
    id: AB::Expr,
    limbs: impl IntoIterator<Item = AB::Expr>,
    dimension: usize,
    count: Count<AB::Expr>,
) {
    builder.push_interaction(WIRE_BUS, message(id, limbs, dimension), count);
}

/// The fields of one wire message: the wire id, then its limbs padded with zeros.
pub(crate) fn message<E: PrimeCharacteristicRing>(
    id: E,
    limbs: impl IntoIterator<Item = E>,
    dimension: usize,
) -> Vec<E> {
    let mut fields: Vec<E> = Vec::with_capacity(dimension + 1);
    fields.push(id);
    fields.extend(limbs);
    fields.resize(dimension + 1, E::ZERO);
    fields
}

/// A row reading a wire, when its flag is set.
pub(crate) const fn read<E>(flag: E) -> Count<E> {
    Count::bounded(flag, 1)
}

/// A row writing a wire, once for every read.
pub(crate) fn write<E: core::ops::Neg<Output = E>>(reads: E) -> Count<E> {
    Count::provided(-reads)
}

/// One table of a circuit.
#[derive(Clone, Debug)]
pub enum CircuitAir<F: Field, EF, P: PermutationChip<F>> {
    /// Constants and prover-chosen values.
    Value(ValueAir<F>),
    /// Public inputs.
    Public(PublicAir<F>),
    /// `out = a * b + c` over the base field.
    BaseAlu(AluAir<F, EF>),
    /// `out = a * b + c` over the challenge field.
    ExtAlu(AluAir<F, EF>),
    /// Challenge-field wires assembled from, or split into, base-field limbs.
    Recompose(RecomposeAir<F>),
    /// Calls of the circuit's permutation.
    Permute(PermuteAir<F, P>),
}

impl<F: Field, EF: Sync, P: PermutationChip<F>> BaseAir<F> for CircuitAir<F, EF, P> {
    fn width(&self) -> usize {
        match self {
            Self::Value(air) => air.width(),
            Self::Public(air) => air.width(),
            Self::BaseAlu(air) | Self::ExtAlu(air) => air.width(),
            Self::Recompose(air) => air.width(),
            Self::Permute(air) => air.width(),
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        match self {
            Self::Value(air) => air.preprocessed_trace(),
            Self::Public(air) => air.preprocessed_trace(),
            Self::BaseAlu(air) | Self::ExtAlu(air) => air.preprocessed_trace(),
            Self::Recompose(air) => air.preprocessed_trace(),
            Self::Permute(air) => air.preprocessed_trace(),
        }
    }

    fn preprocessed_width(&self) -> usize {
        match self {
            Self::Value(air) => air.preprocessed_width(),
            Self::Public(air) => air.preprocessed_width(),
            Self::BaseAlu(air) | Self::ExtAlu(air) => air.preprocessed_width(),
            Self::Recompose(air) => air.preprocessed_width(),
            Self::Permute(air) => air.preprocessed_width(),
        }
    }

    fn num_public_values(&self) -> usize {
        match self {
            Self::Public(air) => air.num_public_values(),
            _ => 0,
        }
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl<AB, EF, P> Air<AB> for CircuitAir<AB::F, EF, P>
where
    AB: PermutationAirBuilder + InteractionBuilder,
    AB::F: Field,
    EF: CircuitExtension<AB::F>,
    P: PermutationChip<AB::F>,
    P::Air: for<'b> Air<SubAirBuilder<'b, AB, P::Air, AB::Var>>,
{
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::Value(air) => air.eval(builder),
            Self::Public(air) => air.eval(builder),
            Self::BaseAlu(air) | Self::ExtAlu(air) => air.eval(builder),
            Self::Recompose(air) => air.eval(builder),
            Self::Permute(air) => air.eval(builder),
        }
    }
}

/// A matrix of the given rows, padded with zero rows to a power-of-two height.
pub(crate) fn padded_matrix<F: Field>(mut values: Vec<F>, width: usize) -> RowMajorMatrix<F> {
    let height = (values.len() / width).max(1).next_power_of_two();
    values.resize(height * width, F::ZERO);
    RowMajorMatrix::new(values, width)
}
