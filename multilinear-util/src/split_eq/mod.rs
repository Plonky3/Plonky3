pub mod accumulate;
pub mod compress;
pub mod eq;
pub mod eval;

use p3_field::{ExtensionField, Field, PackedValue};
use p3_util::log2_strict_usize;

use crate::evals::Poly;
use crate::multilinear::Point;

/// Eq polynomial table, either as scalar extension elements or packed representation.
#[derive(Debug, Clone)]
pub(super) enum EqMaybePacked<F: Field, EF: ExtensionField<F>> {
    Unpacked(Poly<EF>),
    Packed(Poly<<EF as ExtensionField<F>>::ExtensionPacking>),
}

impl<F: Field, EF: ExtensionField<F>> EqMaybePacked<F, EF> {
    /// Constructs an unpacked eq table.
    pub(super) fn new_unpacked(point: &Point<EF>) -> Self {
        Self::Unpacked(Poly::new_from_point(point.as_slice(), EF::ONE))
    }

    /// Constructs a packed eq table when possible, otherwise falls back to unpacked.
    pub(super) fn new_packed(point: &Point<EF>) -> Self {
        if point.num_vars() >= log2_strict_usize(F::Packing::WIDTH) {
            Self::Packed(Poly::new_packed_from_point(point.as_slice(), EF::ONE))
        } else {
            Self::Unpacked(Poly::new_from_point(point.as_slice(), EF::ONE))
        }
    }

    /// Returns the number of variables.
    pub(super) const fn num_vars(&self) -> usize {
        match self {
            Self::Unpacked(poly) => poly.num_vars(),
            Self::Packed(poly) => poly.num_vars() + log2_strict_usize(F::Packing::WIDTH),
        }
    }
}

/// Factored eq polynomial table for `scale · eq(z, ·)`. Splits point `z` at the
/// midpoint into `(z_lo, z_hi)` and stores `scale · eq(z_lo, ·)` and `eq(z_hi, ·)`
/// separately, exploiting the identity `eq(z, x) = eq(z_lo, x_lo) · eq(z_hi, x_hi)`.
/// This avoids materializing the full `2^k` table, using `2 · 2^{k/2}` space instead.
#[derive(Debug, Clone)]
pub struct SplitEq<F: Field, EF: ExtensionField<F>> {
    /// Eq table for the low-half variables `z_lo`.
    pub(super) eq0: Poly<EF>,
    /// Eq table for the high-half variables `z_hi`.
    pub(super) eq1: EqMaybePacked<F, EF>,
}

impl<F: Field, EF: ExtensionField<F>> SplitEq<F, EF> {
    /// Constructs a `SplitEq` with unpacked eq table for the low half.
    pub fn new_unpacked(point: &Point<EF>, scale: EF) -> Self {
        let (z0, z1) = point.split_at(point.num_vars() / 2);
        let eq0 = Poly::new_from_point(z0.as_slice(), scale);
        let eq1 = EqMaybePacked::new_unpacked(&z1);
        Self { eq0, eq1 }
    }

    /// Constructs a `SplitEq`, using packed field representation for the low half.
    /// Falls back to unpacked if the low half has fewer variables than the packing width.
    pub fn new_packed(point: &Point<EF>, scale: EF) -> Self {
        let (z0, z1) = point.split_at(point.num_vars() / 2);
        let eq0 = Poly::new_from_point(z0.as_slice(), scale);
        let eq1 = EqMaybePacked::new_packed(&z1);
        Self { eq0, eq1 }
    }

    /// Returns the number of variables.
    pub const fn num_vars(&self) -> usize {
        self.eq0.num_vars() + self.eq1.num_vars()
    }
}
