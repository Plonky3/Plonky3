//! The convert by 256-entry tables, one per medium position: the reference.

use alloc::vec::Vec;

use p3_binary_field::Poly192;
use p3_field::PrimeCharacteristicRing;

use crate::skip::{SKIP_SIZE, phi8};
use crate::zerocheck::fixed::{MEDIUM_VALUES, gamma_powers};

/// `g^b phi8(v)` for every medium position `b` and byte `v`.
pub(crate) struct ConvertTables(Vec<[Poly192; 256]>);

impl ConvertTables {
    /// The tables, built once per sweep.
    pub(crate) fn new() -> Self {
        Self(
            gamma_powers()
                .iter()
                .map(|&g| core::array::from_fn(|v| g * phi8(v as u8)))
                .collect(),
        )
    }
}

/// One worker's per-lane sums for `a b` and for `c`.
pub(crate) struct Convert {
    /// The `a b` sums.
    ab: [Poly192; SKIP_SIZE],
    /// The `c` sums.
    c: [Poly192; SKIP_SIZE],
}

impl Convert {
    /// Empty sums.
    pub(crate) const fn new() -> Self {
        Self {
            ab: [Poly192::ZERO; SKIP_SIZE],
            c: [Poly192::ZERO; SKIP_SIZE],
        }
    }

    /// Add one window's medium rows at weight `eq_lo`.
    #[inline(always)]
    pub(crate) fn accumulate(
        &mut self,
        tables: &ConvertTables,
        ab: &[[u8; 64]],
        c: &[[u8; 64]],
        eq_lo: Poly192,
    ) {
        debug_assert!(ab.len() <= MEDIUM_VALUES);
        for lane in 0..SKIP_SIZE {
            let mut cf_ab = Poly192::ZERO;
            let mut cf_c = Poly192::ZERO;
            for ((row, ab), c) in tables.0.iter().zip(ab).zip(c) {
                cf_ab += row[usize::from(ab[lane])];
                cf_c += row[usize::from(c[lane])];
            }
            self.ab[lane] += cf_ab * eq_lo;
            self.c[lane] += cf_c * eq_lo;
        }
    }

    /// The `a b` and `c` sums.
    pub(crate) const fn values(&self) -> ([Poly192; SKIP_SIZE], [Poly192; SKIP_SIZE]) {
        (self.ab, self.c)
    }
}
