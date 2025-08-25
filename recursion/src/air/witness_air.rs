use std::borrow::{Borrow, BorrowMut};

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

#[repr(C)]
/// Represents a columns of the witness or the Const Chip
pub struct RomCols<F> {
    address: F,
    value: F,
}

pub type WitnessCols<F> = RomCols<F>;
pub type ConstCols<F> = RomCols<F>;

impl<F> Borrow<RomCols<F>> for [F] {
    fn borrow(&self) -> &RomCols<F> {
        debug_assert_eq!(self.len(), 2);
        let (prefix, shorts, _sufix) = unsafe { self.align_to::<RomCols<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<F> BorrowMut<RomCols<F>> for [F] {
    fn borrow_mut(&mut self) -> &mut RomCols<F> {
        debug_assert_eq!(self.len(), 2);
        let (prefix, shorts, _sufix) = unsafe { self.align_to_mut::<RomCols<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

/// Represents an event in the ROM chip.
pub struct RomAirEvent<F> {
    address: usize,
    value: F,
}

/// Represents the witness AIR for the ROM chip.
pub struct RomAir {}

pub type WitnessAir = RomAir;
pub type ConstAir = RomAir;

impl RomAir {
    pub fn build_trace<F: Field>(events: &[RomAirEvent<F>]) -> RowMajorMatrix<F> {
        let n = events.len();
        let n_padded = n.next_power_of_two();
        let mut trace = RowMajorMatrix::new(F::zero_vec(n_padded * 2), 2);

        let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<RomCols<F>>() };
        assert!(prefix.is_empty(), "Alignment should match");
        assert!(suffix.is_empty(), "Alignment should match");
        assert_eq!(rows.len(), n);

        for (i, event) in events.iter().enumerate() {
            rows[i].address = F::from_usize(event.address);
            rows[i].value = event.value;
        }

        trace
    }
}

impl<F> BaseAir<F> for RomAir {
    fn width(&self) -> usize {
        2
    }
}

impl<AB: AirBuilder> Air<AB> for RomAir {
    fn eval(&self, _builder: &mut AB) {}
}
