use p3_field::Field;

/// Apply the doubling map operation on x coordinate: x.square().double() - F::ONE
/// This is a helper method used commonly in circle computations
#[inline(always)]
pub fn doubling_map<F: Field>(x: F) -> F {
    x.square().double() - F::ONE
} 
