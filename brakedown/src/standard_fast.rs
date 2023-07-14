use alloc::boxed::Box;
use alloc::vec;

use p3_code::{LinearCodeFamily, SLCodeRegistry};
use p3_field::Field;
use p3_matrix::sparse::CsrMatrix;
use p3_matrix::MatrixRows;
use rand::distributions::{Distribution, Standard};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::macros::{brakedown, brakedown_to_rs};
use crate::BrakedownCode;

pub fn fast_registry<F, In>() -> impl LinearCodeFamily<F, In>
where
    F: Field,
    Standard: Distribution<F>,
    In: for<'a> MatrixRows<'a, F> + Sync,
{
    #[rustfmt::skip]
    let height_14 = brakedown!(16384, 1967, 8, 2810, 4211, 20,
        brakedown!(1967, 237, 9, 338, 505, 23,
            brakedown!(237, 29, 11, 41, 60, 15,
                brakedown_to_rs!(29, 4, 0, 5, 7, 0))));

    #[rustfmt::skip]
    let height_16 = brakedown!(65536, 7865, 8, 11235, 16851, 19,
        brakedown!(7865, 944, 8, 1348, 2022, 21,
            brakedown!(944, 114, 9, 162, 242, 23,
                brakedown_to_rs!(114, 14, 7, 20, 28, 11))));

    SLCodeRegistry::new(vec![Box::new(height_14), Box::new(height_16)])
}
