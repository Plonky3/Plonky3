use crate::BrakedownCode;
use alloc::boxed::Box;
use p3_code::SystematicLinearCode;
use p3_field::Field;
use p3_matrix::sparse::CsrMatrix;
use p3_matrix::MatrixRows;
use rand::distributions::{Distribution, Standard};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

macro_rules! brakedown {
    ($a_width:literal, $a_height:literal, $a_density:literal,
     $b_width:literal, $b_height:literal, $b_density:literal,
     $inner_code:expr) => {{
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        // TODO: Should actually by fixed column weight.
        let a = CsrMatrix::<F>::rand_fixed_row_weight(&mut rng, $a_height, $a_width, $a_density);
        let b = CsrMatrix::<F>::rand_fixed_row_weight(&mut rng, $b_height, $b_width, $b_density);
        let inner_code = Box::new($inner_code);
        BrakedownCode { a, b, inner_code }
    }};
}

macro_rules! brakedown_to_rs {
    ($a_width:literal, $a_height:literal, $a_density:literal,
     $b_width:literal, $b_height:literal, $b_density:literal) => {
        brakedown!(
            $a_width,
            $a_height,
            $a_density,
            $b_width,
            $b_height,
            $b_density,
            p3_reed_solomon::UndefinedReedSolomonCode::new(
                p3_lde::NaiveUndefinedLDE,
                $b_width,
                $a_height
            )
        )
    };
}

#[rustfmt::skip]
pub fn fast_height_14<F, In>() -> impl SystematicLinearCode<F, In>
where
    F: Field,
    Standard: Distribution<F>,
    In: for<'a> MatrixRows<'a, F> + Sync,
{
    // TODO: These numbers aren't 100% correct...
    brakedown!(16384, 1967, 8, 2812, 4213, 20,
        brakedown!(1967, 237, 9, 339, 506, 23,
            brakedown!(237, 29, 11, 41, 61, 15,
                brakedown_to_rs!(29, 4, 1, 4, 8, 1))))
}
