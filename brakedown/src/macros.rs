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

pub(crate) use brakedown;
pub(crate) use brakedown_to_rs;
