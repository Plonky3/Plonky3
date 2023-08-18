//! Utilities for testing field implementations.

#![no_std]

extern crate alloc;

use p3_field::Field;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

pub fn test_inverse<F: Field>()
where
    Standard: Distribution<F>,
{
    assert_eq!(None, F::ZERO.try_inverse());

    assert_eq!(Some(F::ONE), F::ONE.try_inverse());

    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        let x = rng.gen::<F>();
        if !x.is_zero() && !x.is_one() {
            let z = x.inverse();
            assert_ne!(x, z);
            assert_eq!(x * z, F::ONE);
        }
    }
}
