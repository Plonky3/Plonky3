//! Utilities for testing field implementations.

#![no_std]

extern crate alloc;

use p3_field::{
    cyclic_subgroup_coset_known_order, cyclic_subgroup_known_order, two_adic_coset_zerofier,
    two_adic_subgroup_zerofier, Field, TwoAdicField,
};
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

pub fn test_two_adic_subgroup_zerofier<F: TwoAdicField>() {
    for log_n in 0..5 {
        let g = F::primitive_root_of_unity(log_n);
        for x in cyclic_subgroup_known_order(g, 1 << log_n) {
            let zerofier_eval = two_adic_subgroup_zerofier(log_n, x);
            assert_eq!(zerofier_eval, F::ZERO);
        }
    }
}

pub fn test_two_adic_coset_zerofier<F: TwoAdicField>() {
    for log_n in 0..5 {
        let g = F::primitive_root_of_unity(log_n);
        let shift = F::multiplicative_group_generator();
        for x in cyclic_subgroup_coset_known_order(g, shift, 1 << log_n) {
            let zerofier_eval = two_adic_coset_zerofier(log_n, shift, x);
            assert_eq!(zerofier_eval, F::ZERO);
        }
    }
}
