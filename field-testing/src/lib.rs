//! Utilities for testing field implementations.

#![no_std]

extern crate alloc;

use p3_field::Field;

pub fn test_inverse<F: Field>() {
    assert_eq!(None, F::ZERO.try_inverse());

    assert_eq!(Some(F::ONE), F::ONE.try_inverse());

    let r = F::from_canonical_u8(5);
    assert_eq!(F::ONE, r.try_inverse().unwrap() * r);
}
