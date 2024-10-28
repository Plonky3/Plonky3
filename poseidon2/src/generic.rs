//! Whilst high speed implementations of Poseidon2 rely on a detailed understanding of the underlying field structure
//! it is also useful to have a generic constructor which works for a much larger range of fields.
//!
//! Indeed, for a fixed field F, the Poseidon2 permutation consists of three basic operations:
//! - Addition by elements in F.
//! - A power map x -> x^n.
//! - Multiplication by an F valued matrix.
//!
//! This means that it is possible to define a Poseidon2 over any abstract field AF which has implementations of:
//! - Add<F, Output = AF>
//! - Mul<F, Output = AF>
//!
//! This file implements the two matrix multiplications methods from which Poseidon2 can be built.

use p3_field::AbstractField;

use crate::{mds_light_permutation, MDSMat4};

/// A generic method performing the transformation:
///
/// `s -> (s + rc)^D`
///
/// This is a little slower than field specific implementations (particularly for packed fields) so should
/// only be used in non performance critical places.
#[inline(always)]
pub fn add_rc_and_sbox_generic<AF: AbstractField, const D: u64>(val: &mut AF, rc: AF::F) {
    *val += AF::from_f(rc);
    *val = val.exp_const_u64::<D>();
}

pub trait GenericPoseidon2LinearLayers<AF: AbstractField, const WIDTH: usize>: Sync {
    /// A generic implementation of the internal linear layer.
    fn internal_linear_layer(state: &mut [AF; WIDTH]);

    /// A generic implementation of the external linear layer.
    fn external_linear_layer(state: &mut [AF; WIDTH]) {
        mds_light_permutation(state, &MDSMat4);
    }
}
