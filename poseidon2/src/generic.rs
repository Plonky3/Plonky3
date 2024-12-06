//! Whilst high speed implementations of Poseidon2 rely on a detailed understanding of the underlying field structure
//! it is also useful to have a generic constructor which works for a much larger range of fields.
//!
//! Indeed, for a fixed field F, the Poseidon2 permutation consists of three basic operations:
//! - Addition by elements in F.
//! - A power map x -> x^n.
//! - Multiplication by an F valued matrix.
//!
//! This means that it is possible to define a Poseidon2 over any abstract field FA which has implementations of:
//! - Add<F, Output = FA>
//! - Mul<F, Output = FA>
//!
//! This file implements the two matrix multiplications methods from which Poseidon2 can be built.

use p3_field::{FieldAlgebra, InjectiveMonomial};

use crate::{mds_light_permutation, MDSMat4};

/// A generic method performing the transformation:
///
/// `s -> (s + rc)^D`
///
/// This is a little slower than field specific implementations (particularly for packed fields) so should
/// only be used in non performance critical places.
#[inline(always)]
pub fn add_rc_and_sbox_generic<FA: FieldAlgebra + InjectiveMonomial<D>, const D: u64>(
    val: &mut FA,
    rc: FA::F,
) {
    *val += FA::from_f(rc);
    *val = val.injective_exp_n();
}

pub trait GenericPoseidon2LinearLayers<FA: FieldAlgebra, const WIDTH: usize>: Sync {
    /// A generic implementation of the internal linear layer.
    fn internal_linear_layer(state: &mut [FA; WIDTH]);

    /// A generic implementation of the external linear layer.
    fn external_linear_layer(state: &mut [FA; WIDTH]) {
        mds_light_permutation(state, &MDSMat4);
    }
}
