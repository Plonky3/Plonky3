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

pub trait GenericPoseidon2LinearLayers<AF: AbstractField, const WIDTH: usize>: Sync {
    /// A generic implementation of the internal linear layer.
    fn internal_linear_layer(state: &mut [AF; WIDTH]);

    /// A generic implementation of the external linear layer.
    fn external_linear_layer(state: &mut [AF; WIDTH]) {
        // TODO: Not loving having to initialise mat4 every time. Would ideally be able to find a way around it
        // while still supporting the option for the user to change their choice of external layer matrix.
        // Want to avoid having to pass another argument to the function to.
        // Might involve refactoring mds_light_permutation.
        let mat4 = MDSMat4 {};

        mds_light_permutation(state, &mat4);
    }
}
