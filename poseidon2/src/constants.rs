use alloc::vec::Vec;
// We split the internal and external constant traits as the external constant method depends on the WIDTH
// but the internal one does not.

/// Data needed to generate constants for the internal rounds of the Poseidon2 permutation.
pub trait Poseidon2InternalPackedConstants<F>: Sync + Clone {
    fn convert_from_field(internal_constants: Vec<F>) -> Self;
}

/// Data needed to generate constants for the external rounds of the Poseidon2 permutation.
pub trait Poseidon2ExternalPackedConstants<F, const WIDTH: usize>: Sync + Clone {
    fn convert_from_field_array(external_constants: [Vec<[F; WIDTH]>; 2]) -> Self;
}
