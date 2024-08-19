// We split the internal and external constant traits as the external constant method depends on the WIDTH
// but the internal one does not.

/// Data needed to generate constants for the internal rounds of the Poseidon2 permutation.
pub trait Poseidon2InternalPackedConstants<F>: Sync + Clone {
    // In the scalar case this is AF::F but it may be different for PackedFields.
    type ConstantsType: Clone + core::fmt::Debug + Sync;

    fn convert_from_field(internal_constants: &F) -> Self::ConstantsType;
}

/// Data needed to generate constants for the external rounds of the Poseidon2 permutation.
pub trait Poseidon2ExternalPackedConstants<F, const WIDTH: usize>: Sync + Clone {
    // In the scalar case, ExternalConstantsType = [AF::F; WIDTH] but it may be different for PackedFields.
    type ConstantsType: Clone + core::fmt::Debug + Sync;

    fn convert_from_field_array(external_constants: &[F; WIDTH]) -> Self::ConstantsType;
}

/// We prove a simple option for fields which do not have a specialised Poseidon2 Packed implementation.
/// Any field which implements NoPackedImplementation automatically gets a trivial constants implementation.
pub trait NoPackedImplementation: Sync + Clone {}

impl<F, NoPacking: NoPackedImplementation> Poseidon2InternalPackedConstants<F> for NoPacking {
    // If there is no specialised implementation, the Internal Packed Constants will never be read.
    // So we set it to the empty type.
    type ConstantsType = ();

    fn convert_from_field(_internal_constants: &F) -> Self::ConstantsType {}
}

impl<F, const WIDTH: usize, NoPacking: NoPackedImplementation>
    Poseidon2ExternalPackedConstants<F, WIDTH> for NoPacking
{
    // If there is no specialised implementation, the External Packed Constants will never be read.
    // So we set it to the empty type.
    type ConstantsType = ();

    fn convert_from_field_array(_external_constants: &[F; WIDTH]) -> Self::ConstantsType {}
}
