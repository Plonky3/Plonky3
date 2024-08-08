// We split the internal and external constant traits as the external constant method depends on the WIDTH
// but the internal one does not.

/// Data needed to generate constants for the internal rounds of the Poseidon2 permutation.
pub trait Poseidon2InternalPackedConstants<F>: Sync + Clone {
    // In the scalar case this is AF::F but it may be different for PackedFields.
    type InternalConstantsType: Clone + core::fmt::Debug + Sync;

    fn manipulate_internal_constants(internal_constants: &F) -> Self::InternalConstantsType;
}

/// Data needed to generate constants for the external rounds of the Poseidon2 permutation.
pub trait Poseidon2ExternalPackedConstants<F, const WIDTH: usize>: Sync + Clone {
    // In the scalar case, ExternalConstantsType = [AF::F; WIDTH] but it may be different for PackedFields.
    type ExternalConstantsType: Clone + core::fmt::Debug + Sync;

    fn manipulate_external_constants(
        external_constants: &[F; WIDTH],
    ) -> Self::ExternalConstantsType;
}

#[derive(Debug, Clone, Default)]
pub struct NoPackedImplementation;

impl<F> Poseidon2InternalPackedConstants<F> for NoPackedImplementation {
    // In the scalar case this is AF::F but it may be different for PackedFields.
    type InternalConstantsType = ();

    fn manipulate_internal_constants(_internal_constants: &F) -> Self::InternalConstantsType {}
}

impl<F, const WIDTH: usize> Poseidon2ExternalPackedConstants<F, WIDTH> for NoPackedImplementation {
    // In the scalar case this is AF::F but it may be different for PackedFields.
    type ExternalConstantsType = ();

    fn manipulate_external_constants(
        _external_constants: &[F; WIDTH],
    ) -> Self::ExternalConstantsType {
    }
}
