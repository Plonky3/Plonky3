// TODO: Split into Internal and External. Internal doesn't need the WIDTH.
pub trait Poseidon2PackedTypesAndConstants<F, const WIDTH: usize>: Sync + Clone {
    // In the scalar case this is AF::F but it may be different for PackedFields.
    type InternalConstantsType: Clone + core::fmt::Debug + Sync;

    fn manipulate_internal_constants(internal_constants: &F) -> Self::InternalConstantsType;

    // In the scalar case, ExternalConstantsType = [AF::F; WIDTH] but it may be different for PackedFields.
    type ExternalConstantsType: Clone + core::fmt::Debug + Sync;

    fn manipulate_external_constants(
        external_constants: &[F; WIDTH],
    ) -> Self::ExternalConstantsType;
}
