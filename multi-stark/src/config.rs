// No additional imports needed.

// Re-export the uni-stark generic config and useful associated types for convenience.
pub use p3_uni_stark::{
    Domain as UniDomain, PackedChallenge as UniPackedChallenge, PackedVal as UniPackedVal,
    StarkGenericConfig, Val as UniVal,
};

pub type Domain<SC> = UniDomain<SC>;
pub type Val<SC> = UniVal<SC>;
pub type PackedVal<SC> = UniPackedVal<SC>;
pub type PackedChallenge<SC> = UniPackedChallenge<SC>;

pub type PcsError<SC> = <<SC as p3_uni_stark::StarkGenericConfig>::Pcs as p3_commit::Pcs<
    <SC as p3_uni_stark::StarkGenericConfig>::Challenge,
    <SC as p3_uni_stark::StarkGenericConfig>::Challenger,
>>::Error;

// Marker trait aliasing uni-stark’s StarkGenericConfig for clarity in this crate.
pub trait MultiStarkGenericConfig: p3_uni_stark::StarkGenericConfig {}
impl<T: p3_uni_stark::StarkGenericConfig> MultiStarkGenericConfig for T {}
