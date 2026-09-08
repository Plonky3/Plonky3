pub use p3_baby_bear::BabyBear as Val;
use p3_baby_bear::{
    Poseidon2BabyBear, default_babybear_poseidon2_16, default_babybear_poseidon2_24,
};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
pub use p3_fri::FriParameters;
use p3_fri::TwoAdicFriPcs;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;

/// Quartic extension used for challenges.
pub type Challenge = BinomialExtensionField<Val, 4>;
/// Field Merkle tree with width-24 hashing and width-16 compression.
pub type Mmcs = MerkleTreeMmcs<
    <Val as Field>::Packing,
    <Val as Field>::Packing,
    PaddingFreeSponge<Poseidon2BabyBear<24>, 24, 16, 8>,
    TruncatedPermutation<Poseidon2BabyBear<16>, 2, 8, 16>,
    2,
    8,
>;
/// Extension-field Merkle tree used by FRI.
pub type ChallengeMmcs = ExtensionMmcs<Val, Challenge, Mmcs>;
/// Width-24, rate-16 Poseidon2 transcript.
pub type Challenger = DuplexChallenger<Val, Poseidon2BabyBear<24>, 24, 16>;
/// Non-hiding two-adic FRI commitment scheme.
pub type Pcs = TwoAdicFriPcs<Val, Radix2DitParallel<Val>, Mmcs, ChallengeMmcs>;
/// Concrete configuration accepted by [`crate::uni_stark`].
pub type Config = StarkConfig<Pcs, Challenge, Challenger>;

/// Assemble the fixed published Poseidon2 stack with explicit FRI parameters.
///
/// Both Merkle trees use `cap_height`. `params.mmcs` is `()` because this
/// constructor supplies the commitment schemes; all other fields are preserved.
/// No FRI/security defaults are selected. Configure STARK-level grinding with
/// [`Config::with_ood_proof_of_work_bits`] and
/// [`Config::with_lookup_proof_of_work_bits`] on the returned configuration.
///
/// The field crate's `default_babybear_poseidon2_16/24` constructors use
/// checked-in Grain-LFSR round constants, with no runtime randomness. Each
/// prove/verify call starts from a clone of the initial challenger.
///
/// Like [`TwoAdicFriPcs::new`], this only assembles the configuration. Invalid
/// FRI parameters or incompatible trace sizes may panic during proving or
/// fail verification. The caller must assess the full proof protocol.
#[must_use]
pub fn new(params: FriParameters<()>, cap_height: usize) -> Config {
    let perm16 = default_babybear_poseidon2_16();
    let perm24 = default_babybear_poseidon2_24();
    let mmcs = Mmcs::new(
        PaddingFreeSponge::new(perm24.clone()),
        TruncatedPermutation::new(perm16),
        cap_height,
    );
    let fri = crate::prime::with_mmcs(params, ChallengeMmcs::new(mmcs.clone()));
    let pcs = Pcs::new(Radix2DitParallel::default(), mmcs, fri);
    Config::new(pcs, Challenger::new(perm24))
}
