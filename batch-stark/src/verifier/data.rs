use p3_air::Air;
use p3_challenger::GrindingChallenger;
use p3_lookup::logup::LogUpGadget;
use p3_lookup::{Lookup, LookupProtocol};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_uni_stark::{ConstraintInputs, StarkArith, VerifierConstraintFolder};
use p3_verifier_arith::{Arith, Native};

use crate::config::{StarkGenericConfig as SGC, Val};
use crate::folder::VerifierConstraintFolderWithLookups;

/// One instance's openings and lookup data, as its constraints read them at `zeta`.
pub struct VerifierData<'a, AR: Arith + ?Sized> {
    /// The openings every AIR constraint reads.
    pub inputs: ConstraintInputs<'a, AR>,
    /// Permutation columns at `zeta`, one extension element per column.
    pub permutation_local: &'a [AR::EF],
    /// Permutation columns at the point after `zeta`.
    pub permutation_next: &'a [AR::EF],
    /// Lookup challenges of this instance.
    pub permutation_challenges: &'a [AR::EF],
    /// Lookup terminal of this instance, if it declares lookups.
    pub permutation_values: &'a [AR::EF],
    /// Lookups this instance declares.
    pub lookups: &'a [Lookup<AR::Val>],
}

/// What the batch verifier delegates to its arithmetic, on top of [`StarkArith`].
pub trait BatchArith<SC: SGC, A>: StarkArith<SC> {
    /// The AIR's constraints and lookups at `zeta`, folded by `alpha`.
    fn fold_constraints(
        &mut self,
        air: &A,
        data: &VerifierData<'_, Self>,
        lookup_gadget: &LogUpGadget,
    ) -> Self::EF;
}

impl<SC, A> BatchArith<SC, A> for Native<Val<SC>, SC::Challenge>
where
    SC: SGC<Challenger: GrindingChallenger<Witness = Val<SC>>>,
    A: for<'b> Air<VerifierConstraintFolderWithLookups<'b, SC>>,
{
    fn fold_constraints(
        &mut self,
        air: &A,
        data: &VerifierData<'_, Self>,
        lookup_gadget: &LogUpGadget,
    ) -> SC::Challenge {
        let mut folder = VerifierConstraintFolderWithLookups {
            inner: VerifierConstraintFolder::new(&data.inputs),
            permutation: VerticalPair::new(
                RowMajorMatrixView::new_row(data.permutation_local),
                RowMajorMatrixView::new_row(data.permutation_next),
            ),
            permutation_challenges: data.permutation_challenges,
            permutation_values: data.permutation_values,
        };
        lookup_gadget.eval_air_and_lookups(air, &mut folder, data.lookups);
        folder.inner.accumulator
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::{ExtensionMmcs, Pcs, PolynomialSpace};
    use p3_dft::Radix2DitParallel;
    use p3_field::Field;
    use p3_field::extension::BinomialExtensionField;
    use p3_fri::{FriParameters, TwoAdicFriPcs};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_uni_stark::StarkConfig;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    // Standard two-adic test stack: BabyBear + Poseidon2 + FRI.
    type TestVal = BabyBear;
    type TestChallenge = BinomialExtensionField<TestVal, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs = MerkleTreeMmcs<
        <TestVal as Field>::Packing,
        <TestVal as Field>::Packing,
        MyHash,
        MyCompress,
        2,
        8,
    >;
    type ChallengeMmcs = ExtensionMmcs<TestVal, TestChallenge, ValMmcs>;
    type Challenger = DuplexChallenger<TestVal, Perm, 16, 8>;
    type Dft = Radix2DitParallel<TestVal>;
    type MyPcs = TwoAdicFriPcs<TestVal, Dft, ValMmcs, ChallengeMmcs>;
    type MyConfig = StarkConfig<MyPcs, TestChallenge, Challenger>;

    fn make_config() -> MyConfig {
        let mut rng = SmallRng::seed_from_u64(0);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let dft = Dft::default();
        let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
        let pcs = MyPcs::new(dft, val_mmcs, fri_params);
        let challenger = Challenger::new(perm);
        StarkConfig::new(pcs, challenger)
    }

    #[test]
    fn ood_point_inside_trace_domain_is_rejected_not_panicked() {
        let config = make_config();

        // A size-8 trace domain `gH`.
        let domain =
            <MyPcs as Pcs<TestChallenge, Challenger>>::natural_domain_for_degree(config.pcs(), 8);

        // `first_point()` is the coset shift `g`, which lies in `gH`.
        //
        //     Z_H(g) = (g^{-1} g)^|H| - 1 = 1 - 1 = 0
        //
        // so the selector inverse `1/Z_H` would panic.
        let zeta = TestChallenge::from(domain.first_point());

        // The domain-coincident point has no selectors, rather than a panic.
        let selectors = <Native<TestVal, TestChallenge> as StarkArith<MyConfig>>::selectors(
            &mut Native::new(),
            &domain,
            &zeta,
        );
        assert!(selectors.is_none());
    }
}
