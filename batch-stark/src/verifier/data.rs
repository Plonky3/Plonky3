use core::fmt::Debug;

use p3_air::{Air, RowWindow};
use p3_commit::PolynomialSpace;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_lookup::folder::VerifierConstraintFolderWithLookups;
use p3_lookup::{Lookup, LookupProtocol};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_uni_stark::{VerificationError, VerifierConstraintFolder};

use crate::config::{Domain, StarkGenericConfig as SGC, Val};

/// Structure storing all data needed for verifying one instance's constraints at the out-of-domain point.
pub struct VerifierData<'a, SC: SGC> {
    /// Out-of-domain point at which constraints are evaluated.
    pub(crate) zeta: SC::Challenge,
    /// Challenge used to fold constraints.
    pub(crate) alpha: SC::Challenge,
    /// Main trace evaluated at `zeta`.
    pub(crate) trace_local: &'a [SC::Challenge],
    /// Main trace evaluated at the following point `g * zeta`, where `g` is the subgroup generator.
    pub(crate) trace_next: &'a [SC::Challenge],
    /// Preprocessed trace evaluated at `zeta`.
    pub(crate) preprocessed_local: &'a [SC::Challenge],
    /// Preprocessed trace evaluated at the following point `g * zeta`, where `g` is the subgroup generator.
    pub(crate) preprocessed_next: &'a [SC::Challenge],
    /// Permutation trace evaluated at `zeta`.
    pub(crate) permutation_local: &'a [SC::Challenge],
    /// Permutation trace evaluated at the following point `g * zeta`, where `g` is the subgroup generator.
    pub(crate) permutation_next: &'a [SC::Challenge],
    /// Challenges used for the lookup argument.
    pub(crate) permutation_challenges: &'a [SC::Challenge],
    /// Expected cumulated values for global lookup arguments.
    pub(crate) permutation_values: &'a [SC::Challenge],
    /// Periodic column polynomials evaluated at the OOD point `zeta`.
    pub(crate) periodic_values: &'a [SC::Challenge],
    /// Lookup contexts for this instance.
    pub(crate) lookups: &'a [Lookup<Val<SC>>],
    /// Public values for this instance.
    pub(crate) public_values: &'a [Val<SC>],
    /// Trace domain for this instance.
    pub(crate) trace_domain: Domain<SC>,
    /// Quotient polynomial evaluated at `zeta`.
    pub(crate) quotient: SC::Challenge,
}

impl<'a, SC: SGC> VerifierData<'a, SC> {
    /// Verifies that the folded constraints match the quotient polynomial at zeta.
    ///
    /// This evaluates the AIR constraints at the out-of-domain point and checks
    /// that constraints(zeta) / Z_H(zeta) = quotient(zeta).
    pub fn verify_constraints_with_lookups<A, LG: LookupProtocol, PcsErr: Debug>(
        &self,
        air: &A,
        lookup_gadget: &LG,
    ) -> Result<(), VerificationError<PcsErr>>
    where
        A: for<'b> Air<VerifierConstraintFolderWithLookups<'b, SC>>,
    {
        // The constraint check below divides by the vanishing polynomial of this trace domain.
        // Reject any zeta on the domain, where that polynomial is zero and `inv_vanishing` panics.
        // Honest Fiat-Shamir sampling reaches this only with probability |H| / |EF|.
        if self
            .trace_domain
            .vanishing_poly_at_point(self.zeta)
            .is_zero()
        {
            return Err(VerificationError::OodPointInDomain);
        }

        let sels = self.trace_domain.selectors_at_point(self.zeta);

        let main = VerticalPair::new(
            RowMajorMatrixView::new_row(self.trace_local),
            RowMajorMatrixView::new_row(self.trace_next),
        );

        let preprocessed = VerticalPair::new(
            RowMajorMatrixView::new_row(self.preprocessed_local),
            RowMajorMatrixView::new_row(self.preprocessed_next),
        );

        let preprocessed_window =
            RowWindow::from_two_rows(preprocessed.top.values, preprocessed.bottom.values);
        let inner_folder = VerifierConstraintFolder {
            main,
            preprocessed,
            preprocessed_window,
            periodic_values: self.periodic_values,
            public_values: self.public_values,
            is_first_row: sels.is_first_row,
            is_last_row: sels.is_last_row,
            is_transition: sels.is_transition,
            alpha: self.alpha,
            accumulator: SC::Challenge::ZERO,
        };
        let mut folder = VerifierConstraintFolderWithLookups {
            inner: inner_folder,
            permutation: VerticalPair::new(
                RowMajorMatrixView::new_row(self.permutation_local),
                RowMajorMatrixView::new_row(self.permutation_next),
            ),
            permutation_challenges: self.permutation_challenges,
            permutation_values: self.permutation_values,
        };
        // Evaluate AIR and lookup constraints.
        lookup_gadget.eval_air_and_lookups(air, &mut folder, self.lookups);

        // Check that constraints(zeta) / Z_H(zeta) = quotient(zeta)
        if folder.inner.accumulator * sels.inv_vanishing != self.quotient {
            return Err(VerificationError::OodEvaluationMismatch { index: None });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::{ExtensionMmcs, Pcs};
    use p3_dft::Radix2DitParallel;
    use p3_field::extension::BinomialExtensionField;
    use p3_fri::{FriParameters, TwoAdicFriPcs};
    use p3_lookup::logup::LogUpGadget;
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

    // No columns, no constraints.
    // Only present to satisfy the generic bound; the guard returns before it is evaluated.
    struct EmptyAir;

    impl<F> BaseAir<F> for EmptyAir {
        fn width(&self) -> usize {
            0
        }
    }

    impl<AB: AirBuilder> Air<AB> for EmptyAir {
        fn eval(&self, _builder: &mut AB) {}
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

        // Every trace slice is empty; the guard fires before any of them is read.
        let verifier_data = VerifierData::<MyConfig> {
            zeta,
            alpha: TestChallenge::ZERO,
            trace_local: &[],
            trace_next: &[],
            preprocessed_local: &[],
            preprocessed_next: &[],
            permutation_local: &[],
            permutation_next: &[],
            permutation_challenges: &[],
            permutation_values: &[],
            periodic_values: &[],
            lookups: &[],
            public_values: &[],
            trace_domain: domain,
            quotient: TestChallenge::ZERO,
        };

        let result = verifier_data.verify_constraints_with_lookups::<EmptyAir, LogUpGadget, ()>(
            &EmptyAir,
            &LogUpGadget::new(),
        );

        // The domain-coincident point is rejected with a clean error, not a panic.
        assert!(
            matches!(result, Err(VerificationError::OodPointInDomain)),
            "expected OodPointInDomain, got {result:?}"
        );
    }
}
