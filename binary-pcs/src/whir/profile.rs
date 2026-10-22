//! Supported parameter profiles, one per proximity regime, derived from the soundness analysis.

use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_whir::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirConfigError, WhirDomain,
};

use crate::whir::error::ProfileError;

/// Widest candidate space a grinding search enumerates, in bits.
///
/// The search counts with a sixty-four bit index, so no alphabet offers more than that.
const GRIND_SEARCH_BITS: usize = 64;

/// Headroom a grinding search keeps below the candidate space, so an exhaustive one rarely fails.
const GRIND_MARGIN_BITS: usize = 8;

/// Grinding one witness of the alphabet can carry.
fn grinding_ceiling<F: Field>() -> usize {
    F::bits()
        .min(GRIND_SEARCH_BITS)
        .saturating_sub(GRIND_MARGIN_BITS)
}

/// Raise the grinding allowance to a figure the analysis asks for.
///
/// # Errors
///
/// Returns an error when one witness of the alphabet cannot carry the figure.
fn raise_allowance<F: Field>(required: usize) -> Result<usize, ProfileError> {
    let ceiling = grinding_ceiling::<F>();
    if required > ceiling {
        return Err(ProfileError::Grinding { required, ceiling });
    }
    Ok(required)
}

/// A named parameter profile for one proximity regime.
///
/// The regime, the code rate and the folding width are chosen by the caller.
///
/// Query counts, samples and grinding are all derived from the soundness analysis.
///
/// Nothing here defaults to a regime: each one is a constructor of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryWhirProfile {
    /// Proximity regime the derivation is graded under.
    assumption: SecurityAssumption,
    /// Target in bits for every error term the schedule charges.
    security_level: usize,
    /// Base-two logarithm of the inverse rate of the first codeword.
    starting_log_inv_rate: usize,
    /// Variables eliminated by each round.
    folding_factor: usize,
    /// Variables the first round eliminates instead, and bits the codeword domain then loses.
    first_round: Option<(usize, usize)>,
    /// Grinding the derivation starts from, so queries close only the rest of the target.
    query_grinding: usize,
}

impl BinaryWhirProfile {
    /// The regime that assumes only unique decoding, and conjectures nothing.
    #[must_use]
    pub const fn unique_decoding(
        security_level: usize,
        starting_log_inv_rate: usize,
        folding_factor: usize,
    ) -> Self {
        Self {
            assumption: SecurityAssumption::UniqueDecoding,
            security_level,
            starting_log_inv_rate,
            folding_factor,
            first_round: None,
            query_grinding: 0,
        }
    }

    /// The regime that decodes to the Johnson radius, where list decoding is proven.
    ///
    /// Mutual correlated agreement at this radius is a theorem, resting on no conjecture.
    #[must_use]
    pub const fn proven_list_decoding(
        security_level: usize,
        starting_log_inv_rate: usize,
        folding_factor: usize,
    ) -> Self {
        Self {
            assumption: SecurityAssumption::JohnsonBound,
            security_level,
            starting_log_inv_rate,
            folding_factor,
            first_round: None,
            query_grinding: 0,
        }
    }

    /// The regime that decodes to capacity, which no theorem supports.
    ///
    /// Every additive binary domain refuses it, because the assumption is refuted there.
    #[must_use]
    pub const fn conjectural(
        security_level: usize,
        starting_log_inv_rate: usize,
        folding_factor: usize,
    ) -> Self {
        Self {
            assumption: SecurityAssumption::CapacityBound,
            security_level,
            starting_log_inv_rate,
            folding_factor,
            first_round: None,
            query_grinding: 0,
        }
    }

    /// Fold `folding_factor` variables in the first round, then shrink the domain by `domain_reduction` bits.
    ///
    /// Later rounds keep the profile's folding and shrink the domain by one bit each.
    #[must_use]
    pub const fn with_first_round(
        mut self,
        folding_factor: usize,
        domain_reduction: usize,
    ) -> Self {
        self.first_round = Some((folding_factor, domain_reduction));
        self
    }

    /// Grind at least `bits` bits before every query phase, so fewer queries close the target.
    #[must_use]
    pub const fn with_query_grinding(mut self, bits: usize) -> Self {
        self.query_grinding = bits;
        self
    }

    /// The folding factors and the explicit round rates of this profile.
    fn schedule(&self, num_variables: usize) -> Result<(FoldingFactor, Vec<usize>), ProfileError> {
        let Some((first, reduction)) = self.first_round else {
            return Ok((FoldingFactor::Constant(self.folding_factor), Vec::new()));
        };
        let folding = FoldingFactor::ConstantFromSecondRound(first, self.folding_factor);
        let schedule = folding
            .compute_folding_schedule(num_variables)
            .map_err(|error| ProfileError::Schedule(WhirConfigError::FoldingFactor(error)))?;
        // A reduction past the first fold leaves a zero rate, which the schedule refuses.
        let mut rate = (self.starting_log_inv_rate + first).saturating_sub(reduction);
        let mut rates = Vec::with_capacity(schedule.len() - 1);
        for &factor in &schedule[1..] {
            rates.push(rate);
            rate += factor - 1;
        }
        Ok((folding, rates))
    }

    /// The proximity regime this profile is graded under.
    #[must_use]
    pub const fn assumption(&self) -> SecurityAssumption {
        self.assumption
    }

    /// The target in bits for every error term.
    #[must_use]
    pub const fn security_level(&self) -> usize {
        self.security_level
    }

    /// Derive the schedule this profile stands for over a concrete evaluation domain.
    ///
    /// Grinding is raised to exactly what the analysis demands, rather than guessed at.
    ///
    /// The domain vetoes a regime whose distance assumption fails for its code.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived schedule opens no position.
    ///
    /// Returns an error when the domain refuses the regime.
    ///
    /// Returns an error when the analysis demands more grinding than one witness can carry.
    ///
    /// Returns an error when the schedule itself is infeasible.
    pub fn config<EF, F, Challenger, Domain>(
        &self,
        num_variables: usize,
        domain: &Domain,
    ) -> Result<WhirConfig<EF, F, Challenger>, ProfileError>
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        Domain: WhirDomain<F, EF>,
    {
        let (folding_factor, round_log_inv_rates) = self.schedule(num_variables)?;
        let mut pow_bits = raise_allowance::<F>(self.query_grinding)?;
        loop {
            let parameters = ProtocolParameters {
                starting_log_inv_rate: self.starting_log_inv_rate,
                round_log_inv_rates: round_log_inv_rates.clone(),
                folding_factor: folding_factor.clone(),
                soundness_type: self.assumption,
                security_level: self.security_level,
                pow_bits,
            };
            match WhirConfig::new_with_domain(num_variables, parameters, domain) {
                Ok(config) => return Ok(config),
                // The analysis reports exactly how much grinding the gap needs.
                // Raising the allowance to that figure is the derivation, not a search.
                Err(WhirConfigError::PowBitsExceedBudget { required, .. })
                    if required > pow_bits =>
                {
                    pow_bits = raise_allowance::<F>(required)?;
                }
                // The schedule refuses a run that opens nothing, whichever route built it.
                //
                // Naming it here keeps callers from reading a bare scheduling failure.
                Err(WhirConfigError::ZeroQueries) => return Err(ProfileError::ZeroQueries),
                Err(error) => return Err(ProfileError::Schedule(error)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryChallenger, BinaryField32, BinaryField64, BinaryField128};
    use p3_challenger::{GrindingChallenger, HashChallenger};
    use p3_keccak::Keccak256Hash;
    use p3_whir::{FoldingFactorError, SecurityAssumption, WhirConfigError};

    use super::{BinaryWhirProfile, grinding_ceiling, raise_allowance};
    use crate::test_util::MyChallenger;
    use crate::whir::error::ProfileError;
    use crate::whir::{BinaryWhirDomain, BooleanWhirDomain, ProofShape};

    type EF = BinaryField128;
    type NarrowChallenger = BinaryChallenger<BinaryField32, HashChallenger<u8, Keccak256Hash, 32>>;

    const NUM_VARIABLES: usize = 9;
    const SECURITY_LEVEL: usize = 100;
    const LOG_INV_RATE: usize = 2;
    const FOLDING: usize = 3;

    #[test]
    fn a_schedule_that_opens_no_position_is_refused() {
        // Invariant: a schedule must open at least one position.
        //
        // Queries alone test proximity, so a schedule opening none accepts any codeword.
        //
        // Crediting grinding with the whole target derives exactly that.
        //
        // This profile reaches it with a zero target, which credits the whole of nothing.
        let domain = BooleanWhirDomain::default();
        let profile = BinaryWhirProfile::proven_list_decoding(0, LOG_INV_RATE, FOLDING);
        assert!(matches!(
            profile
                .config::<EF, EF, MyChallenger, _>(NUM_VARIABLES, &domain)
                .err(),
            Some(ProfileError::ZeroQueries)
        ));

        // The floor is on the derived count, so a target that does buy queries is untouched.
        let real = BinaryWhirProfile::proven_list_decoding(SECURITY_LEVEL, LOG_INV_RATE, FOLDING)
            .config::<EF, EF, MyChallenger, _>(NUM_VARIABLES, &domain)
            .unwrap();
        assert!(real.terminal().num_queries > 0);
    }

    #[test]
    fn a_profile_that_folds_nothing_is_refused_by_the_schedule() {
        // Invariant: a round must eliminate at least one variable.
        //
        // This one needs no check of our own, since the schedule already refuses it.
        //
        // The test pins that refusal, so nobody adds a redundant guard here later.
        let domain = BooleanWhirDomain::default();
        let profile = BinaryWhirProfile::proven_list_decoding(SECURITY_LEVEL, LOG_INV_RATE, 0);
        assert!(matches!(
            profile
                .config::<EF, EF, MyChallenger, _>(NUM_VARIABLES, &domain)
                .err(),
            Some(ProfileError::Schedule(WhirConfigError::FoldingFactor(
                FoldingFactorError::ZeroFactor
            )))
        ));
    }

    #[test]
    fn a_wide_first_round_shrinks_the_domain_by_its_reduction() {
        // Fixture state: fold six variables first and drop three domain bits, then fold four.
        let domain = BooleanWhirDomain::default();
        let config = BinaryWhirProfile::proven_list_decoding(SECURITY_LEVEL, 1, 4)
            .with_first_round(6, 3)
            .config::<EF, EF, MyChallenger, _>(20, &domain)
            .unwrap();
        assert_eq!(config.folding_schedule(), [6, 4, 4]);
        assert_eq!(config.rs_reduction_factor(0), 3);
        assert_eq!(config.rs_reduction_factor(1), 1);
        let rates: alloc::vec::Vec<_> = config
            .round_parameters()
            .iter()
            .map(|round| round.log_inv_rate)
            .collect();
        assert_eq!(rates, [4, 7]);

        // A reduction past the first fold leaves no redundancy, which the schedule refuses.
        assert!(matches!(
            BinaryWhirProfile::proven_list_decoding(SECURITY_LEVEL, 1, 4)
                .with_first_round(6, 7)
                .config::<EF, EF, MyChallenger, _>(20, &domain),
            Err(ProfileError::Schedule(_))
        ));
    }

    #[test]
    fn query_grinding_buys_fewer_queries_and_is_ground() {
        let domain = BooleanWhirDomain::default();
        let derive = |bits| {
            BinaryWhirProfile::proven_list_decoding(SECURITY_LEVEL, 1, 4)
                .with_query_grinding(bits)
                .config::<EF, EF, MyChallenger, _>(16, &domain)
                .unwrap()
        };
        let (plain, ground) = (derive(0), derive(12));
        assert!(ground.round_parameters()[0].num_queries < plain.round_parameters()[0].num_queries);
        assert!(ground.round_parameters()[0].pow_bits > 0);
        assert!(ground.max_pow_bits() <= 12);

        // A request past what one witness carries is refused.
        assert!(matches!(
            BinaryWhirProfile::proven_list_decoding(SECURITY_LEVEL, 1, 4)
                .with_query_grinding(57)
                .config::<EF, EF, MyChallenger, _>(16, &domain),
            Err(ProfileError::Grinding { .. })
        ));
    }

    #[test]
    fn each_regime_is_reached_by_its_own_constructor() {
        assert_eq!(
            BinaryWhirProfile::unique_decoding(SECURITY_LEVEL, LOG_INV_RATE, FOLDING).assumption(),
            SecurityAssumption::UniqueDecoding
        );
        assert_eq!(
            BinaryWhirProfile::proven_list_decoding(SECURITY_LEVEL, LOG_INV_RATE, FOLDING)
                .assumption(),
            SecurityAssumption::JohnsonBound
        );
        assert_eq!(
            BinaryWhirProfile::conjectural(SECURITY_LEVEL, LOG_INV_RATE, FOLDING).assumption(),
            SecurityAssumption::CapacityBound
        );
    }

    #[test]
    fn decoding_further_buys_fewer_opened_positions() {
        let domain = BooleanWhirDomain::default();
        let shape = |profile: BinaryWhirProfile| {
            ProofShape::of(
                &profile
                    .config::<EF, EF, MyChallenger, _>(NUM_VARIABLES, &domain)
                    .unwrap(),
                &crate::test_util::mmcs(),
                1,
            )
            .unwrap()
        };
        let unique = shape(BinaryWhirProfile::unique_decoding(
            SECURITY_LEVEL,
            LOG_INV_RATE,
            FOLDING,
        ));
        let proven = shape(BinaryWhirProfile::proven_list_decoding(
            SECURITY_LEVEL,
            LOG_INV_RATE,
            FOLDING,
        ));
        assert!(proven.stir_queries < unique.stir_queries);
    }

    #[test]
    fn the_conjectural_regime_is_refused_by_the_additive_domain() {
        let refused = BinaryWhirProfile::conjectural(SECURITY_LEVEL, LOG_INV_RATE, FOLDING)
            .config::<EF, EF, MyChallenger, _>(NUM_VARIABLES, &BooleanWhirDomain::default())
            .unwrap_err();
        // The schedule error carries no equality.
        //
        // So the variant is pinned whole, and the rendered message is checked as well.
        assert!(matches!(
            refused,
            ProfileError::Schedule(WhirConfigError::UnsupportedSecurityAssumption {
                assumption: SecurityAssumption::CapacityBound
            })
        ));
        assert_eq!(
            alloc::format!("{refused}"),
            "the evaluation domain does not support the CapacityBound soundness regime"
        );
    }

    // A schedule the challenger can grind, or a refusal naming a figure above what it can.
    fn inside_the_ceiling(derived: &Result<usize, ProfileError>, ceiling: usize) {
        match derived {
            Ok(pow_bits) => assert!(*pow_bits <= ceiling, "{pow_bits} bits past {ceiling}"),
            Err(ProfileError::Grinding {
                required,
                ceiling: named,
            }) => {
                assert_eq!(*named, ceiling);
                assert!(*required > ceiling, "{required} bits is not past {ceiling}");
            }
            // A derivation refused for another reason is not this test's business.
            Err(ProfileError::Schedule(_) | ProfileError::ZeroQueries) => {}
        }
    }

    #[test]
    fn the_ceiling_is_the_headroom_the_search_keeps_below_one_witness() {
        // Integer arithmetic over the alphabet's width, so every host reads the same figure.
        assert_eq!(grinding_ceiling::<BinaryField32>(), 24);
        assert_eq!(grinding_ceiling::<BinaryField64>(), 56);
        assert_eq!(grinding_ceiling::<EF>(), 56);
    }

    #[test]
    fn a_request_one_bit_past_the_ceiling_is_refused() {
        // The request is constructed rather than derived, so no analysis can move the premise.
        let refused = raise_allowance::<BinaryField32>(25).unwrap_err();
        assert!(
            matches!(
                refused,
                ProfileError::Grinding {
                    required: 25,
                    ceiling: 24
                }
            ),
            "{refused:?}"
        );
        assert_eq!(
            alloc::format!("{refused}"),
            "the schedule needs 25 grinding bits, one witness allows 24"
        );
        assert!(matches!(
            raise_allowance::<EF>(57).unwrap_err(),
            ProfileError::Grinding {
                required: 57,
                ceiling: 56
            }
        ));

        // The ceiling itself is allowed, so the refusal starts exactly one bit above it.
        assert_eq!(raise_allowance::<BinaryField32>(24).unwrap(), 24);
        assert_eq!(raise_allowance::<EF>(56).unwrap(), 56);
    }

    #[test]
    fn no_schedule_the_profile_returns_can_abort_the_prover() {
        // What must never come back is a schedule the challenger would refuse to grind.
        let narrow = BinaryWhirDomain::<BinaryField32>::default();
        let wide = BooleanWhirDomain::default();
        for security_level in [80, 100, 128, 160, 200, 256] {
            for log_inv_rate in [1, 2, 3, 4] {
                for num_variables in [8, 12, 16, 20] {
                    for folding in [2, 3, 4] {
                        let profile = BinaryWhirProfile::proven_list_decoding(
                            security_level,
                            log_inv_rate,
                            folding,
                        );
                        inside_the_ceiling(
                            &profile
                                .config::<EF, BinaryField32, NarrowChallenger, _>(
                                    num_variables,
                                    &narrow,
                                )
                                .map(|config| config.max_pow_bits()),
                            grinding_ceiling::<BinaryField32>(),
                        );
                        inside_the_ceiling(
                            &profile
                                .config::<EF, EF, MyChallenger, _>(num_variables, &wide)
                                .map(|config| config.max_pow_bits()),
                            grinding_ceiling::<EF>(),
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic = "too small a margin"]
    fn the_narrow_ceiling_is_the_one_the_challenger_enforces() {
        // The refusal above is worth nothing unless one more bit really does abort a prover.
        let mut challenger = NarrowChallenger::from_hasher(alloc::vec::Vec::new(), Keccak256Hash);
        let _ = challenger.grind(grinding_ceiling::<BinaryField32>() + 1);
    }

    #[test]
    #[should_panic = "too small a margin"]
    fn the_wide_ceiling_is_the_one_the_challenger_enforces() {
        let mut challenger = MyChallenger::from_hasher(alloc::vec::Vec::new(), Keccak256Hash);
        let _ = challenger.grind(grinding_ceiling::<EF>() + 1);
    }
}
