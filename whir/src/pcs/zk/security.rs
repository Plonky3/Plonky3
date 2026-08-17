//! Maps a concrete hiding-WHIR configuration into the generic security
//! accounting provided by `p3-security`.

use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
pub use p3_security::whir::hiding::{
    HidingBaseCaseSecurityReport, HidingBoundClassification, HidingCodeRole,
    HidingCodeSecurityReport, HidingErrorBound, HidingGammaRoundReport, HidingListBound,
    HidingQueryRoundReport,
};
use p3_security::whir::hiding::{
    HidingCodeGeometry, hiding_base_case_security_report as build_hiding_security_report,
};

use super::config::ZkWhirConfig;

impl<EF, F, Challenger> ZkWhirConfig<EF, F, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Reports the heterogeneous hiding base case's ideal-IOP RBR terms.
    ///
    /// This is diagnostic only: it neither rejects the configuration nor
    /// certifies the compiled PCS or the preceding hiding reductions.
    #[must_use]
    pub fn hiding_base_case_security_report(&self) -> HidingBaseCaseSecurityReport {
        let final_config = self.final_round_config();
        let source = HidingCodeGeometry::new(
            HidingCodeRole::Source,
            1 << final_config.num_variables,
            self.oracle_randomness[self.n_rounds()],
            final_config.domain_size >> final_config.folding_factor,
            1,
            self.final_queries,
        );

        let mask_groups = self
            .mask_groups()
            .into_iter()
            .enumerate()
            .map(|(index, group)| {
                let role = if index % 2 == 0 {
                    HidingCodeRole::SumcheckMask { round: index / 2 }
                } else {
                    HidingCodeRole::CodeSwitchMask { round: index / 2 }
                };
                HidingCodeGeometry::new(
                    role,
                    group.shape.message_len,
                    group.shape.randomness_len,
                    group.shape.domain_size,
                    group.width,
                    self.mask_queries,
                )
            })
            .collect::<Vec<_>>();

        build_hiding_security_report::<EF>(
            self.security_level,
            self.soundness_type,
            source,
            mask_groups,
            self.final_pow_bits,
        )
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeField32;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};

    use super::*;
    use crate::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption};
    use crate::pcs::zk::config::ZkParameters;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2KoalaBear<16>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;

    fn stock_config(assumption: SecurityAssumption) -> ZkWhirConfig<EF, F, Challenger> {
        ZkWhirConfig::new(
            16,
            ProtocolParameters {
                security_level: 100,
                pow_bits: 16,
                round_log_inv_rates: vec![],
                folding_factor: FoldingFactor::Constant(4),
                soundness_type: assumption,
                starting_log_inv_rate: 2,
            },
            ZkParameters {
                ell_zk: 16,
                mask_log_inv_rate: 5,
            },
        )
        .unwrap()
    }

    fn classification_config(assumption: SecurityAssumption) -> ZkWhirConfig<EF, F, Challenger> {
        ZkWhirConfig::new(
            16,
            ProtocolParameters {
                security_level: 32,
                pow_bits: 0,
                round_log_inv_rates: vec![],
                folding_factor: FoldingFactor::Constant(4),
                soundness_type: assumption,
                starting_log_inv_rate: 2,
            },
            ZkParameters {
                ell_zk: 4,
                mask_log_inv_rate: 1,
            },
        )
        .unwrap()
    }

    #[test]
    fn field_report_separates_estimate_from_rigorous_floor() {
        let report =
            stock_config(SecurityAssumption::CapacityBound).hiding_base_case_security_report();
        let exact = 4. * libm::log2(F::ORDER_U32 as f64);

        assert!((report.field_order_log2_estimate - exact).abs() < 1e-12);
        assert_eq!(report.field_order_log2_floor, 123);
        assert!(report.field_order_log2_estimate < 124.);
    }

    #[test]
    fn stock_capacity_report_exposes_randomized_source_rate_loss() {
        let report =
            stock_config(SecurityAssumption::CapacityBound).hiding_base_case_security_report();

        assert_eq!(report.assumption, SecurityAssumption::CapacityBound);
        assert_eq!(report.field_order_log2_floor, 123);
        assert!((report.field_order_log2_estimate - 123.954_738_749_797).abs() < 1e-12);
        assert_eq!(report.source.message_len, 16);
        assert_eq!(report.source.randomness_len, 11);
        assert_eq!(report.source.randomized_dimension, 27);
        assert_eq!(report.source.dyadic_dimension, 32);
        assert_eq!(report.source.domain_size, 4096);
        assert_eq!(report.source.randomized_rate, 27. / 4096.);
        assert_eq!(report.source.analysis_log_inv_rate, Some(7));
        assert!(report.source.analysis_radius.is_some());
        assert_eq!(report.source.requested_queries, 11);
        assert_eq!(report.source.effective_queries, 11);
        assert_eq!(report.query_round.shared_pow_bits, 13);
        assert!(
            (report.query_round.source_with_pow.bits.unwrap() - 89.225_717_393_194_63).abs()
                < 1e-12
        );
        assert_eq!(
            report.query_round.with_pow.bits,
            report.query_round.source_with_pow.bits
        );
        assert_eq!(
            report.query_round.with_pow.classification,
            HidingBoundClassification::Conditional
        );
        assert!(report.query_round.before_pow.bits.unwrap() < report.target_bits as f64);
        assert!(report.query_round.with_pow.bits.unwrap() < report.target_bits as f64);
        assert_eq!(
            report.gamma_round.combined.classification,
            HidingBoundClassification::Conditional
        );
        assert!(report.gamma_round.product_list_over_field.vacuous);
        assert!(report.round_by_round.vacuous);
    }

    #[test]
    fn mask_groups_keep_chronological_roles() {
        let report =
            stock_config(SecurityAssumption::CapacityBound).hiding_base_case_security_report();
        let roles: Vec<_> = report.mask_groups.iter().map(|group| group.role).collect();

        assert_eq!(
            roles,
            vec![
                HidingCodeRole::SumcheckMask { round: 0 },
                HidingCodeRole::CodeSwitchMask { round: 0 },
                HidingCodeRole::SumcheckMask { round: 1 },
                HidingCodeRole::CodeSwitchMask { round: 1 },
                HidingCodeRole::SumcheckMask { round: 2 },
            ]
        );
        assert!(
            report
                .mask_groups
                .iter()
                .all(|group| group.analysis_log_inv_rate == Some(5))
        );
        assert!(
            report
                .mask_groups
                .iter()
                .all(|group| group.randomness_len == 21)
        );
    }

    #[test]
    fn regime_classifications_propagate_without_overstatement() {
        let unique = classification_config(SecurityAssumption::UniqueDecoding)
            .hiding_base_case_security_report();
        assert_eq!(
            unique.gamma_round.combined.classification,
            HidingBoundClassification::Proven
        );
        assert_eq!(
            unique.round_by_round.classification,
            HidingBoundClassification::Proven
        );
        assert!(unique.round_by_round.proven_bits().is_some());
        assert!(!unique.round_by_round.vacuous);

        let johnson = classification_config(SecurityAssumption::JohnsonBound)
            .hiding_base_case_security_report();
        assert_eq!(
            johnson.query_round.before_pow.classification,
            HidingBoundClassification::Proven
        );
        assert_eq!(
            johnson.source.mca.classification,
            HidingBoundClassification::Approximation
        );
        assert_eq!(
            johnson.gamma_round.combined.classification,
            HidingBoundClassification::Approximation
        );
        assert_eq!(
            johnson.round_by_round.classification,
            HidingBoundClassification::Approximation
        );
        assert_eq!(johnson.round_by_round.proven_bits(), None);

        let capacity =
            stock_config(SecurityAssumption::CapacityBound).hiding_base_case_security_report();
        assert_eq!(
            capacity.source.query_miss.classification,
            HidingBoundClassification::Proven
        );
        assert!(
            capacity.mask_groups.iter().all(|group| {
                group.query_miss.classification == HidingBoundClassification::Proven
            })
        );
        assert_eq!(
            capacity.query_round.before_pow.classification,
            HidingBoundClassification::Proven
        );
        assert_eq!(
            capacity.gamma_round.combined.classification,
            HidingBoundClassification::Conditional
        );
        assert_eq!(
            capacity.round_by_round.classification,
            HidingBoundClassification::Conditional
        );
        assert_eq!(capacity.round_by_round.proven_bits(), None);
    }

    #[test]
    fn stock_capacity_gamma_coordinate_is_pinned() {
        let report =
            stock_config(SecurityAssumption::CapacityBound).hiding_base_case_security_report();
        let product_list_bits = report.gamma_round.product_list.log2_size.unwrap();
        let raw_product_list_over_field_bits =
            report.field_order_log2_floor as f64 - product_list_bits;

        assert!((report.gamma_round.mca_sum.bits.unwrap() - 92.123_483_053_435).abs() < 1e-10);
        assert!((product_list_bits - 617.657_842_846_621).abs() < 1e-10);
        assert!((raw_product_list_over_field_bits + 494.657_842_846_621).abs() < 1e-10);
        assert_eq!(report.gamma_round.product_list_over_field.bits, Some(0.));
        assert!(report.gamma_round.product_list_over_field.vacuous);
        assert_eq!(report.gamma_round.combined.bits, Some(0.));
        assert!(report.gamma_round.combined.vacuous);
    }
}
