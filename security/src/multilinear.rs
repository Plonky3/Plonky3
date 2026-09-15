//! Algebraic error sources in the multilinear AIR reduction.
//!
//! These are union-bound contributions for independent uniform transcript challenges,
//! before commitment openings. They inherit the caller's Fiat-Shamir assumptions;
//! they do not certify a concrete hash or challenger. No grinding credit is taken.
//!
//! The sumcheck bound is the sum of its round degrees divided by the challenge
//! space size (Thaler, *Proofs, Arguments, and Zero-Knowledge*, Chapter 4,
//! <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf>).
//! The lookup fingerprint bound is the cleared-denominator argument described in
//! [`crate::logup`]. Fractional GKR and coupling counts below follow the concrete
//! multi-STARK transcript, rather than treating the lookup fingerprint as its
//! entire reduction error.

use alloc::vec::Vec;

use libm::log2;

use crate::{ErrorBits, SecurityTerm};

/// Verifier-derived shape of the AIR-to-opening reduction.
#[derive(Clone, Copy, Debug)]
pub struct MultilinearAirParams {
    /// Number of AIRs batched with powers of beta.
    pub num_instances: usize,
    /// Largest number of constraints batched with powers of alpha in any AIR.
    pub max_num_constraints: usize,
    /// Number of variables in the common zerocheck cube, including lookup blocks.
    pub num_variables: usize,
    /// Largest constraint or lookup-expression degree, before the equality weight.
    pub constraint_degree: usize,
    /// Fractional lookup reduction, if the statement declares one.
    pub lookup: Option<MultilinearLookupParams>,
    /// Univariate-skip round, if the statement opens with one.
    pub skip: Option<MultilinearSkipParams>,
}

/// Shape of a univariate-skip round opening the constraint sumcheck.
///
/// A skip round binds several variables with one challenge instead of one each.
///
/// That moves error out of the per-round terms and into one reconstruction term.
///
/// The two shapes are therefore charged apart, rather than one bounding the other.
#[derive(Clone, Copy, Debug)]
pub struct MultilinearSkipParams {
    /// Dimension of the subspace the round polynomial vanishes on.
    ///
    /// This is how many variables the round binds in one go.
    pub log_size: usize,
    /// Dimension of the subspace the round polynomial is transmitted on.
    ///
    /// The verifier admits any polynomial of degree below this size.
    ///
    /// That is what the reconstruction term charges for.
    pub log_extended: usize,
}

/// Exact tuple count and padded GKR shape derived from the lookup plan.
#[derive(Clone, Copy, Debug)]
pub struct MultilinearLookupParams {
    /// Arity of the padded fraction tables, hence number of GKR layers.
    pub num_variables: usize,
    /// Sum of tuple counts times their respective trace heights, excluding padding.
    pub num_fractions: usize,
    /// Largest payload width; the next power is reserved for the bus identifier.
    pub max_message_width: usize,
}

/// Terms to union-compose with each prescribed-point PCS opening and a hash cap.
///
/// `field_bits` and `nonzero_field_bits` must be lower bounds on respectively
/// `log2(q)` and `log2(q - 1)`. The latter accounts for the zerocheck's nonzero
/// tau rejection sampling. A degree-zero term is omitted because that batching
/// challenge cannot erase a nonzero claim. Invalid scalar shapes yield zero bits.
pub fn reduction_terms(
    air: &MultilinearAirParams,
    field_bits: usize,
    nonzero_field_bits: usize,
) -> Vec<SecurityTerm> {
    if air.num_instances == 0 || air.num_variables == 0 || air.constraint_degree == 0 {
        return alloc::vec![SecurityTerm::new(
            "invalid-multilinear-shape",
            ErrorBits::from_log2(0.0)
        )];
    }
    let mut terms = Vec::new();
    let mut add = |label, degree: f64, bits: usize| {
        if degree > 0.0 {
            terms.push(SecurityTerm::new(
                label,
                ErrorBits::from_log2((bits as f64 - log2(degree)).max(0.0)),
            ));
        }
    };
    add(
        "constraint-batching",
        air.max_num_constraints.saturating_sub(1) as f64,
        field_bits,
    );
    add(
        "air-batching",
        air.num_instances.saturating_sub(1) as f64,
        field_bits,
    );
    // A skip round takes its variables out of both the point and the rounds.
    //
    //     plain:  a point over m variables, then m rounds
    //     skip:   a point over m - k variables, then one skip round, then m - k rounds
    //
    // The skipped variables are never drawn into the equality point.
    //
    // The zerocheck term therefore shrinks with the round count.
    let skipped = air.skip.map_or(0, |skip| skip.log_size);
    let kept = air.num_variables.saturating_sub(skipped);

    // A fixed nonzero constraint table has a multilinear extension. Its
    // evaluation at tau vanishes with probability at most h/(q-1). The tail
    // inherited from GKR is also rejection-sampled over the nonzero elements.
    add("zerocheck", kept as f64, nonzero_field_bits);

    if let Some(skip) = air.skip {
        // The verifier reconstructs the round polynomial itself, admitting any degree
        // below the extension size that vanishes on the subspace.
        //
        // That is one degree more than an honest prover uses.
        //
        // The challenge must separate a dishonest message from the true one over all of it.
        let reconstruction = ((1u64 << skip.log_extended) - 1) as f64;
        add("skip-reconstruction", reconstruction, field_bits);

        // Collapsing the blend takes one degree-two round per skipped variable.
        add("skip-opening", skipped as f64 * 2.0, field_bits);
    }

    // The equality weight raises the native AIR degree by one in every round.
    add(
        "constraint-sumcheck",
        kept as f64 * (air.constraint_degree as f64 + 1.0),
        field_bits,
    );
    if let Some(lookup) = air.lookup {
        add(
            "logup-fingerprint",
            lookup.num_fractions as f64 * (lookup.max_message_width as f64 + 2.0),
            field_bits,
        );
        let layers = lookup.num_variables as f64;
        // Each layer has one numerator/denominator batching draw and one
        // branch draw. Layer i additionally has i cubic sumcheck rounds.
        add(
            "fractional-gkr",
            2.0 * layers + 3.0 * layers * (layers - 1.0) / 2.0,
            nonzero_field_bits,
        );
        add("lookup-opening-link", 1.0, field_bits); // theta
        add("lookup-air-link", 1.0, field_bits); // eta
    }
    terms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduction_charges_batching_zerocheck_and_every_sumcheck_round() {
        let terms = reduction_terms(
            &MultilinearAirParams {
                num_instances: 2,
                max_num_constraints: 3,
                num_variables: 4,
                constraint_degree: 3,
                lookup: None,
                skip: None,
            },
            100,
            99,
        );
        // 2 alpha roots + 1 beta root + 4/(q-1) + 4*4 sumcheck roots.
        // q >= 2^100 and q-1 >= 2^99, so the union bound is 27/2^100.
        let bits = crate::ErrorBits::sum(
            &terms
                .iter()
                .map(|term| term.bits)
                .collect::<alloc::vec::Vec<_>>(),
        )
        .bits();
        assert!((bits - (100.0 - libm::log2(27.0))).abs() < 1e-12);
    }

    /// The plain shape the skip variants are compared against.
    fn plain(num_variables: usize) -> MultilinearAirParams {
        MultilinearAirParams {
            num_instances: 1,
            max_num_constraints: 1,
            num_variables,
            constraint_degree: 2,
            lookup: None,
            skip: None,
        }
    }

    /// The bits charged under one label.
    fn charged(terms: &[SecurityTerm], label: &str) -> Option<f64> {
        terms
            .iter()
            .find(|term| term.label == label)
            .map(|term| term.bits.bits())
    }

    #[test]
    fn a_statement_without_a_skip_is_charged_exactly_as_before() {
        // Adding the skip shape must not move a statement that declares none.
        //
        // Fixture state: 10 variables, degree 2, no lookup.
        //
        //     zerocheck            10
        //     constraint-sumcheck  10 * 3
        //     and no skip term at all
        let terms = reduction_terms(&plain(10), 100, 99);

        assert_eq!(charged(&terms, "zerocheck"), Some(99.0 - libm::log2(10.0)));
        assert_eq!(
            charged(&terms, "constraint-sumcheck"),
            Some(100.0 - libm::log2(30.0))
        );
        assert_eq!(charged(&terms, "skip-reconstruction"), None);
        assert_eq!(charged(&terms, "skip-opening"), None);
    }

    #[test]
    fn a_skip_moves_error_out_of_the_rounds_and_into_the_reconstruction() {
        // Fixture state: 10 variables, 6 of them skipped, transmitted on dimension 7.
        //
        //     zerocheck            4          the point covers the kept variables only
        //     constraint-sumcheck  4 * 3      four rounds, not ten
        //     skip-reconstruction  2^7 - 1    the degree the verifier's interpolation admits
        //     skip-opening         6 * 2      one degree-two round per skipped variable
        let mut params = plain(10);
        params.skip = Some(MultilinearSkipParams {
            log_size: 6,
            log_extended: 7,
        });
        let terms = reduction_terms(&params, 100, 99);

        assert_eq!(charged(&terms, "zerocheck"), Some(99.0 - libm::log2(4.0)));
        assert_eq!(
            charged(&terms, "constraint-sumcheck"),
            Some(100.0 - libm::log2(12.0))
        );
        assert_eq!(
            charged(&terms, "skip-reconstruction"),
            Some(100.0 - libm::log2(127.0))
        );
        assert_eq!(
            charged(&terms, "skip-opening"),
            Some(100.0 - libm::log2(12.0))
        );
    }

    #[test]
    fn the_reconstruction_term_dominates_what_the_rounds_gave_up() {
        // The skip is not a free saving in soundness: it trades many small round terms for one
        // large reconstruction term, and the reconstruction is the bigger of the two.
        //
        // Nothing may read the skip as strictly cheaper, so the direction is pinned here.
        let mut params = plain(10);
        params.skip = Some(MultilinearSkipParams {
            log_size: 6,
            log_extended: 7,
        });

        let plain_terms = reduction_terms(&plain(10), 100, 99);
        let skip_terms = reduction_terms(&params, 100, 99);

        let total = |terms: &[SecurityTerm]| {
            ErrorBits::sum(
                &terms
                    .iter()
                    .map(|term| term.bits)
                    .collect::<alloc::vec::Vec<_>>(),
            )
            .bits()
        };

        assert!(
            total(&skip_terms) < total(&plain_terms),
            "the skip costs bits overall: skip={} plain={}",
            total(&skip_terms),
            total(&plain_terms)
        );
    }

    #[test]
    fn skipping_every_variable_leaves_no_rounds_to_charge() {
        // The widest legal skip binds the whole cube, so the residual terms disappear rather
        // than going negative.
        let mut params = plain(6);
        params.skip = Some(MultilinearSkipParams {
            log_size: 6,
            log_extended: 7,
        });
        let terms = reduction_terms(&params, 100, 99);

        assert_eq!(charged(&terms, "zerocheck"), None);
        assert_eq!(charged(&terms, "constraint-sumcheck"), None);
        assert_eq!(
            charged(&terms, "skip-reconstruction"),
            Some(100.0 - libm::log2(127.0))
        );
    }

    #[test]
    fn lookup_charges_fingerprint_gkr_and_both_links() {
        let params = MultilinearAirParams {
            num_instances: 1,
            max_num_constraints: 1,
            num_variables: 5,
            constraint_degree: 1,
            lookup: Some(MultilinearLookupParams {
                num_variables: 5,
                num_fractions: 64,
                max_message_width: 1,
            }),
            skip: None,
        };
        let terms = reduction_terms(&params, 100, 99);
        let find = |label| {
            terms
                .iter()
                .find(|term| term.label == label)
                .unwrap()
                .bits
                .bits()
        };
        assert!((find("logup-fingerprint") - (100.0 - libm::log2(192.0))).abs() < 1e-12);
        // Five lambda and branch draws, plus 0+1+2+3+4 cubic rounds.
        assert!((find("fractional-gkr") - (99.0 - libm::log2(40.0))).abs() < 1e-12);
        assert_eq!(find("lookup-opening-link"), 100.0);
        assert_eq!(find("lookup-air-link"), 100.0);
    }
}
