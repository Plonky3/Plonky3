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
    /// Indexed-lookup reduction, if the statement declares one.
    pub logup_star: Option<MultilinearLogupStarParams>,
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

/// Shape of the indexed-lookup reduction, derived from its plan.
///
/// A reader names the entry it reads, and the values it pulled are claimed at one point.
///
/// The reduction turns those claims into claims about the table and the position column.
#[derive(Clone, Copy, Debug)]
pub struct MultilinearLogupStarParams {
    /// Arity of the padded leaf table, hence the number of GKR layers.
    pub num_variables: usize,
    /// Leaves carrying a fraction: every reader row, and every entry of every table.
    ///
    /// Padding leaves carry a zero over a one, so they have no pole and are excluded.
    pub num_leaves: usize,
    /// Largest number of readers pulling from any one table.
    pub max_readers_per_table: usize,
    /// Arity of the widest table, hence the product sumcheck's round count.
    pub max_table_variables: usize,
    /// Column claims the reduction closes on, counted over every table.
    pub num_column_claims: usize,
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
    // A fixed nonzero constraint table has a multilinear extension. Its
    // evaluation at tau vanishes with probability at most h/(q-1). The tail
    // inherited from GKR is also rejection-sampled over the nonzero elements.
    add("zerocheck", air.num_variables as f64, nonzero_field_bits);
    // The equality weight raises the native AIR degree by one in every round.
    add(
        "constraint-sumcheck",
        air.num_variables as f64 * (air.constraint_degree as f64 + 1.0),
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
    if let Some(logup_star) = air.logup_star {
        // Readers of one table are combined under consecutive powers of one challenge.
        //
        // Per-reader errors reach the batched claim as a polynomial in that challenge.
        //
        // Its coefficients are the errors, so it vanishes only at one of its roots.
        add(
            "logup-star-reader-batching",
            logup_star.max_readers_per_table.saturating_sub(1) as f64,
            field_bits,
        );
        // The reduction rests on one rational identity per table:
        //
        //     sum_i w_i / (c - iota(I_i))  -  sum_v Y_v / (c - iota(v))  =  0
        //
        // It holds for every `c` exactly when `Y` is the pushforward of the positions.
        //
        // Clearing the denominators leaves a numerator of degree below the pole count.
        //
        // Every pole is a reader row or a table entry.
        //
        // A table's own challenge is fixed while the others vary.
        //
        // The other tables therefore move the identity by a constant, adding one degree.
        //
        // Each challenge is rejection-sampled away from zero.
        //
        // The first entry embeds to zero, and would otherwise zero a denominator.
        add(
            "logup-star-entry-challenge",
            logup_star.num_leaves as f64,
            nonzero_field_bits,
        );
        // The same reduction the lookup argument uses, over this statement's leaf table.
        let layers = logup_star.num_variables as f64;
        add(
            "logup-star-fractional-gkr",
            2.0 * layers + 3.0 * layers * (layers - 1.0) / 2.0,
            nonzero_field_bits,
        );
        // The product sumcheck's summand is a pushforward times a column.
        //
        // Both are multilinear, so every round polynomial is quadratic.
        add(
            "logup-star-product-sumcheck",
            2.0 * logup_star.max_table_variables as f64,
            field_bits,
        );
        // Every column claim in the reduction earns its own power of one challenge.
        //
        // Errors across tables and columns cancel only at a root of that polynomial.
        add(
            "logup-star-column-batching",
            logup_star.num_column_claims.saturating_sub(1) as f64,
            field_bits,
        );
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
                logup_star: None,
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
            logup_star: None,
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

    /// A statement whose indexed lookups have a shape distinct in every dimension.
    ///
    ///     leaf table   2^5 padded, 24 leaves carrying a fraction
    ///     readers      at most 3 on one table
    ///     tables       widest over 4 variables, 6 column claims in all
    fn logup_star_params() -> MultilinearAirParams {
        MultilinearAirParams {
            num_instances: 1,
            max_num_constraints: 1,
            num_variables: 5,
            constraint_degree: 1,
            lookup: None,
            logup_star: Some(MultilinearLogupStarParams {
                num_variables: 5,
                num_leaves: 24,
                max_readers_per_table: 3,
                max_table_variables: 4,
                num_column_claims: 6,
            }),
        }
    }

    #[test]
    fn logup_star_charges_every_challenge_it_draws() {
        let terms = reduction_terms(&logup_star_params(), 100, 99);
        let find = |label| {
            terms
                .iter()
                .find(|term| term.label == label)
                .unwrap()
                .bits
                .bits()
        };

        // Two roots separate three readers.
        assert!((find("logup-star-reader-batching") - (100.0 - libm::log2(2.0))).abs() < 1e-12);

        // One pole per leaf carrying a fraction, drawn away from zero.
        assert!((find("logup-star-entry-challenge") - (99.0 - libm::log2(24.0))).abs() < 1e-12);

        // Five layers: two draws each, plus 0+1+2+3+4 cubic rounds.
        assert!((find("logup-star-fractional-gkr") - (99.0 - libm::log2(40.0))).abs() < 1e-12);

        // Four rounds of a quadratic summand.
        assert!((find("logup-star-product-sumcheck") - (100.0 - libm::log2(8.0))).abs() < 1e-12);

        // Five roots separate six column claims.
        assert!((find("logup-star-column-batching") - (100.0 - libm::log2(5.0))).abs() < 1e-12);
    }

    #[test]
    fn a_batching_term_follows_the_count_it_separates() {
        // Doubling what a challenge separates doubles the degree a prover needs a root of.
        //
        // The term therefore loses exactly one bit.
        //
        //     3 readers -> 2 roots -> 100 - log2(2)
        //     5 readers -> 4 roots -> 100 - log2(4)
        let mut wider = logup_star_params();
        let mut plan = wider.logup_star.unwrap();
        plan.max_readers_per_table = 5;
        wider.logup_star = Some(plan);

        let bits = |params: &MultilinearAirParams| {
            reduction_terms(params, 100, 99)
                .iter()
                .find(|term| term.label == "logup-star-reader-batching")
                .unwrap()
                .bits
                .bits()
        };
        assert!((bits(&logup_star_params()) - bits(&wider) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn one_reader_per_table_needs_no_batching_term() {
        // A single reader is already its own claim.
        //
        // The challenge separates nothing, and cannot erase an error.
        let mut alone = logup_star_params();
        let mut plan = alone.logup_star.unwrap();
        plan.max_readers_per_table = 1;
        alone.logup_star = Some(plan);

        let terms = reduction_terms(&alone, 100, 99);
        assert!(
            !terms
                .iter()
                .any(|term| term.label == "logup-star-reader-batching")
        );
    }

    #[test]
    fn a_statement_without_indexed_lookups_charges_nothing_new() {
        // The reduction is opt-in.
        //
        // A batch declaring none must cost what it did before the reduction existed.
        let terms = reduction_terms(
            &MultilinearAirParams {
                num_instances: 2,
                max_num_constraints: 3,
                num_variables: 4,
                constraint_degree: 3,
                lookup: None,
                logup_star: None,
            },
            100,
            99,
        );
        assert!(
            !terms
                .iter()
                .any(|term| term.label.starts_with("logup-star-"))
        );
    }
}
