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

/// Label for the bit-alphabet ring-switch term.
pub const BIT_RING_SWITCH_LABEL: &str = "bit-ring-switch";

/// Label for the independent column-point challenge in a batched opening.
pub const COLUMN_BATCH_LABEL: &str = "column-batching";

/// Error of batching `num_batches` column openings with `k` fresh coordinates each.
///
/// Every coordinate is sampled independently from the transcript after the claimed
/// column values are bound. A nonzero discrepancy therefore survives with probability
/// at most `num_batches * k / |F|`.
#[must_use]
pub fn column_batch_error(num_batches: usize, k: usize, field_bits: usize) -> ErrorBits {
    if num_batches == 0 || k == 0 {
        return ErrorBits::from_log2(f64::INFINITY);
    }
    // Keep the product in the logarithm's domain: usize multiplication can overflow even
    // though the union-bound degree is representable as a floating-point number.
    let log_challenges = log2(num_batches as f64) + log2(k as f64);
    ErrorBits::from_log2(field_bits as f64 - log_challenges)
}

/// The labelled column-batching soundness term.
#[must_use]
pub fn column_batch_term(num_batches: usize, k: usize, field_bits: usize) -> SecurityTerm {
    SecurityTerm::new(
        COLUMN_BATCH_LABEL,
        column_batch_error(num_batches, k, field_bits),
    )
}

/// Error of reducing claims about a bit witness to claims about the elements packing it.
///
/// The single-tensor case of [`bit_ring_switch_tensors_error`], with `num_tensors = 1`.
///
/// ```text
///     error = (absorbed_log + 2 * surviving) / |F|
/// ```
///
/// # Arguments
///
/// - `num_reductions`: points reduced, one reduction each.
/// - `absorbed_log`: log of the bits one packed element holds.
/// - `surviving_variables`: variables the packed multilinear has.
/// - `field_bits`: bit width of the field the challenges are drawn from.
#[must_use]
pub fn bit_ring_switch_error(
    num_reductions: usize,
    absorbed_log: usize,
    surviving_variables: usize,
    field_bits: usize,
) -> ErrorBits {
    bit_ring_switch_tensors_error(
        num_reductions,
        1,
        absorbed_log,
        surviving_variables,
        field_bits,
    )
}

/// The bit-alphabet ring-switch term, labelled for a report.
#[must_use]
pub fn bit_ring_switch_term(
    num_reductions: usize,
    absorbed_log: usize,
    surviving_variables: usize,
    field_bits: usize,
) -> SecurityTerm {
    bit_ring_switch_tensors_term(
        num_reductions,
        1,
        absorbed_log,
        surviving_variables,
        field_bits,
    )
}

/// Error of reducing claims about a bit witness to claims about the elements packing it,
/// batching `num_tensors` claims at the same point under powers of a challenge alpha.
///
/// One element holds `2^absorbed_log` bits, so a point of `n` variables splits in two:
///
/// ```text
///     absorbed_log         coordinates inside one element
///     n - absorbed_log     coordinates addressing the elements
/// ```
///
/// One reduction draws `absorbed_log` batching coordinates and, when `num_tensors > 1`, one
/// challenge alpha.
/// The batched row claim is multilinear in the coordinates and of degree `num_tensors - 1`
/// in alpha. It then runs one degree-two sumcheck round per surviving variable.
///
/// ```text
///     error = (absorbed_log + (num_tensors - 1) + 2 * surviving) / |F|
/// ```
///
/// Reductions at different points share no challenge, so `k` of them union to `k` times that.
///
/// The commitment the surviving claims are discharged against charges its own budget.
///
/// # Arguments
///
/// - `num_reductions`: points reduced, one reduction each.
/// - `num_tensors`: tensors batched together at each point under powers of alpha.
/// - `absorbed_log`: log of the bits one packed element holds.
/// - `surviving_variables`: variables the packed multilinear has.
/// - `field_bits`: bit width of the field the challenges are drawn from.
#[must_use]
pub fn bit_ring_switch_tensors_error(
    num_reductions: usize,
    num_tensors: usize,
    absorbed_log: usize,
    surviving_variables: usize,
    field_bits: usize,
) -> ErrorBits {
    // No reduction runs, so no challenge separates anything.
    if num_reductions == 0 {
        return ErrorBits::from_log2(f64::INFINITY);
    }
    // Rounds per reduction: one batching draw per absorbed coordinate, degree
    // num_tensors - 1 in the alpha challenge, two per sumcheck round.
    let per_reduction = absorbed_log as f64
        + num_tensors.saturating_sub(1) as f64
        + 2.0 * surviving_variables as f64;
    // A reduction over a point the packing absorbs whole, with a single tensor, runs no
    // sumcheck round and no batching draw.
    if per_reduction == 0.0 {
        return ErrorBits::from_log2(f64::INFINITY);
    }
    ErrorBits::from_log2(field_bits as f64 - log2(num_reductions as f64 * per_reduction))
}

/// The bit-alphabet ring-switch term over batched tensors, labelled for a report.
#[must_use]
pub fn bit_ring_switch_tensors_term(
    num_reductions: usize,
    num_tensors: usize,
    absorbed_log: usize,
    surviving_variables: usize,
    field_bits: usize,
) -> SecurityTerm {
    SecurityTerm::new(
        BIT_RING_SWITCH_LABEL,
        bit_ring_switch_tensors_error(
            num_reductions,
            num_tensors,
            absorbed_log,
            surviving_variables,
            field_bits,
        ),
    )
}

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
    /// Indexed-lookup reduction, if the statement declares one.
    pub logup_star: Option<MultilinearLogupStarParams>,
}

/// Shape of a univariate-skip round opening the constraint sumcheck.
///
/// A skip round binds several variables with one challenge instead of one each.
/// That moves error out of the per-round terms into one reconstruction term.
///
/// The two shapes are charged apart, rather than one bounding the other.
#[derive(Clone, Copy, Debug)]
pub struct MultilinearSkipParams {
    /// Dimension of the subspace the round polynomial vanishes on.
    ///
    /// This is how many variables the round binds in one go.
    pub log_size: usize,
    /// Dimension of the subspace the round polynomial is transmitted on.
    ///
    /// The verifier admits any polynomial of degree below this size.
    /// That is what the reconstruction term charges for.
    ///
    /// This must be the dimension the verifier actually interpolates on.
    /// It is not a value re-derived from the constraint degree.
    /// A domain wider than the verifier reads would be charged too little.
    pub log_extended: usize,
    /// Number of committed polynomials the opening reduction batches at once.
    ///
    /// One challenge separates them, and separating `n` costs `(n - 1)/q`.
    pub num_polynomials: usize,
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

/// Largest subspace dimension a term may be computed from.
///
/// Two of the skip terms raise two to a dimension.
/// At or above the word width that shift leaves range.
///
/// It panics in debug builds and drops the term in release.
/// No real statement comes near it, so refusing the shape is the safe reading.
const SHIFT_LIMIT: usize = 63;
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
    /// Variable count of the tallest reader, which the claim point spans.
    pub max_reader_variables: usize,
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
    // Every scalar a term uses is checked before any term is formed.
    // A shape this function cannot charge honestly yields zero bits.
    // The alternative is a term that silently drops, rounds down, or overflows.
    let skip_is_valid = air.skip.is_none_or(|skip| {
        // The transmitted dimension has to exceed the skipped one.
        // Otherwise the term is zero and the round is charged nothing.
        skip.log_extended > skip.log_size
            // Both dimensions reach a shift, kept in range by the widths below.
            && skip.log_extended < SHIFT_LIMIT
            // A round cannot bind more variables than the statement has.
            && skip.log_size <= air.num_variables
            // An empty batch has no claims to separate and no reduction to run.
            && skip.num_polynomials > 0
    });
    if air.num_instances == 0
        || air.num_variables == 0
        || air.constraint_degree == 0
        || !skip_is_valid
    {
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
    //     skip:   a point over m - k, one skip round, then m - k rounds
    //
    // The skipped variables are never drawn into the equality point.
    // The zerocheck term therefore shrinks with the round count.
    let skipped = air.skip.map_or(0, |skip| skip.log_size);
    let kept = air.num_variables.saturating_sub(skipped);

    // A fixed nonzero constraint table has a multilinear extension. Its
    // evaluation at tau vanishes with probability at most h/(q-1). The tail
    // inherited from GKR is also rejection-sampled over the nonzero elements.
    add("zerocheck", kept as f64, nonzero_field_bits);

    if let Some(skip) = air.skip {
        // The verifier reconstructs the round polynomial itself, admitting any
        // below the extension size that vanishes on the subspace.
        //
        //     admitted   2^(k + e) - 1        what the interpolation accepts
        //     honest     d * (2^k - 1)        what a correct prover sends
        //
        // A dishonest message is separated over whichever of the two is wider.
        // A narrow transmitted domain does not make the round cheap.
        // It makes the honest degree binding, and honest proofs stop fitting.
        let admitted = ((1u64 << skip.log_extended) - 1) as f64;
        let honest = air.constraint_degree as f64 * ((1u64 << skip.log_size) - 1) as f64;
        add("skip-reconstruction", admitted.max(honest), field_bits);

        // Collapsing the blend takes one degree-two round per skipped variable.
        add("skip-opening", skipped as f64 * 2.0, field_bits);

        // One challenge separates the polynomials the blend is taken over.
        // Its powers keep their claims apart, so a wider batch costs more.
        add(
            "skip-opening-batching",
            skip.num_polynomials.saturating_sub(1) as f64,
            field_bits,
        );
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
        // One table's challenge varies while every other table's is held fixed.
        //
        // The others then move the identity by a constant rather than cancelling it.
        //
        // Clearing that constant against the denominator adds one degree.
        //
        // The event is bounded by the largest table's pole count.
        //
        // Summing over tables is what is charged.
        //
        // That is safe, and looser than the bound by up to the table count.
        //
        // Each challenge is rejection-sampled away from zero.
        //
        // The first entry embeds to zero, and would otherwise zero a denominator.
        add(
            "logup-star-entry-challenge",
            logup_star.num_leaves as f64,
            nonzero_field_bits,
        );
        // The reduction also inherits the point its claims are taken at.
        //
        // Turning `X(r) = <T, I_* eq_r>` into `X = T . I` is a Schwartz-Zippel event.
        //
        // So is the range check the argument gets for free.
        //
        // One wrong pulled value survives every check above where its weight vanishes.
        //
        // The weight is multilinear in the reader's own variables.
        //
        // Its degree is therefore their count.
        add(
            "logup-star-claim-point",
            logup_star.max_reader_variables as f64,
            field_bits,
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

    /// One skip shape, with the three numbers every test below varies.
    const fn skip(
        log_size: usize,
        log_extended: usize,
        num_polynomials: usize,
    ) -> MultilinearSkipParams {
        MultilinearSkipParams {
            log_size,
            log_extended,
            num_polynomials,
        }
    }

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

    /// The plain shape the skip variants are compared against.
    fn plain(num_variables: usize) -> MultilinearAirParams {
        MultilinearAirParams {
            num_instances: 1,
            max_num_constraints: 1,
            num_variables,
            constraint_degree: 2,
            lookup: None,
            logup_star: None,
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
        // Fixture state: 10 variables, 6 skipped, transmitted on dimension 7.
        //
        //     zerocheck            4          the point covers kept only
        //     constraint-sumcheck  4 * 3      four rounds, not ten
        //     skip-reconstruction  2^7 - 1    the degree admitted
        //     skip-opening         6 * 2      one degree-two round per skip
        let mut params = plain(10);
        params.skip = Some(skip(6, 7, 4));
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
        assert_eq!(
            charged(&terms, "skip-opening-batching"),
            Some(100.0 - libm::log2(3.0))
        );
    }

    #[test]
    fn the_honest_degree_binds_when_the_transmitted_domain_is_narrow() {
        // Fixture state: degree 5, 6 skipped, one extra dimension sent.
        //
        //     admitted  2^7 - 1        = 127
        //     honest    5 * (2^6 - 1)  = 315
        //
        // The wider of the two is what a dishonest message is separated over.
        let mut params = plain(10);
        params.constraint_degree = 5;
        params.skip = Some(skip(6, 7, 1));

        assert_eq!(
            charged(&reduction_terms(&params, 100, 99), "skip-reconstruction"),
            Some(100.0 - libm::log2(315.0))
        );
    }

    #[test]
    fn the_admitted_degree_binds_when_the_transmitted_domain_is_wide() {
        // Fixture state: degree 2, 6 skipped, two extra dimensions sent.
        //
        //     admitted  2^8 - 1        = 255
        //     honest    2 * (2^6 - 1)  = 126
        //
        // Here the interpolation is what admits more, which is the usual case.
        let mut params = plain(10);
        params.constraint_degree = 2;
        params.skip = Some(skip(6, 8, 1));

        assert_eq!(
            charged(&reduction_terms(&params, 100, 99), "skip-reconstruction"),
            Some(100.0 - libm::log2(255.0))
        );
    }

    #[test]
    fn a_single_polynomial_batch_costs_no_separating_draw() {
        // One claim needs no challenge to keep it apart from anything.
        let mut params = plain(10);
        params.skip = Some(skip(6, 7, 1));

        assert_eq!(
            charged(&reduction_terms(&params, 100, 99), "skip-opening-batching"),
            None
        );
    }

    #[test]
    fn a_skip_shape_no_term_can_charge_honestly_is_refused() {
        // Each of these would otherwise be charged wrongly rather than refused:
        //
        //     transmitted <= skipped   the term is zero, so it vanishes
        //     transmitted at 63        the shift leaves range
        //     skipped > variables      a round wider than the statement
        //     empty batch              nothing to separate, nothing to run
        for (log_size, log_extended, num_polynomials) in
            [(6, 6, 1), (6, 5, 1), (6, 63, 1), (11, 12, 1), (6, 7, 0)]
        {
            let mut params = plain(10);
            params.skip = Some(skip(log_size, log_extended, num_polynomials));
            let terms = reduction_terms(&params, 100, 99);

            assert_eq!(
                terms.len(),
                1,
                "{log_size} {log_extended} {num_polynomials}"
            );
            assert_eq!(terms[0].label, "invalid-multilinear-shape");
            assert_eq!(terms[0].bits.bits(), 0.0);
        }
    }

    #[test]
    fn a_skip_shape_every_term_can_charge_is_accepted() {
        // The boundary just inside each refusal, so the guard is not too tight.
        for (log_size, log_extended, num_polynomials) in [(6, 7, 1), (10, 11, 1), (6, 62, 1)] {
            let mut params = plain(10);
            params.skip = Some(skip(log_size, log_extended, num_polynomials));
            let terms = reduction_terms(&params, 100, 99);

            assert!(
                terms.len() > 1,
                "{log_size} {log_extended} {num_polynomials}"
            );
        }
    }

    #[test]
    fn the_reconstruction_term_dominates_what_the_rounds_gave_up() {
        // The skip is not a free saving in soundness.
        // It trades many small round terms for one large reconstruction term.
        // Nothing may read the skip as strictly cheaper, so this pins it.
        let mut params = plain(10);
        params.skip = Some(skip(6, 7, 1));

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

        // The rounds the skip gave up: the plain terms, less the kept ones.
        // Fewer bits is more error, so the comparison is on error either side.
        let error = |bits: f64| libm::exp2(-bits);
        let given_up = error(charged(&plain_terms, "zerocheck").unwrap())
            + error(charged(&plain_terms, "constraint-sumcheck").unwrap())
            - error(charged(&skip_terms, "zerocheck").unwrap())
            - error(charged(&skip_terms, "constraint-sumcheck").unwrap());
        let reconstruction = error(charged(&skip_terms, "skip-reconstruction").unwrap());

        // The one term the skip adds outweighs every term it removed.
        assert!(
            reconstruction > given_up,
            "reconstruction={reconstruction} given_up={given_up}"
        );

        // And so the composed budget is worse, not better.
        assert!(
            total(&skip_terms) < total(&plain_terms),
            "the skip costs bits overall: skip={} plain={}",
            total(&skip_terms),
            total(&plain_terms)
        );
    }

    #[test]
    fn skipping_every_variable_leaves_no_rounds_to_charge() {
        // The widest legal skip binds the whole cube, so residual terms go
        // than going negative.
        let mut params = plain(6);
        params.skip = Some(skip(6, 7, 1));
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
            skip: None,
            logup_star: Some(MultilinearLogupStarParams {
                num_variables: 5,
                num_leaves: 24,
                max_readers_per_table: 3,
                max_reader_variables: 3,
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

        // The tallest reader's variables, over the field its claim point comes from.
        assert!((find("logup-star-claim-point") - (100.0 - libm::log2(3.0))).abs() < 1e-12);
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
                skip: None,
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

    #[test]
    fn the_bit_ring_switch_charges_its_draws_and_its_rounds() {
        // Invariant: one reduction costs (absorbed + 2 * surviving) / |F|.
        //
        //     absorbed = 7, surviving = 4   ->  7 + 8  = 15 draws
        //     one reduction, 128-bit field  ->  128 - log2(15) bits
        let one = bit_ring_switch_error(1, 7, 4, 128).bits();
        assert!((one - (128.0 - log2(15.0))).abs() < 1e-9, "{one}");

        // Reductions at different points share no challenge, so k of them cost k times that.
        //
        //     4 reductions  ->  60 draws  ->  exactly two bits below one reduction
        let four = bit_ring_switch_error(4, 7, 4, 128).bits();
        assert!((one - four - 2.0).abs() < 1e-9, "{one} {four}");

        // A wider field buys bits one for one, since the width enters additively.
        let gap =
            bit_ring_switch_error(4, 7, 4, 128).bits() - bit_ring_switch_error(4, 7, 4, 64).bits();
        assert!((gap - 64.0).abs() < 1e-9, "{gap}");

        // Nothing to reduce and nothing to round over both leave no error to charge.
        assert!(bit_ring_switch_error(0, 7, 4, 128).bits().is_infinite());
        assert!(bit_ring_switch_error(3, 0, 0, 128).bits().is_infinite());

        // The term carries the same number under its own label.
        let term = bit_ring_switch_term(4, 7, 4, 128);
        assert_eq!(term.label, BIT_RING_SWITCH_LABEL);
        assert_eq!(term.bits.bits(), four);
    }

    #[test]
    fn column_batching_charges_one_coordinate_per_batch() {
        let one = column_batch_error(1, 3, 128).bits();
        let four = column_batch_error(4, 3, 128).bits();
        assert!((one - (128.0 - log2(3.0))).abs() < 1e-9, "{one}");
        assert!((one - four - 2.0).abs() < 1e-9, "{one} {four}");
        assert!(column_batch_error(0, 3, 128).bits().is_infinite());
        assert!(column_batch_error(4, 0, 128).bits().is_infinite());

        let term = column_batch_term(4, 3, 128);
        assert_eq!(term.label, COLUMN_BATCH_LABEL);
        assert_eq!(term.bits.bits(), four);
    }

    #[test]
    fn column_batching_does_not_cap_an_overflowing_count() {
        let batches = usize::MAX;
        let coordinates = usize::MAX;
        let expected = 128.0 - log2(batches as f64) - log2(coordinates as f64);
        let actual = column_batch_error(batches, coordinates, 128).bits();
        assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
    }

    #[test]
    fn successor_tensors_charge_their_batching_degree() {
        // Invariant: K tensors batched under powers of alpha add K - 1 to the per-reduction degree.
        //
        //     K = 1   ->  the plain reduction, bit for bit
        //     K = 3   ->  (7 + 2 + 2 * 4) / 2^128 per reduction
        assert_eq!(
            bit_ring_switch_tensors_error(4, 1, 7, 4, 128).bits(),
            bit_ring_switch_error(4, 7, 4, 128).bits()
        );
        let three = bit_ring_switch_tensors_error(1, 3, 7, 4, 128).bits();
        assert!((three - (128.0 - libm::log2(17.0))).abs() < 1e-9);
        let term = bit_ring_switch_tensors_term(1, 3, 7, 4, 128);
        assert_eq!(term.label, BIT_RING_SWITCH_LABEL);
    }
}
