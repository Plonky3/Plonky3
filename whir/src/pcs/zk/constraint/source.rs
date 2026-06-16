//! Symbolic source-side claim tracking.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

/// One symbolic linear constraint on the current source message.
///
/// # Basis kinds
///
/// With the message viewed as a multilinear table over `{0,1}^m`
/// (big-endian index convention):
///
/// ```text
///     Eq  : W[b] = eq(point, b)            evaluation claims
///     Pow : W[b] = var^{index(b)}          OOD and STIR consistency claims
/// ```
///
/// Both kinds are closed under prefix folding.
/// That closure keeps the verifier's bookkeeping symbolic across rounds.
#[derive(Debug, Clone)]
pub enum SourceTerm<EF> {
    /// Equality covector at a multilinear point.
    Eq {
        /// Constraint point, with arity equal to the message arity.
        point: Point<EF>,
    },
    /// Power covector `(var^0, var^1, ..., var^{2^m - 1})`.
    Pow {
        /// Power base.
        var: EF,
        /// Current message arity `m`.
        num_variables: usize,
        /// Squaring ladder `[var^{2^0}, var^{2^1}, ...]`, extended on demand.
        ///
        /// Each fold needs `var^{2^{m-k}}`; the index it reads only ever
        /// decreases, so squares built for one fold serve every later one.
        squares: Vec<EF>,
    },
}

/// A symbolic source term with its accumulated batching scale.
#[derive(Debug, Clone)]
pub struct SourceConstraint<EF> {
    /// Basis covector.
    pub term: SourceTerm<EF>,
    /// Accumulated scale: batching coefficients and sumcheck `eps` factors.
    pub coeff: EF,
}

/// The full symbolic source-side claim.
#[derive(Debug, Clone, Default)]
pub struct SourceClaim<EF> {
    /// Accumulated constraints, in introduction order.
    pub constraints: Vec<SourceConstraint<EF>>,
}

impl<EF: Field> SourceClaim<EF> {
    /// Starts with no constraints.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Records an equality constraint `coeff * eq(point, .)`.
    pub fn push_eq(&mut self, point: Point<EF>, coeff: EF) {
        self.constraints.push(SourceConstraint {
            term: SourceTerm::Eq { point },
            coeff,
        });
    }

    /// Records a power constraint `coeff * (var^index)` over `m` variables.
    pub fn push_pow(&mut self, var: EF, num_variables: usize, coeff: EF) {
        self.constraints.push(SourceConstraint {
            term: SourceTerm::Pow {
                var,
                num_variables,
                squares: vec![var],
            },
            coeff,
        });
    }

    /// Folds every constraint by a prefix of sumcheck challenges.
    ///
    /// # Math
    ///
    /// Folding the first `k` variables at `gamma` rescales each covector in kind:
    ///
    /// ```text
    ///     eq(z, .)   ->  eq(z[..k], gamma) * eq(z[k..], .)
    ///     var^index  ->  select(gamma, pow(var^{2^{m-k}})) * var^index
    /// ```
    ///
    /// The `select` factor sums the power covector against `eq(., gamma)`.
    /// Coordinate `i` carries exponent `2^{m-1-i}`.
    pub fn fold(&mut self, gamma: &Point<EF>) {
        let k = gamma.num_variables();
        for constraint in &mut self.constraints {
            match &mut constraint.term {
                SourceTerm::Eq { point } => {
                    let (head, tail) = point.split_at(k);
                    constraint.coeff *= Point::eval_eq(head.as_slice(), gamma.as_slice());
                    *point = tail;
                }
                SourceTerm::Pow {
                    num_variables,
                    squares,
                    ..
                } => {
                    debug_assert!(k <= *num_variables);
                    // Folding the first k bits scales a power covector by
                    //
                    //     prod_{i < k} (1 - g_i + g_i * var^{2^{m-1-i}})
                    //
                    // select_poly squares the seed var^{2^{m-k}} up to
                    // var^{2^{m-1}}, in reverse coordinate order.
                    //
                    //     seed index m-k shrinks every fold
                    //     -> the ladder from the first fold covers all later ones
                    //     -> extend it once, here
                    let seed_index = *num_variables - k;
                    while squares.len() <= seed_index {
                        squares.push(squares[squares.len() - 1].square());
                    }
                    constraint.coeff *= Point::eval_select(squares[seed_index], gamma.as_slice());
                    *num_variables -= k;
                }
            }
        }
    }

    /// Materializes the dense covector over the remaining message slots.
    ///
    /// Used at the base case, where the remaining message is small.
    pub fn materialize(&self, num_variables: usize) -> Poly<EF> {
        let mut dense = EF::zero_vec(1 << num_variables);
        // Each constraint adds `coeff * basis-term` into the accumulator.
        for constraint in &self.constraints {
            match &constraint.term {
                SourceTerm::Eq { point } => {
                    assert_eq!(point.num_variables(), num_variables);
                    // Dense layer: coeff * eq(point, b) for every slot b.
                    let table = Poly::new_from_point(point.as_slice(), constraint.coeff);
                    for (dst, &src) in dense.iter_mut().zip(table.as_slice()) {
                        *dst += src;
                    }
                }
                SourceTerm::Pow {
                    var,
                    num_variables: m,
                    ..
                } => {
                    assert_eq!(*m, num_variables);
                    // Running power: slot b gains coeff * var^b.
                    // Index order matches the big-endian table layout.
                    let mut power = constraint.coeff;
                    for dst in dense.iter_mut() {
                        *dst += power;
                        power *= *var;
                    }
                }
            }
        }
        Poly::new(dense)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PrimeCharacteristicRing, dot_product};
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    impl<EF: Field> SourceClaim<EF> {
        /// Evaluates the claim against a concrete message.
        #[must_use]
        fn evaluate(&self, message: &[EF]) -> EF {
            let num_variables = p3_util::log2_strict_usize(message.len());
            let dense = self.materialize(num_variables);
            dot_product::<EF, _, _>(message.iter().copied(), dense.as_slice().iter().copied())
        }
    }

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Dense reference fold: W'[c] = sum_b eq(b, gamma) * W[b, c].
    ///
    /// `b` walks the first `k` high-order index bits.
    /// That matches the prefix-binding commit layout.
    fn fold_dense(dense: &[EF], gamma: &Point<EF>) -> Vec<EF> {
        let k = gamma.num_variables();
        let chunk = dense.len() >> k;
        let eq_table = Poly::new_from_point(gamma.as_slice(), EF::ONE);
        let mut out = EF::zero_vec(chunk);
        for (b, &weight) in eq_table.as_slice().iter().enumerate() {
            for (dst, &src) in out.iter_mut().zip(&dense[b * chunk..(b + 1) * chunk]) {
                *dst += weight * src;
            }
        }
        out
    }

    #[test]
    fn pow_term_materializes_running_powers() {
        // Pow over 2 variables at var = 3, coeff = 5: (5, 15, 45, 135).
        let mut claim = SourceClaim::new();
        claim.push_pow(EF::from_u64(3), 2, EF::from_u64(5));
        let dense = claim.materialize(2);
        assert_eq!(
            dense.as_slice(),
            &[
                EF::from_u64(5),
                EF::from_u64(15),
                EF::from_u64(45),
                EF::from_u64(135)
            ]
        );
    }

    #[test]
    fn eq_term_materializes_eq_table() {
        // Eq at a boolean point selects exactly one slot.
        let mut claim = SourceClaim::new();
        claim.push_eq(Point::new(vec![EF::ONE, EF::ZERO]), EF::from_u64(7));
        let dense = claim.materialize(2);
        // Big-endian: point (1, 0) is index 2.
        assert_eq!(
            dense.as_slice(),
            &[
                EF::from_u64(0),
                EF::from_u64(0),
                EF::from_u64(7),
                EF::from_u64(0)
            ]
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_symbolic_fold_matches_dense_fold(
            seed in any::<u64>(),
            m in 2usize..=6,
            k_raw in 1usize..=5,
        ) {
            // Invariant: materialize-then-fold == fold-then-materialize,
            // for both term kinds and arbitrary scales.
            let k = k_raw.min(m - 1);
            let mut rng = SmallRng::seed_from_u64(seed);

            let mut claim = SourceClaim::<EF>::new();
            claim.push_eq(Point::rand(&mut rng, m), rng.random());
            claim.push_pow(rng.random(), m, rng.random());
            claim.push_pow(rng.random(), m, rng.random());

            let dense_before = claim.materialize(m);
            let gamma = Point::rand(&mut rng, k);
            let reference = fold_dense(dense_before.as_slice(), &gamma);

            claim.fold(&gamma);
            let folded = claim.materialize(m - k);

            prop_assert_eq!(folded.as_slice(), reference.as_slice());
        }

        #[test]
        fn prop_evaluate_matches_dense_dot(
            seed in any::<u64>(),
            m in 1usize..=5,
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut claim = SourceClaim::<EF>::new();
            claim.push_pow(rng.random(), m, rng.random());
            claim.push_eq(Point::rand(&mut rng, m), rng.random());

            let message: Vec<EF> = (0..1usize << m).map(|_| rng.random()).collect();
            let dense = claim.materialize(m);
            let expected = dense
                .as_slice()
                .iter()
                .zip(&message)
                .map(|(&w, &v)| w * v)
                .sum::<EF>();

            prop_assert_eq!(claim.evaluate(&message), expected);
        }
    }
}
