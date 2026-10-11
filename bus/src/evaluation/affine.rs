//! Affine projections of a compiled bus factor, retaining activation separately.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use p3_field::{Algebra, Field};

use super::{BusEvaluationError, BusFactorPlan, BusNode};

/// An affine expression in current-row columns and multilinear row selectors.
///
/// Column entries are sorted by their original indices, with duplicates merged
/// and zero coefficients omitted. Public inputs have been absorbed into constants.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct BusAffineExpression<R> {
    /// Constant term, including any fixed public-input contribution.
    pub constant: R,
    /// Main-column coefficients.
    pub main: Vec<(usize, R)>,
    /// Preprocessed-column coefficients.
    pub preprocessed: Vec<(usize, R)>,
    /// Periodic-column coefficients.
    pub periodic: Vec<(usize, R)>,
    /// First-row selector coefficient.
    pub first_row: R,
    /// Last-row selector coefficient.
    pub last_row: R,
    /// Transition selector coefficient.
    pub transition: R,
}

/// A selected leaf factor written as `1 + factor_minus_one * activation`.
///
/// An absent activation means one. The two expressions must remain separate
/// multilinears: interpolating their rowwise product changes the polynomial.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct BusAffineFactor<R> {
    /// Unselected fingerprint factor minus its multiplicative identity.
    pub factor_minus_one: BusAffineExpression<R>,
    /// Conditional activation, if present in the declaration.
    pub activation: Option<BusAffineExpression<R>>,
}

#[derive(Clone, Copy)]
enum Affinity<R> {
    Constant(R),
    Affine,
    Nonlinear,
}

impl<F, R> BusFactorPlan<'_, F, R>
where
    F: Field,
    R: Field + Algebra<F>,
{
    /// Extract affine factor and activation expressions after substituting public inputs.
    ///
    /// Conservatively returns `None` for nonlinear payloads or activations.
    /// In particular, Boolean-packed source values do not make `x * x` affine
    /// away from the Boolean cube. The returned coefficients own their storage.
    ///
    /// # Errors
    /// Returns an error if a referenced public input is missing.
    pub fn affine(&self, public: &[F]) -> Result<Option<BusAffineFactor<R>>, BusEvaluationError> {
        use Affinity::{Affine, Constant, Nonlinear};

        // Classify each shared node once, with public inputs treated as constants.
        // A Boolean column remains a variable at a non-Boolean evaluation point.
        let mut classes: Vec<Affinity<R>> = Vec::with_capacity(self.nodes.len());
        for &node in &self.nodes {
            let class = match node {
                BusNode::Constant(value) => Constant(value.into()),
                BusNode::Public(index) => Constant(
                    (*public
                        .get(index)
                        .ok_or(BusEvaluationError::PublicValue { index })?)
                    .into(),
                ),
                BusNode::Add(x, y) | BusNode::Sub(x, y) => match (classes[x], classes[y]) {
                    (Constant(a), Constant(b)) => Constant(if matches!(node, BusNode::Add(..)) {
                        a + b
                    } else {
                        a - b
                    }),
                    (Nonlinear, _) | (_, Nonlinear) => Nonlinear,
                    _ => Affine,
                },
                BusNode::Neg(x) => match classes[x] {
                    Constant(value) => Constant(-value),
                    class => class,
                },
                BusNode::Mul(x, y) => match (classes[x], classes[y]) {
                    (Constant(a), Constant(b)) => Constant(a * b),
                    (Constant(a), _) | (_, Constant(a)) if a == R::ZERO => Constant(R::ZERO),
                    (Nonlinear, _) | (_, Nonlinear) => Nonlinear,
                    (Constant(_), _) | (_, Constant(_)) => Affine,
                    _ => Nonlinear,
                },
                _ => Affine,
            };
            classes.push(class);
        }
        if self
            .payload
            .iter()
            .chain(self.activation.iter())
            .any(|&root| matches!(classes[root], Nonlinear))
        {
            return Ok(None);
        }

        let factor_minus_one = self.affine_expression(
            &classes,
            self.payload
                .iter()
                .zip(self.payload_weights)
                .map(|(&root, &weight)| (root, -weight)),
            self.shifted_offset - R::ONE,
        );
        let activation = self
            .activation
            .map(|root| self.affine_expression(&classes, [(root, R::ONE)], R::ZERO));
        Ok(Some(BusAffineFactor {
            factor_minus_one,
            activation,
        }))
    }

    /// Propagate coefficients backward through the DAG, without expanding shared paths.
    fn affine_expression(
        &self,
        classes: &[Affinity<R>],
        roots: impl IntoIterator<Item = (usize, R)>,
        constant: R,
    ) -> BusAffineExpression<R> {
        let mut expression = BusAffineExpression {
            constant,
            main: Vec::new(),
            preprocessed: Vec::new(),
            periodic: Vec::new(),
            first_row: R::ZERO,
            last_row: R::ZERO,
            transition: R::ZERO,
        };
        let mut main = BTreeMap::new();
        let mut preprocessed = BTreeMap::new();
        let mut periodic = BTreeMap::new();
        let mut weights = R::zero_vec(self.nodes.len());
        for (root, weight) in roots {
            weights[root] += weight;
        }
        for index in (0..self.nodes.len()).rev() {
            let weight = weights[index];
            if weight == R::ZERO {
                continue;
            }
            if let Affinity::Constant(value) = classes[index] {
                expression.constant += weight * value;
                continue;
            }
            match self.nodes[index] {
                BusNode::Main(column) => *main.entry(column).or_insert(R::ZERO) += weight,
                BusNode::Preprocessed(column) => {
                    *preprocessed.entry(column).or_insert(R::ZERO) += weight;
                }
                BusNode::Periodic(column) => *periodic.entry(column).or_insert(R::ZERO) += weight,
                BusNode::IsFirstRow => expression.first_row += weight,
                BusNode::IsLastRow => expression.last_row += weight,
                BusNode::IsTransition => expression.transition += weight,
                BusNode::Add(x, y) => {
                    weights[x] += weight;
                    weights[y] += weight;
                }
                BusNode::Sub(x, y) => {
                    weights[x] += weight;
                    weights[y] -= weight;
                }
                BusNode::Neg(x) => weights[x] -= weight,
                BusNode::Mul(x, y) => match (classes[x], classes[y]) {
                    (Affinity::Constant(a), _) => weights[y] += weight * a,
                    (_, Affinity::Constant(a)) => weights[x] += weight * a,
                    _ => unreachable!("only affine roots propagate coefficients"),
                },
                BusNode::Constant(_) | BusNode::Public(_) => {
                    unreachable!("constant nodes handled above")
                }
            }
        }
        let sparse = |columns: BTreeMap<usize, R>| {
            columns
                .into_iter()
                .filter(|&(_, coefficient)| coefficient != R::ZERO)
                .collect()
        };
        expression.main = sparse(main);
        expression.preprocessed = sparse(preprocessed);
        expression.periodic = sparse(periodic);
        expression
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use p3_air::symbolic::{BaseEntry, BaseLeaf, SymbolicExpression, SymbolicVariable};
    use p3_baby_bear::BabyBear;
    use p3_binary_field::{BinaryField32, BinaryField128, Ghash128};
    use p3_field::extension::BinomialExtensionField;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::{
        BusActivation, BusBoundary, BusDirection, BusEvaluation, BusName, BusPlan, BusPlanInput,
        SymbolicBusInteraction,
    };

    fn variable<F: Field>(entry: BaseEntry, index: usize) -> SymbolicExpression<F> {
        SymbolicVariable::new(entry, index).into()
    }

    fn evaluate<F: Field, R: Field + Algebra<F>>(
        expression: &BusAffineExpression<R>,
        values: BusEvaluation<'_, F, R>,
    ) -> R {
        let columns = expression
            .main
            .iter()
            .map(|&(i, c)| c * values.main[i])
            .chain(
                expression
                    .preprocessed
                    .iter()
                    .map(|&(i, c)| c * values.preprocessed[i]),
            )
            .chain(
                expression
                    .periodic
                    .iter()
                    .map(|&(i, c)| c * values.periodic[i]),
            );
        expression.constant
            + columns.sum::<R>()
            + expression.first_row * values.is_first_row
            + expression.last_row * values.is_last_row
            + expression.transition * values.is_transition
    }

    fn check<F: Field, R: Field + Algebra<F>>(samples: &[R]) {
        let x = variable::<F>(BaseEntry::Main { offset: 0 }, 0);
        let y = variable::<F>(BaseEntry::Main { offset: 0 }, 1);
        let fixed = variable::<F>(BaseEntry::Preprocessed { offset: 0 }, 1);
        let periodic = variable::<F>(BaseEntry::Periodic, 0);
        let public = variable::<F>(BaseEntry::Public, 0);
        let first: SymbolicExpression<F> = SymbolicExpression::Leaf(BaseLeaf::IsFirstRow);
        let last: SymbolicExpression<F> = SymbolicExpression::Leaf(BaseLeaf::IsLastRow);
        let transition: SymbolicExpression<F> = SymbolicExpression::Leaf(BaseLeaf::IsTransition);
        for activation in [
            BusActivation::Always,
            BusActivation::Boundary(BusBoundary::First),
            BusActivation::Boundary(BusBoundary::Last),
            BusActivation::Boolean(
                public.clone() * y.clone() + first.clone() - last.clone() + transition.clone(),
            ),
        ] {
            let interaction = SymbolicBusInteraction::<F> {
                bus_name: "affine".to_string(),
                direction: BusDirection::Push,
                fields: vec![
                    x.clone() * public.clone() - fixed.clone() + periodic.clone() + first.clone(),
                    -y.clone() + y.clone(),
                    (last.clone() - transition.clone()) * public.clone(),
                    SymbolicExpression::from(F::from_u8(7)),
                ],
                activation,
            };
            let plan = BusPlan::build(&[BusPlanInput {
                log_height: 3,
                interactions: core::slice::from_ref(&interaction),
            }])
            .unwrap()
            .unwrap();
            let weights: Vec<R> = (0..plan.fingerprint_width())
                .map(|i| samples[i % samples.len()])
                .collect();
            let factor = plan
                .compile_factor(0, &interaction, &weights, samples[9])
                .unwrap();
            assert!(matches!(
                factor.affine(&[]),
                Err(BusEvaluationError::PublicValue { index: 0 })
            ));
            for public in [F::ZERO, F::ONE, F::GENERATOR] {
                let public = [public];
                let affine = factor.affine(&public).unwrap().expect("affine expression");
                for row in samples.windows(9) {
                    let values = BusEvaluation {
                        main: &row[..2],
                        preprocessed: &row[2..4],
                        periodic: &row[4..5],
                        public: &public,
                        is_first_row: row[5],
                        is_last_row: row[6],
                        is_transition: row[7],
                    };
                    let projected = R::ONE
                        + evaluate(&affine.factor_minus_one, values)
                            * affine
                                .activation
                                .as_ref()
                                .map_or(R::ONE, |selector| evaluate(selector, values));
                    assert_eq!(
                        projected,
                        factor.evaluate::<R>(&mut vec![], values).unwrap()
                    );
                }
            }
        }

        for (payload, activation) in [
            (x.clone() * y.clone(), BusActivation::Always),
            (x.clone(), BusActivation::Boolean(y.clone() * y.clone())),
        ] {
            let interaction = SymbolicBusInteraction::<F> {
                bus_name: "nonlinear".to_string(),
                direction: BusDirection::Push,
                fields: vec![payload],
                activation,
            };
            let plan = BusPlan::build(&[BusPlanInput {
                log_height: 3,
                interactions: core::slice::from_ref(&interaction),
            }])
            .unwrap()
            .unwrap();
            let weights = vec![R::ONE; plan.fingerprint_width()];
            let factor = plan
                .compile_factor(0, &interaction, &weights, R::ONE)
                .unwrap();
            assert!(factor.affine(&[]).unwrap().is_none());
        }

        // Public zero can erase a nonlinear subexpression. Repeated columns must
        // combine even when they occur in different payload expressions.
        let interactions = [
            SymbolicBusInteraction::<F> {
                bus_name: "short".to_string(),
                direction: BusDirection::Push,
                fields: vec![x.clone() * x.clone() * public + y.clone() - y.clone()],
                activation: BusActivation::Always,
            },
            SymbolicBusInteraction::<F> {
                bus_name: "wide".to_string(),
                direction: BusDirection::Push,
                fields: vec![y.clone(), x, y],
                activation: BusActivation::Always,
            },
        ];
        let plan = BusPlan::build(&[BusPlanInput {
            log_height: 3,
            interactions: &interactions,
        }])
        .unwrap()
        .unwrap();
        let weights: Vec<R> = (0..plan.fingerprint_width())
            .map(|i| samples[i % samples.len()])
            .collect();
        for interaction in &interactions {
            let bus = plan
                .domain_index(BusName::new(&interaction.bus_name))
                .unwrap();
            let factor = plan
                .compile_factor(bus, interaction, &weights, samples[9])
                .unwrap();
            let public = [F::ZERO];
            let affine = factor.affine(&public).unwrap().unwrap();
            if interaction.bus_name == "short" {
                assert!(affine.factor_minus_one.main.is_empty());
                assert!(factor.affine(&[F::ONE]).unwrap().is_none());
            } else {
                assert!(
                    affine
                        .factor_minus_one
                        .main
                        .windows(2)
                        .all(|w| w[0].0 < w[1].0)
                );
                assert!(
                    affine
                        .factor_minus_one
                        .main
                        .iter()
                        .all(|&(_, c)| c != R::ZERO)
                );
            }
            let values = BusEvaluation {
                main: &samples[..2],
                preprocessed: &[],
                periodic: &[],
                public: &public,
                is_first_row: R::ZERO,
                is_last_row: R::ZERO,
                is_transition: R::ZERO,
            };
            assert_eq!(
                R::ONE + evaluate(&affine.factor_minus_one, values),
                factor.evaluate::<R>(&mut vec![], values).unwrap()
            );
        }
    }

    #[test]
    fn affine_projection_matches_the_compiled_factor_at_arbitrary_points() {
        let mut rng = SmallRng::seed_from_u64(0x00af_f14e);
        check::<BabyBear, BinomialExtensionField<BabyBear, 4>>(&[(); 32].map(|_| rng.random()));
        let samples = [(); 32].map(|_| rng.random::<BinaryField128>());
        check::<BinaryField32, BinaryField128>(&samples);
        check::<BinaryField32, Ghash128>(&samples.map(Ghash128::from));
        check::<BinaryField128, Ghash128>(&samples.map(Ghash128::from));
    }
}
