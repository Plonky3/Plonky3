use alloc::borrow::Cow;
use alloc::vec;

use p3_air::{Air, BaseAir, WindowAccess};
use p3_baby_bear::BabyBear;
use p3_binary_field::{BinaryField128, Ghash128};
use p3_bus::{
    BusActivation, BusBoundary, BusChallenges, BusInteractionBuilder, BusName, ProductGkrOutput,
};
use p3_field::extension::BinomialExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;

#[derive(Clone, Copy)]
enum Kind {
    Conditional,
    Always,
    Mixed,
    Repeated,
    Nonlinear,
    Selectors,
    Shared,
    Distinct,
    TooMany,
    SharedBoundary,
    NonlinearActivation,
    Cancellation,
}

struct ProjectionAir {
    width: usize,
    name: &'static str,
    kind: Kind,
}

impl<F: Field> BaseAir<F> for ProjectionAir {
    fn width(&self) -> usize {
        self.width
    }

    fn preprocessed_width(&self) -> usize {
        2 * usize::from(matches!(self.kind, Kind::Selectors))
    }

    fn num_public_values(&self) -> usize {
        1
    }

    fn num_periodic_columns(&self) -> usize {
        2 * usize::from(matches!(self.kind, Kind::Selectors))
    }

    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
        if matches!(self.kind, Kind::Selectors) {
            Cow::Owned(vec![
                vec![F::ONE, F::GENERATOR],
                vec![F::ZERO, F::ONE, F::ONE, F::GENERATOR],
            ])
        } else {
            Cow::Borrowed(&[])
        }
    }
}

impl<AB> Air<AB> for ProjectionAir
where
    AB: BusInteractionBuilder,
    AB::F: Field,
    AB::Expr: From<AB::PeriodicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        let values = row
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<AB::Expr>>();
        let name = BusName::new(self.name);
        match self.kind {
            Kind::Cancellation => {
                for direction in BusDirection::ALL {
                    builder.push_bus_interaction(
                        name,
                        direction,
                        values[..self.width - 1].iter().cloned(),
                        BusActivation::Boolean(values[self.width - 1].clone()),
                    );
                }
            }
            Kind::Conditional => builder.push_bus_interaction(
                name,
                BusDirection::Push,
                values[..self.width - 1].iter().cloned(),
                BusActivation::Boolean(values[self.width - 1].clone()),
            ),
            Kind::Mixed => {
                for activation in [
                    BusActivation::Always,
                    BusActivation::Boolean(values[self.width - 1].clone()),
                ] {
                    builder.push_bus_interaction(
                        name,
                        BusDirection::Push,
                        values[..self.width - 1].iter().cloned(),
                        activation,
                    );
                }
            }
            Kind::Always | Kind::Nonlinear => {
                builder.push_bus_interaction(
                    name,
                    BusDirection::Pull,
                    values.clone(),
                    BusActivation::Always,
                );
                if matches!(self.kind, Kind::Nonlinear) {
                    let mut fields = values.clone();
                    fields[0] = values[0].clone() * values[0].clone();
                    builder.push_bus_interaction(
                        name,
                        BusDirection::Push,
                        fields,
                        BusActivation::Always,
                    );
                }
            }
            Kind::Repeated => {
                // A large declared width must not disguise that only one column is read.
                for _ in 0..4 {
                    builder.push_bus_interaction(
                        name,
                        BusDirection::Push,
                        [values[0].clone()],
                        BusActivation::Always,
                    );
                }
            }
            Kind::Shared | Kind::Distinct | Kind::TooMany | Kind::NonlinearActivation => {
                let selectors = match self.kind {
                    Kind::Distinct => 2,
                    Kind::TooMany => 3,
                    _ => 1,
                };
                let payloads = self.width - selectors;
                for (index, value) in values[..payloads].iter().enumerate() {
                    let mut selector = values[payloads + index % selectors].clone();
                    if matches!(self.kind, Kind::NonlinearActivation) && index + 1 == payloads {
                        selector = selector.clone() * selector;
                    }
                    // Named domains and directions do not prevent grouping when
                    // the activation polynomial and AIR row point agree.
                    let (bus, direction, fields) = if index % 2 == 0 {
                        (name, BusDirection::Push, vec![value.clone()])
                    } else {
                        let public: AB::Expr = builder.public_values()[0].into();
                        (
                            BusName::new("shared-pair"),
                            BusDirection::Pull,
                            vec![value.clone(), value.clone() + public],
                        )
                    };
                    builder.push_bus_interaction(
                        bus,
                        direction,
                        fields,
                        BusActivation::Boolean(selector),
                    );
                }
            }
            Kind::SharedBoundary => {
                for (index, value) in values.iter().enumerate() {
                    // Extraction cancels the repeated column before activation hashing.
                    let activation = if index == 0 {
                        BusActivation::Boundary(BusBoundary::First)
                    } else {
                        BusActivation::Boolean(
                            builder.is_first_row() + value.clone() - value.clone(),
                        )
                    };
                    builder.push_bus_interaction(
                        name,
                        BusDirection::Push,
                        [value.clone()],
                        activation,
                    );
                }
            }
            Kind::Selectors => {
                let fixed = builder.preprocessed();
                let prep = fixed.current_slice();
                let periodic = builder.periodic_values();
                let public: AB::Expr = builder.public_values()[0].into();
                let fixed_value: AB::Expr = prep[0].into();
                let mut fields = values;
                fields.push(fixed_value - prep[1] + public.clone());
                fields.push(
                    AB::Expr::from(periodic[0])
                        + AB::Expr::from(periodic[1])
                        + builder.is_transition()
                        - builder.is_last_row(),
                );
                let selector = public * builder.is_first_row() - builder.is_last_row()
                    + builder.is_transition();
                for (direction, activation) in [
                    (
                        BusDirection::Push,
                        BusActivation::Boundary(BusBoundary::First),
                    ),
                    (
                        BusDirection::Pull,
                        BusActivation::Boundary(BusBoundary::Last),
                    ),
                    (BusDirection::Push, BusActivation::Boolean(selector)),
                ] {
                    builder.push_bus_interaction(name, direction, fields.clone(), activation);
                }
            }
        }
    }
}

fn table<F: Field>(width: usize, log_height: usize, packed: bool, rng: &mut SmallRng) -> Table<F>
where
    StandardUniform: Distribution<F>,
{
    let height = 1usize << log_height;
    if packed {
        let mut words = (0..width * height.div_ceil(64))
            .map(|_| rng.random::<u64>())
            .collect::<Vec<u64>>();
        if height < 64 {
            for word in &mut words {
                *word &= u64::MAX >> (64 - height);
            }
        }
        Table::from_packed_bits(RowMajorMatrix::new(words, width), log_height)
    } else {
        Table::new(RowMajorMatrix::new(
            (0..width * height).map(|_| rng.random()).collect(),
            height,
        ))
    }
}

fn check<F, EF, R>(
    to_repr: fn(EF) -> R,
    from_repr: fn(R) -> EF,
    anchor_kind: usize,
    fold_kind: usize,
) where
    F: Field,
    EF: ExtensionField<F>,
    R: Field + Algebra<F>,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    let fixtures = [
        (4, "eligible", Kind::Conditional, 7, Some(2)),
        (16, "short-wide", Kind::Conditional, 2, Some(2)),
        (3, "narrow", Kind::Conditional, 4, None),
        (128, "repeated", Kind::Repeated, 3, None),
        (4, "nonlinear", Kind::Nonlinear, 5, None),
        (8, "selectors", Kind::Selectors, 2, Some(6)),
        (4, "single-row", Kind::Always, 0, None),
        (4, "always", Kind::Always, 5, Some(1)),
        (8, "mixed", Kind::Mixed, 4, Some(3)),
        (8, "shared", Kind::Shared, 3, Some(2)),
        (8, "distinct", Kind::Distinct, 4, Some(4)),
        (8, "too-many", Kind::TooMany, 4, None),
        (8, "shared-boundary", Kind::SharedBoundary, 5, Some(2)),
        (
            8,
            "nonlinear-activation",
            Kind::NonlinearActivation,
            3,
            None,
        ),
    ];
    let airs = fixtures
        .iter()
        .map(|&(width, name, kind, ..)| ProjectionAir { width, name, kind })
        .collect::<Vec<_>>();
    let air_refs = airs.iter().collect::<Vec<_>>();
    let heights = fixtures.iter().map(|f| f.3).collect::<Vec<_>>();
    let context = BusContext::<F, EF>::build(&air_refs, &heights)
        .unwrap()
        .unwrap();
    // A generic cubic body raises the shared degree above projected quadratic bodies.
    assert_eq!(context.composition_degree(), 4);
    let periodic = context.period_vectors(&air_refs, &heights);
    let public = [F::GENERATOR];
    let public_values = vec![public.as_slice(); airs.len()];
    let mut rng = SmallRng::seed_from_u64(0x0aff_1b05);
    for packed in [false, true] {
        let tables = fixtures
            .iter()
            .map(|f| table(f.0, f.3, packed, &mut rng))
            .collect::<Vec<_>>();
        let table_refs = tables.iter().collect::<Vec<_>>();
        let fixed = fixtures
            .iter()
            .map(|f| matches!(f.2, Kind::Selectors).then(|| table(2, f.3, !packed, &mut rng)))
            .collect::<Vec<_>>();
        let fixed_refs = fixed.iter().map(Option::as_ref).collect::<Vec<_>>();
        let output = BusReductionOutput {
            challenges: BusChallenges {
                fingerprint: (0..log2_strict_usize(context.plan().fingerprint_width()))
                    .map(|_| rng.random())
                    .collect(),
                offset: rng.random(),
            },
            product: ProductGkrOutput {
                roots: Vec::new(),
                point: (0..context.plan().security_geometry().log_logical_leaf_count())
                    .map(|index| match anchor_kind {
                        0 => rng.random(),
                        1 => EF::ZERO,
                        2 => EF::ONE,
                        _ => EF::from_bool(index % 2 == 0),
                    })
                    .collect(),
                values: Vec::new(),
            },
        };
        for direction in [EF::ZERO, -EF::ONE, rng.random()] {
            for num_variables in [7, 9] {
                let make = || {
                    BusCompositionProver::new(
                        &context,
                        &output,
                        &table_refs,
                        &fixed_refs,
                        &periodic,
                        &public_values,
                        direction,
                        num_variables,
                        to_repr,
                        from_repr,
                    )
                };
                let mut projected = make();
                let mut generic = make();
                for (index, (actual, reference)) in
                    projected.airs.iter().zip(&mut generic.airs).enumerate()
                {
                    assert_eq!(
                        actual.projected.is_some(),
                        fixtures[index].4.is_some(),
                        "projection choice for {}",
                        fixtures[index].1
                    );
                    if let Some(projection) = &actual.projected {
                        let tables = projection
                            .groups
                            .iter()
                            .map(|group| 1 + usize::from(group.activation.is_some()))
                            .sum::<usize>();
                        assert_eq!(Some(tables), fixtures[index].4);
                        assert!(
                            tables
                                <= (actual.main.len()
                                    + actual.preprocessed.len()
                                    + context.periodic_columns(index).len())
                                    / 2
                        );
                    }
                    reference.projected = None;
                    // Projection may release its now-unused owned period vectors.
                    reference.periodic = context
                        .periodic_columns(index)
                        .iter()
                        .map(|&column| Source::Periodic {
                            period: periodic[index].as_ref().unwrap()[column].clone(),
                            height: 1 << heights[index],
                        })
                        .collect();
                    if reference.unused_prefix > 0 {
                        reference.initialize_claims(
                            &context,
                            &generic.fingerprint_weights,
                            generic.offset,
                        );
                        let weighted_claim = |air: &AirState<'_, F, R>| {
                            air.terms
                                .iter()
                                .map(|term| term.coefficient * term.row_claim)
                                .sum::<R>()
                        };
                        assert_eq!(
                            actual.projected.as_ref().map_or_else(
                                || weighted_claim(actual),
                                |projection| projection.row_claim
                            ),
                            weighted_claim(reference),
                        );
                    }
                }
                for round in 0..num_variables {
                    assert_eq!(
                        projected.round_poly(),
                        generic.round_poly(),
                        "round {round}, packed {packed}"
                    );
                    let challenge = match fold_kind {
                        0 => rng.random(),
                        1 => EF::ZERO,
                        _ => EF::ONE,
                    };
                    projected.fold(challenge);
                    generic.fold(challenge);
                    for (actual, reference) in projected.airs.iter().zip(&generic.airs) {
                        if let Some(projection) = &actual.projected {
                            let scaled = actual
                                .equality
                                .as_slice()
                                .iter()
                                .map(|&value| projection.equality_prefix * value)
                                .collect::<Vec<_>>();
                            assert_eq!(scaled.as_slice(), reference.equality.as_slice());
                            // The suffix remains an equality table even after its prefix is zero.
                            let remaining = actual.equality.num_variables();
                            let anchor =
                                &output.product.point[output.product.point.len() - remaining..];
                            let expected =
                                Point::new(anchor.iter().copied().map(to_repr).collect::<Vec<_>>())
                                    .equality_weights_msb();
                            assert_eq!(actual.equality.as_slice(), expected.as_slice());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn affine_projection_preserves_every_round_and_bounds_its_table_storage() {
    for anchor_kind in 0..4 {
        for fold_kind in 0..3 {
            check::<BinaryField128, BinaryField128, BinaryField128>(
                core::convert::identity,
                core::convert::identity,
                anchor_kind,
                fold_kind,
            );
            check::<BinaryField128, BinaryField128, Ghash128>(
                Ghash128::from,
                BinaryField128::from,
                anchor_kind,
                fold_kind,
            );
            type EF = BinomialExtensionField<BabyBear, 4>;
            check::<BabyBear, EF, EF>(
                core::convert::identity,
                core::convert::identity,
                anchor_kind,
                fold_kind,
            );
        }
    }
}

fn check_cancellation<F, EF, R>(to_repr: fn(EF) -> R, from_repr: fn(R) -> EF)
where
    F: Field,
    EF: ExtensionField<F>,
    R: Field + Algebra<F>,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    let air = ProjectionAir {
        width: 4,
        name: "cancel",
        kind: Kind::Cancellation,
    };
    let context = BusContext::<F, EF>::build(&[&air], &[3]).unwrap().unwrap();
    let mut rng = SmallRng::seed_from_u64(0xcace_11ed);
    let output = BusReductionOutput {
        challenges: BusChallenges {
            fingerprint: (0..log2_strict_usize(context.plan().fingerprint_width()))
                .map(|_| rng.random())
                .collect(),
            offset: rng.random(),
        },
        product: ProductGkrOutput {
            roots: Vec::new(),
            point: (0..context.plan().security_geometry().log_logical_leaf_count())
                .map(|_| rng.random())
                .collect(),
            values: Vec::new(),
        },
    };
    for packed in [false, true] {
        let table = table(4, 3, packed, &mut rng);
        let make = || {
            BusCompositionProver::new(
                &context,
                &output,
                &[&table],
                &[None],
                &[None],
                &[&[F::GENERATOR]],
                -EF::ONE,
                5,
                to_repr,
                from_repr,
            )
        };
        let mut projected = make();
        let mut generic = make();
        let projection = projected.airs[0].projected.as_ref().unwrap();
        assert_eq!(projection.groups.len(), 1);
        assert_eq!(projection.row_claim, R::ZERO);
        assert!(
            projection.groups[0]
                .difference
                .as_slice()
                .iter()
                .all(|&x| x == R::ZERO)
        );
        generic.airs[0].projected = None;
        generic.airs[0].initialize_claims(&context, &generic.fingerprint_weights, generic.offset);
        for _ in 0..5 {
            let round = projected.round_poly();
            assert!(round.iter().all(|&x| x == EF::ZERO));
            assert_eq!(round, generic.round_poly());
            let challenge = rng.random();
            projected.fold(challenge);
            generic.fold(challenge);
        }
    }
}

#[test]
fn grouped_bus_terms_cancel_at_every_round() {
    check_cancellation::<BinaryField128, BinaryField128, BinaryField128>(
        core::convert::identity,
        core::convert::identity,
    );
    check_cancellation::<BinaryField128, BinaryField128, Ghash128>(
        Ghash128::from,
        BinaryField128::from,
    );
    type EF = BinomialExtensionField<BabyBear, 4>;
    check_cancellation::<BabyBear, EF, EF>(core::convert::identity, core::convert::identity);
}
