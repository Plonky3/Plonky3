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
    Repeated,
    Nonlinear,
    Selectors,
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
            Kind::Conditional => builder.push_bus_interaction(
                name,
                BusDirection::Push,
                values[..self.width - 1].iter().cloned(),
                BusActivation::Boolean(values[self.width - 1].clone()),
            ),
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

fn check<F, EF, R>(to_repr: fn(EF) -> R, from_repr: fn(R) -> EF)
where
    F: Field,
    EF: ExtensionField<F>,
    R: Field + Algebra<F>,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    let fixtures = [
        (4, "eligible", Kind::Conditional, 7, true),
        (3, "narrow", Kind::Conditional, 4, false),
        (128, "repeated", Kind::Repeated, 3, false),
        (4, "nonlinear", Kind::Nonlinear, 5, false),
        (8, "selectors", Kind::Selectors, 2, true),
        (4, "single-row", Kind::Always, 0, false),
        (4, "always", Kind::Always, 5, true),
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
                    .map(|_| rng.random())
                    .collect(),
                values: Vec::new(),
            },
        };
        let direction = rng.random();
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
                    fixtures[index].4,
                    "projection choice for {}",
                    fixtures[index].1
                );
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
                    for (term, original) in actual.terms.iter().zip(&reference.terms) {
                        assert_eq!(term.row_claim, original.row_claim);
                    }
                }
            }
            for round in 0..num_variables {
                assert_eq!(
                    projected.round_poly(),
                    generic.round_poly(),
                    "round {round}, packed {packed}"
                );
                let challenge = rng.random();
                projected.fold(challenge);
                generic.fold(challenge);
            }
        }
    }
}

#[test]
fn affine_projection_preserves_every_round_and_bounds_its_table_storage() {
    check::<BinaryField128, BinaryField128, BinaryField128>(
        core::convert::identity,
        core::convert::identity,
    );
    check::<BinaryField128, BinaryField128, Ghash128>(Ghash128::from, BinaryField128::from);
    type EF = BinomialExtensionField<BabyBear, 4>;
    check::<BabyBear, EF, EF>(core::convert::identity, core::convert::identity);
}
