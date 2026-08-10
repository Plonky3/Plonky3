use alloc::vec;
use alloc::vec::Vec;

use p3_air::symbolic::SymbolicExpression;
use p3_air::{AirBuilder, RowWindow};
use p3_field::{Algebra, ExtensionField, Field, PackedValue, PrimeCharacteristicRing, dot_product};
use p3_lookup::Challenges;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::{Poly, PolyMaybePacked};
use p3_sumcheck::layout::Table;
use p3_util::DisjointMutPtr;

use super::Fraction;
use crate::lookup::LookupPlan;
use crate::selectors::BoundaryEvals;

/// Base-field context used to resolve only the symbolic expressions retained
/// by [`p3_lookup::Lookups::from_air`].
struct LookupRowEvaluator<'row, F, Var> {
    main_window: RowWindow<'row, Var>,
    preprocessed_window: RowWindow<'row, Var>,
    boundary: BoundaryEvals<Var>,
    public_values: &'row [F],
}

impl<'row, F, Var> AirBuilder for LookupRowEvaluator<'row, F, Var>
where
    F: Field,
    Var: Algebra<F> + Copy + Send + Sync,
{
    type F = F;
    type Expr = Var;
    type Var = Var;
    type PreprocessedWindow = RowWindow<'row, Var>;
    type MainWindow = RowWindow<'row, Var>;
    type PublicVar = F;
    type PeriodicVar = Var;

    fn main(&self) -> Self::MainWindow {
        self.main_window
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed_window
    }

    fn is_first_row(&self) -> Self::Expr {
        self.boundary.first
    }

    fn is_last_row(&self) -> Self::Expr {
        self.boundary.last
    }

    fn is_transition(&self) -> Self::Expr {
        self.boundary.transition
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, _x: I) {
        unreachable!("lookup row evaluator only resolves retained symbolic expressions")
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

struct Scratch<Var> {
    local: Vec<Var>,
    next: Vec<Var>,
    preprocessed_local: Vec<Var>,
    preprocessed_next: Vec<Var>,
}

impl<Var: PrimeCharacteristicRing> Scratch<Var> {
    fn new(main_width: usize, preprocessed_width: usize) -> Self {
        Self {
            local: vec![Var::ZERO; main_width],
            next: vec![Var::ZERO; main_width],
            preprocessed_local: vec![Var::ZERO; preprocessed_width],
            preprocessed_next: vec![Var::ZERO; preprocessed_width],
        }
    }
}

struct ContributionEval<'a, F, E> {
    fields: &'a [SymbolicExpression<F>],
    multiplicity: &'a SymbolicExpression<F>,
    bus_prefix: E,
    beta_powers: &'a [E],
}

impl<F: Field> LookupPlan<F> {
    /// Materializes the numerator and denominator tables consumed by fractional GKR.
    ///
    /// `main`, `preprocessed`, and `public_values` use the original AIR order
    /// supplied to [`LookupPlan::build`]. Each atomic lookup contribution owns
    /// one exact-height block in the output. Any remaining tail is padded with
    /// the neutral fraction `0 / 1`.
    ///
    /// `alpha` and `beta` are the lookup challenges used to separate buses and
    /// compress each contribution's payload.
    ///
    /// # Panics
    ///
    /// Panics if a planned AIR has no corresponding entry in one of the input
    /// slices or a retained lookup expression refers to unavailable trace,
    /// preprocessed, or public-value data.
    pub fn materialize_fraction<EF>(
        &self,
        main: &[&Table<F>],
        preprocessed: &[Option<&Table<F>>],
        public_values: &[&[F]],
        alpha: EF,
        beta: EF,
    ) -> Fraction<Poly<F>, PolyMaybePacked<F, EF>>
    where
        EF: ExtensionField<F>,
    {
        let packed_beta_powers = beta
            .powers()
            .take(self.max_width)
            .map(EF::ExtensionPacking::from)
            .collect::<Vec<_>>();
        let challenges = Challenges::new(alpha, beta, self.max_width, self.num_buses);

        let height = 1 << self.num_variables;
        let mut numerators = F::zero_vec(height);
        let mut denominators = vec![EF::ExtensionPacking::ONE; height / F::Packing::WIDTH];

        for planned in &self.instances {
            let main = main[planned.air_index];
            let preprocessed = preprocessed[planned.air_index];
            let public_values = public_values[planned.air_index];
            debug_assert_eq!(main.num_variables(), planned.num_variables);
            if let Some(preprocessed) = preprocessed {
                debug_assert_eq!(preprocessed.num_variables(), planned.num_variables);
            }
            let num_evals = 1 << planned.num_variables;
            let packed_len = num_evals / F::Packing::WIDTH;
            let beta_powers = packed_beta_powers.as_slice();
            let contributions = planned
                .lookups
                .iter()
                .zip(&planned.bus_ids)
                .flat_map(|(lookup, &bus_id)| {
                    let bus_prefix = EF::ExtensionPacking::from(challenges.bus_prefix[bus_id]);
                    lookup.elements.iter().zip(&lookup.multiplicities).map(
                        move |(fields, multiplicity)| ContributionEval {
                            fields,
                            multiplicity,
                            bus_prefix,
                            beta_powers,
                        },
                    )
                })
                .collect::<Vec<_>>();

            let numerator_len = contributions.len() * num_evals;
            let denominator_len = contributions.len() * packed_len;
            debug_assert_eq!(planned.base_offset % F::Packing::WIDTH, 0);
            let packed_start = planned.base_offset / F::Packing::WIDTH;
            materialize_contributions_packed::<F, EF>(
                main,
                preprocessed,
                public_values,
                &contributions,
                &mut numerators[planned.base_offset..planned.base_offset + numerator_len],
                &mut denominators[packed_start..packed_start + denominator_len],
            );
        }

        Fraction {
            n: Poly::new(numerators),
            d: PolyMaybePacked::Packed(Poly::new(denominators)),
        }
    }
}

/// Materialize every atomic lookup contribution of one AIR in a single row scan.
///
/// Output remains block-major: contribution `i` owns one exact-height numerator
/// block and one packed denominator block. Parallel row workers scatter into
/// disjoint positions across those blocks after loading the trace row once.
fn materialize_contributions_packed<F, EF>(
    main: &Table<F>,
    preprocessed: Option<&Table<F>>,
    public_values: &[F],
    contributions: &[ContributionEval<'_, F, EF::ExtensionPacking>],
    numerators: &mut [F],
    denominators: &mut [EF::ExtensionPacking],
) where
    F: Field,
    EF: ExtensionField<F>,
{
    let packing_width = F::Packing::WIDTH;
    let num_evals = 1 << main.num_variables();
    let packed_len = num_evals / packing_width;
    let main_width = main.num_polys();
    let preprocessed_width = preprocessed.map_or(0, Table::num_polys);
    assert_eq!(numerators.len(), contributions.len() * num_evals);
    assert_eq!(denominators.len(), contributions.len() * packed_len);

    let numerator_ptr = DisjointMutPtr::new(numerators);
    let denominator_ptr = DisjointMutPtr::new(denominators);

    (0..packed_len).into_par_iter().for_each_init(
        || Scratch::new(main_width, preprocessed_width),
        |scratch, packed_row| {
            let row = packed_row * packing_width;
            let fill_columns =
                |local: &mut [F::Packing], next: &mut [F::Packing], table: &Table<F>| {
                    local.iter_mut().zip(next).zip(table.iter_polys()).for_each(
                        |((local, next), column)| {
                            *local = *F::Packing::from_slice(&column[row..row + packing_width]);
                            *next = if row + 1 + packing_width <= num_evals {
                                *F::Packing::from_slice(&column[row + 1..row + 1 + packing_width])
                            } else {
                                F::Packing::from_fn(|lane| {
                                    column[core::cmp::min(row + lane + 1, num_evals - 1)]
                                })
                            };
                        },
                    );
                };

            fill_columns(&mut scratch.local, &mut scratch.next, main);
            if let Some(preprocessed) = preprocessed {
                fill_columns(
                    &mut scratch.preprocessed_local,
                    &mut scratch.preprocessed_next,
                    preprocessed,
                );
            }

            let evaluator = LookupRowEvaluator {
                main_window: RowWindow::from_two_rows(&scratch.local, &scratch.next),
                preprocessed_window: RowWindow::from_two_rows(
                    &scratch.preprocessed_local,
                    &scratch.preprocessed_next,
                ),
                boundary: BoundaryEvals::from_packed_row(row, num_evals),
                public_values,
            };

            for (contribution_index, contribution) in contributions.iter().enumerate() {
                let numerator_offset = contribution_index * num_evals + row;
                let denominator_offset = contribution_index * packed_len + packed_row;

                // SAFETY: `packed_row` is unique to this parallel iteration.
                // Different contributions own disjoint exact-height blocks, and
                // different rows own disjoint packed entries within each block.
                let numerator_chunk =
                    unsafe { numerator_ptr.slice_mut(numerator_offset, packing_width) };
                let denominator =
                    unsafe { &mut denominator_ptr.slice_mut(denominator_offset, 1)[0] };

                *F::Packing::from_slice_mut(numerator_chunk) =
                    contribution.multiplicity.resolve(&evaluator);
                let fingerprint: EF::ExtensionPacking = dot_product(
                    contribution.beta_powers.iter().copied(),
                    contribution
                        .fields
                        .iter()
                        .map(|field| field.resolve(&evaluator)),
                );
                *denominator = contribution.bus_prefix - fingerprint;
            }
        },
    );
}

#[cfg(test)]
pub(super) mod tests {
    extern crate std;

    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing, PrimeField};
    use p3_lookup::{Count, InteractionBuilder, InteractionSymbolicBuilder};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::log2_strict_usize;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn test_lookup_num_variables() -> usize {
        log2_strict_usize(<F as Field>::Packing::WIDTH).max(4)
    }

    pub(crate) fn materialize<'a, F, EF, A>(
        airs: &[&'a A],
        main: &[&'a Table<F>],
        preprocessed: &[Option<&'a Table<F>>],
        public_values: &[&'a [F]],
        alpha: EF,
        beta: EF,
    ) -> Option<Fraction<Poly<F>, PolyMaybePacked<F, EF>>>
    where
        F: PrimeField,
        EF: ExtensionField<F>,
        A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>>,
    {
        let num_variables = main
            .iter()
            .map(|table| table.num_variables())
            .collect::<Vec<_>>();
        let plan = LookupPlan::build::<EF, A>(airs, &num_variables)
            .expect("lookup multiplicity height bound should hold")?;
        Some(plan.materialize_fraction(main, preprocessed, public_values, alpha, beta))
    }

    #[derive(Clone, Copy)]
    enum Declaration {
        Global,
        GlobalNamed(&'static str),
        GlobalNamedWide(&'static str),
        LocalPair,
        NextGlobal,
        WideGlobal,
        SymbolicSources,
        EmptyLocal,
        ExcessiveMultiplicity,
        None,
    }

    struct TestAir(Declaration);

    impl<F: Field> BaseAir<F> for TestAir {
        fn width(&self) -> usize {
            3
        }

        fn preprocessed_width(&self) -> usize {
            usize::from(matches!(self.0, Declaration::SymbolicSources))
        }

        fn num_public_values(&self) -> usize {
            usize::from(matches!(self.0, Declaration::SymbolicSources))
        }
    }

    impl<AB> Air<AB> for TestAir
    where
        AB: AirBuilder<F: Field> + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();

            match self.0 {
                Declaration::Global => {
                    builder.push_interaction("range", [local[0]], Count::bounded(AB::Expr::ONE, 1));
                }
                Declaration::GlobalNamed(name) => {
                    builder.push_interaction(name, [local[0]], Count::bounded(AB::Expr::ONE, 1));
                }
                Declaration::GlobalNamedWide(name) => builder.push_interaction(
                    name,
                    [local[0], local[1]],
                    Count::bounded(AB::Expr::ONE, 1),
                ),
                Declaration::LocalPair => builder.push_local_interaction([
                    (vec![local[0].into()], Count::bounded(AB::Expr::ONE, 1)),
                    (vec![local[1].into()], Count::provided(-AB::Expr::ONE)),
                ]),
                Declaration::NextGlobal => {
                    let next = main.next_slice();
                    let payload = local[0] * local[1] - next[0];
                    builder.push_interaction(
                        "range",
                        [payload],
                        Count::bounded(local[2].into(), 3),
                    );
                }
                Declaration::WideGlobal => builder.push_interaction(
                    "wide",
                    [local[0], local[1]],
                    Count::bounded(AB::Expr::ONE, 1),
                ),
                Declaration::SymbolicSources => {
                    let preprocessed = builder.preprocessed();
                    let preprocessed_local: AB::Expr = preprocessed.current_slice()[0].into();
                    let preprocessed_next: AB::Expr = preprocessed.next_slice()[0].into();
                    let public: AB::Expr = builder.public_values()[0].into();
                    let payload = preprocessed_local
                        + preprocessed_next.double()
                        + public
                        + builder.is_first_row() * AB::Expr::from_u64(3)
                        + builder.is_last_row() * AB::Expr::from_u64(5)
                        + builder.is_transition() * AB::Expr::from_u64(7);
                    builder.push_interaction(
                        "sources",
                        [payload, local[0].into()],
                        Count::bounded(local[1].into(), 3),
                    );
                }
                Declaration::EmptyLocal => {
                    builder.push_local_interaction(core::iter::empty::<(
                        Vec<AB::Expr>,
                        Count<AB::Expr>,
                    )>());
                }
                Declaration::ExcessiveMultiplicity => builder.push_interaction(
                    "range",
                    [local[0]],
                    Count::bounded(AB::Expr::ONE, u32::MAX),
                ),
                Declaration::None => {}
            }
        }
    }

    struct CountedAir {
        eval_calls: AtomicUsize,
    }

    impl<F: Field> BaseAir<F> for CountedAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl<AB> Air<AB> for CountedAir
    where
        AB: AirBuilder<F: Field> + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            self.eval_calls.fetch_add(1, Ordering::Relaxed);
            let main = builder.main();
            let local = main.current_slice();
            builder.assert_eq(local[0], local[1]);
            builder.push_interaction("range", [local[0]], 1);
        }
    }

    fn table(columns: &[&[u64]]) -> Table<F> {
        let height = columns[0].len();
        Table::new(RowMajorMatrix::new(
            columns
                .iter()
                .flat_map(|column| column.iter().copied().map(F::from_u64))
                .collect(),
            height,
        ))
    }

    fn unpack_denominators(
        materialization: &Fraction<Poly<F>, PolyMaybePacked<F, EF>>,
    ) -> Poly<EF> {
        materialization.d.clone().unpack()
    }

    #[test]
    fn materializes_local_and_global_buses_with_neutral_padding() {
        let local = TestAir(Declaration::LocalPair);
        let global = TestAir(Declaration::Global);
        let height = 1 << test_lookup_num_variables();
        let local_values = vec![3; height];
        let local_provided = vec![8; height];
        let global_values = vec![5; height];
        let zeros = vec![0; height];
        let local_main = table(&[&local_values, &local_provided, &zeros]);
        let global_main = table(&[&global_values, &zeros, &zeros]);

        let materialization = materialize(
            &[&local, &global],
            &[&local_main, &global_main],
            &[None, None],
            &[&[], &[]],
            EF::from_u64(20),
            EF::from_u64(5),
        )
        .unwrap();
        let numerators = materialization.n.as_slice();
        let denominators = unpack_denominators(&materialization);
        assert!(numerators[..height].iter().all(|&value| value == F::ONE));
        assert!(
            numerators[height..2 * height]
                .iter()
                .all(|&value| value == -F::ONE)
        );
        assert!(
            numerators[2 * height..3 * height]
                .iter()
                .all(|&value| value == F::ONE)
        );
        assert!(
            numerators[3 * height..]
                .iter()
                .all(|&value| value == F::ZERO)
        );
        assert!(
            denominators.as_slice()[..height]
                .iter()
                .all(|&value| value == EF::from_u64(22))
        );
        assert!(
            denominators.as_slice()[height..2 * height]
                .iter()
                .all(|&value| value == EF::from_u64(17))
        );
        assert!(
            denominators.as_slice()[2 * height..3 * height]
                .iter()
                .all(|&value| value == EF::from_u64(25))
        );
        assert!(
            denominators.as_slice()[3 * height..]
                .iter()
                .all(|&value| value == EF::ONE)
        );
    }

    #[test]
    fn shares_named_global_buses_and_separates_different_names() {
        let shared_a = TestAir(Declaration::GlobalNamed("shared"));
        let shared_b = TestAir(Declaration::GlobalNamed("shared"));
        let other = TestAir(Declaration::GlobalNamed("other"));
        let height = 1 << test_lookup_num_variables();
        let values = vec![5; height];
        let zeros = vec![0; height];
        let main = table(&[&values, &zeros, &zeros]);

        let materialization = materialize(
            &[&shared_a, &shared_b, &other],
            &[&main, &main, &main],
            &[None, None, None],
            &[&[], &[], &[]],
            EF::from_u64(100),
            EF::from_u64(7),
        )
        .unwrap();
        let active_height = 3 * height;
        let denominators = unpack_denominators(&materialization);
        assert!(
            materialization.n.as_slice()[..active_height]
                .iter()
                .all(|&value| value == F::ONE)
        );
        assert!(
            materialization.n.as_slice()[active_height..]
                .iter()
                .all(|&value| value == F::ZERO)
        );
        assert!(
            denominators.as_slice()[..2 * height]
                .iter()
                .all(|&value| value == EF::from_u64(102))
        );
        assert!(
            denominators.as_slice()[2 * height..active_height]
                .iter()
                .all(|&value| value == EF::from_u64(109))
        );
        assert!(
            denominators.as_slice()[active_height..]
                .iter()
                .all(|&value| value == EF::ONE)
        );
    }

    #[test]
    #[should_panic(
        expected = "named global lookup bus `shared` uses payload width 2, but its established width is 1"
    )]
    fn rejects_different_payload_widths_on_one_named_global_bus() {
        let narrow = TestAir(Declaration::GlobalNamed("shared"));
        let wide = TestAir(Declaration::GlobalNamedWide("shared"));
        let num_variables = test_lookup_num_variables();

        LookupPlan::<F>::build::<EF, _>(&[&narrow, &wide], &[num_variables, num_variables])
            .unwrap();
    }

    #[test]
    fn rejects_a_multiplicity_height_bound_reaching_the_characteristic() {
        let air = TestAir(Declaration::ExcessiveMultiplicity);
        let result = LookupPlan::<F>::build::<EF, _>(&[&air], &[test_lookup_num_variables()]);

        assert!(matches!(
            result,
            Err(p3_lookup::LookupError::MultiplicityHeightBoundExceeded { .. })
        ));
    }

    #[test]
    fn places_mixed_height_instances_in_descending_stable_order() {
        let air = TestAir(Declaration::Global);
        let short_height = 1 << test_lookup_num_variables();
        let middle_height = 2 * short_height;
        let tall_height = 4 * short_height;
        let short_values = (0..short_height)
            .map(|row| 40 + row as u64)
            .collect::<Vec<_>>();
        let middle_values = (0..middle_height)
            .map(|row| 100 + row as u64)
            .collect::<Vec<_>>();
        let tall_a_values = (0..tall_height)
            .map(|row| 200 + row as u64)
            .collect::<Vec<_>>();
        let tall_b_values = (0..tall_height)
            .map(|row| 300 + row as u64)
            .collect::<Vec<_>>();
        let short_zeros = vec![0; short_height];
        let middle_zeros = vec![0; middle_height];
        let tall_zeros = vec![0; tall_height];
        let short = table(&[&short_values, &short_zeros, &short_zeros]);
        let middle = table(&[&middle_values, &middle_zeros, &middle_zeros]);
        let tall_a = table(&[&tall_a_values, &tall_zeros, &tall_zeros]);
        let tall_b = table(&[&tall_b_values, &tall_zeros, &tall_zeros]);
        let alpha = EF::from_u64(100);
        let beta = EF::from_u64(7);

        let materialization = materialize(
            &[&air, &air, &air, &air],
            &[&short, &tall_a, &middle, &tall_b],
            &[None, None, None, None],
            &[&[], &[], &[], &[]],
            alpha,
            beta,
        )
        .unwrap();

        let active_values = tall_a_values
            .iter()
            .chain(&tall_b_values)
            .chain(&middle_values)
            .chain(&short_values)
            .copied()
            .collect::<Vec<_>>();
        let bus_prefix = Challenges::new(alpha, beta, 1, 1).bus_prefix[0];
        let denominators = unpack_denominators(&materialization);
        assert_eq!(
            materialization.n.num_evals(),
            active_values.len().next_power_of_two()
        );
        assert!(
            materialization.n.as_slice()[..active_values.len()]
                .iter()
                .all(|&numer| numer == F::ONE)
        );
        assert!(
            materialization.n.as_slice()[active_values.len()..]
                .iter()
                .all(|&numer| numer == F::ZERO)
        );
        for (row, value) in active_values.into_iter().enumerate() {
            assert_eq!(
                denominators.as_slice()[row],
                bus_prefix - EF::from_u64(value)
            );
        }
        assert!(
            denominators.as_slice()[tall_height * 2 + middle_height + short_height..]
                .iter()
                .all(|&denom| denom == EF::ONE)
        );
    }

    #[test]
    fn resolves_count_expression_and_repeat_last_next() {
        let air = TestAir(Declaration::NextGlobal);
        let height = 1 << test_lookup_num_variables();
        let left = (0..height).map(|row| row as u64 + 2).collect::<Vec<_>>();
        let right = (0..height).map(|row| row as u64 + 10).collect::<Vec<_>>();
        let multiplicity = (0..height)
            .map(|row| [1, 2, 0, 3][row % 4])
            .collect::<Vec<_>>();
        let main = table(&[&left, &right, &multiplicity]);

        let materialization = materialize(
            &[&air],
            &[&main],
            &[None],
            &[&[]],
            EF::from_u64(100),
            EF::from_u64(7),
        )
        .unwrap();
        let denominators = unpack_denominators(&materialization);
        let bus_prefix = Challenges::new(EF::from_u64(100), EF::from_u64(7), 1, 1).bus_prefix[0];
        for row in 0..height {
            let next_row = core::cmp::min(row + 1, height - 1);
            let payload =
                F::from_u64(left[row]) * F::from_u64(right[row]) - F::from_u64(left[next_row]);
            assert_eq!(
                materialization.n.as_slice()[row],
                F::from_u64(multiplicity[row])
            );
            assert_eq!(denominators.as_slice()[row], bus_prefix - EF::from(payload));
        }
    }

    #[test]
    fn combines_payload_in_extension_field_with_precomputed_powers() {
        let air = TestAir(Declaration::WideGlobal);
        let height = 1 << test_lookup_num_variables();
        let first = (0..height).map(|row| row as u64 + 2).collect::<Vec<_>>();
        let second = (0..height).map(|row| row as u64 + 7).collect::<Vec<_>>();
        let zeros = vec![0; height];
        let main = table(&[&first, &second, &zeros]);

        let materialization = materialize(
            &[&air],
            &[&main],
            &[None],
            &[&[]],
            EF::from_u64(100),
            EF::from_u64(5),
        )
        .unwrap();
        let denominators = unpack_denominators(&materialization);
        let alpha = EF::from_u64(100);
        let beta = EF::from_u64(5);
        let bus_prefix = Challenges::new(alpha, beta, 2, 1).bus_prefix[0];
        for row in 0..height {
            let fingerprint = EF::from_u64(first[row]) + beta * EF::from_u64(second[row]);
            assert_eq!(materialization.n.as_slice()[row], F::ONE);
            assert_eq!(denominators.as_slice()[row], bus_prefix - fingerprint);
        }
    }

    #[test]
    fn packed_materialization_matches_the_row_formula() {
        let air = TestAir(Declaration::NextGlobal);
        let height = core::cmp::max(
            1 << test_lookup_num_variables(),
            2 * <<F as Field>::Packing as PackedValue>::WIDTH,
        );
        let left = (0..height).map(|i| i as u64 + 2).collect::<Vec<_>>();
        let right = (0..height).map(|i| (i % 5) as u64 + 3).collect::<Vec<_>>();
        let multiplicity = (0..height).map(|i| (i % 4) as u64).collect::<Vec<_>>();
        let main = table(&[&left, &right, &multiplicity]);
        let alpha = EF::from_u64(1_000);
        let beta = EF::from_u64(7);

        let materialization = materialize(&[&air], &[&main], &[None], &[&[]], alpha, beta).unwrap();
        let bus_prefix = Challenges::new(alpha, beta, 1, 1).bus_prefix[0];
        let denominators = unpack_denominators(&materialization);

        for row in 0..height {
            let next_row = core::cmp::min(row + 1, height - 1);
            let payload =
                F::from_u64(left[row]) * F::from_u64(right[row]) - F::from_u64(left[next_row]);
            assert_eq!(
                materialization.n.as_slice()[row],
                F::from_u64(multiplicity[row])
            );
            assert_eq!(denominators.as_slice()[row], bus_prefix - EF::from(payload));
        }
    }

    #[test]
    fn resolves_all_symbolic_sources_with_extension_challenges() {
        let air = TestAir(Declaration::SymbolicSources);
        let mut rng = SmallRng::seed_from_u64(0xA11_50CE5);

        for height in [
            1 << test_lookup_num_variables(),
            core::cmp::max(
                2 << test_lookup_num_variables(),
                2 * <<F as Field>::Packing as PackedValue>::WIDTH,
            ),
        ] {
            let key_tail = (0..height).map(|i| i as u64 + 2).collect::<Vec<_>>();
            let multiplicity = (0..height).map(|i| (i % 3) as u64 + 1).collect::<Vec<_>>();
            let unused = vec![0; height];
            let preprocessed_values = (0..height).map(|i| 10 + (i % 7) as u64).collect::<Vec<_>>();
            let main = table(&[&key_tail, &multiplicity, &unused]);
            let preprocessed = table(&[&preprocessed_values]);
            let public_values = [F::from_u64(6)];
            let alpha: EF = rng.random();
            let beta: EF = rng.random();

            let materialization = materialize(
                &[&air],
                &[&main],
                &[Some(&preprocessed)],
                &[&public_values],
                alpha,
                beta,
            )
            .unwrap();
            let bus_prefix = Challenges::new(alpha, beta, 2, 1).bus_prefix[0];
            let denominators = unpack_denominators(&materialization);

            for row in 0..height {
                let next_row = core::cmp::min(row + 1, height - 1);
                let payload = F::from_u64(preprocessed_values[row])
                    + F::from_u64(preprocessed_values[next_row]).double()
                    + public_values[0]
                    + F::from_u64(3) * F::from_bool(row == 0)
                    + F::from_u64(5) * F::from_bool(row + 1 == height)
                    + F::from_u64(7) * F::from_bool(row + 1 < height);
                let fingerprint = EF::from(payload) + beta * F::from_u64(key_tail[row]);

                assert_eq!(
                    materialization.n.as_slice()[row],
                    F::from_u64(multiplicity[row])
                );
                assert_eq!(denominators.as_slice()[row], bus_prefix - fingerprint);
            }
        }
    }

    #[test]
    fn evaluates_air_symbolically_only_once() {
        let air = CountedAir {
            eval_calls: AtomicUsize::new(0),
        };
        let height = 1 << test_lookup_num_variables();
        let values = (0..height).map(|row| row as u64 + 2).collect::<Vec<_>>();
        let main = table(&[&values, &values]);

        let materialization = materialize(
            &[&air],
            &[&main],
            &[None],
            &[&[]],
            EF::from_u64(100),
            EF::from_u64(7),
        );

        assert!(materialization.is_some());
        assert_eq!(air.eval_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn interaction_free_batch_materializes_no_rows() {
        let no_declaration = TestAir(Declaration::None);
        let empty_local = TestAir(Declaration::EmptyLocal);
        let main = table(&[&[0, 0], &[0, 0], &[0, 0]]);

        let no_declaration = materialize(
            &[&no_declaration],
            &[&main],
            &[None],
            &[&[]],
            F::ZERO,
            F::ZERO,
        );
        let empty_local = materialize(&[&empty_local], &[&main], &[None], &[&[]], F::ZERO, F::ZERO);

        assert!(no_declaration.is_none());
        assert!(empty_local.is_none());
    }
}
