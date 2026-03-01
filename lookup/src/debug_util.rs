//! Debug helpers to inspect lookup witnesses.
//!
//! They recompute every lookup tuple from the raw traces and assert that the
//! resulting multiset is balanced (total multiplicity 0). Any mismatch will
//! be reported with its location.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::{format, vec};

use hashbrown::HashMap;
use p3_air::{AirBuilder, PermutationAirBuilder, RowWindow};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::VerticalPair;

use crate::lookup_traits::{Kind, Lookup, symbolic_to_expr};

/// All inputs required to replay lookup evaluations for one AIR instance.
pub struct LookupDebugInstance<'a, F: Field> {
    /// Main execution trace for the AIR.
    pub main_trace: &'a RowMajorMatrix<F>,
    /// Optional preprocessed columns associated with the AIR.
    pub preprocessed_trace: &'a Option<RowMajorMatrix<F>>,
    /// Public inputs provided to the AIR.
    pub public_values: &'a [F],
    /// Lookup contexts registered by the AIR.
    pub lookups: &'a [Lookup<F>],
    /// Challenges used for the lookup permutation argument.
    pub permutation_challenges: &'a [F],
}

/// Location information used in debug messages.
#[allow(unused)] // Only used in panic message upon mismatch.
#[derive(Clone, Debug)]
struct Location {
    instance: usize,
    lookup: usize,
    row: usize,
}

/// Accumulates tuples and their multiplicities, tracking where each was seen.
#[derive(Default)]
struct MultiSet<F: Field> {
    /// Key: field-element tuple. Value: (net multiplicity, source locations).
    entries: HashMap<Vec<F>, (F, Vec<Location>)>,
}

impl<F: Field> MultiSet<F> {
    /// Record one occurrence of a tuple with the given multiplicity.
    /// Zero-multiplicity entries are silently skipped.
    fn add(&mut self, key: Vec<F>, multiplicity: F, location: Location) {
        if multiplicity.is_zero() {
            return;
        }

        self.entries
            .entry(key)
            .and_modify(|(total, locations)| {
                *total += multiplicity;
                locations.push(location.clone());
            })
            .or_insert_with(|| (multiplicity, vec![location]));
    }

    /// Panic if any tuple has a non-zero net multiplicity.
    ///
    /// Entries are sorted lexicographically before checking so that the
    /// *first* reported mismatch is deterministic (hash-map iteration
    /// order is not).
    fn assert_empty(&self, label: &str) {
        let mut entries: Vec<_> = self.entries.iter().collect();
        entries.sort_by(|(a, _), (b, _)| {
            let a_str: Vec<String> = a.iter().map(|v| v.to_string()).collect();
            let b_str: Vec<String> = b.iter().map(|v| v.to_string()).collect();
            a_str.cmp(&b_str)
        });
        for (key, (total, locations)) in entries {
            if !total.is_zero() {
                let rendered_key: Vec<String> = key.iter().map(|v| v.to_string()).collect();
                panic!(
                    "Lookup mismatch ({label}): tuple {:?} has net multiplicity {:?}. Locations: {:?}",
                    rendered_key, total, locations
                );
            }
        }
    }
}

/// Recompute all lookup tuples/multiplicities from the traces and assert that
/// every lookup represents a balanced multiset equality.
///
/// - Local lookups are checked independently per instance.
/// - Global lookups are grouped by interaction name; every tuple's total
///   multiplicity across all participants must be zero.
pub fn check_lookups<F: Field>(instances: &[LookupDebugInstance<'_, F>]) {
    // 1) Check each local lookup independently.
    for (instance_idx, instance) in instances.iter().enumerate() {
        for (lookup_idx, lookup) in instance.lookups.iter().enumerate() {
            if matches!(lookup.kind, Kind::Local) {
                let mut multiset = MultiSet::default();
                accumulate_lookup(instance_idx, lookup_idx, instance, lookup, &mut multiset);
                multiset.assert_empty(&format!(
                    "instance {instance_idx} local lookup {lookup_idx}"
                ));
            }
        }
    }

    // 2) Aggregate all global lookups that share the same interaction name,
    //    then verify each group sums to zero.
    //    A name-to-index map gives O(1) group lookups instead of a linear scan.
    let mut global_sets: Vec<(String, MultiSet<F>)> = Vec::new();
    let mut global_index: HashMap<String, usize> = HashMap::new();

    for (instance_idx, instance) in instances.iter().enumerate() {
        for (lookup_idx, lookup) in instance.lookups.iter().enumerate() {
            if let Kind::Global(name) = &lookup.kind {
                let idx = *global_index.entry(name.clone()).or_insert_with(|| {
                    global_sets.push((name.clone(), MultiSet::default()));
                    global_sets.len() - 1
                });

                accumulate_lookup(
                    instance_idx,
                    lookup_idx,
                    instance,
                    lookup,
                    &mut global_sets[idx].1,
                );
            }
        }
    }

    for (name, multiset) in global_sets {
        multiset.assert_empty(&format!("global lookup '{name}'"));
    }
}

fn accumulate_lookup<F: Field>(
    instance_idx: usize,
    lookup_idx: usize,
    instance: &LookupDebugInstance<'_, F>,
    lookup: &Lookup<F>,
    multiset: &mut MultiSet<F>,
) {
    let height = instance.main_trace.height();

    for row in 0..height {
        let local_main = instance.main_trace.row_slice(row).unwrap();
        let next_main = instance.main_trace.row_slice((row + 1) % height).unwrap();
        let main_rows = VerticalPair::new(
            RowMajorMatrixView::new_row(&*local_main),
            RowMajorMatrixView::new_row(&*next_main),
        );

        let preprocessed_rows_data = instance.preprocessed_trace.as_ref().map(|prep| {
            (
                prep.row_slice(row).unwrap(),
                prep.row_slice((row + 1) % height).unwrap(),
            )
        });
        let preprocessed_rows = preprocessed_rows_data
            .as_ref()
            .map(|(prep_local, prep_next)| {
                VerticalPair::new(
                    RowMajorMatrixView::new_row(&**prep_local),
                    RowMajorMatrixView::new_row(&**prep_next),
                )
            });

        let builder = MiniLookupBuilder {
            main: main_rows,
            preprocessed: preprocessed_rows,
            public_values: instance.public_values,
            permutation_challenges: instance.permutation_challenges,
            row,
            height,
        };

        for (tuple_idx, elements) in lookup.element_exprs.iter().enumerate() {
            let key = elements
                .iter()
                .map(|expr| symbolic_to_expr(&builder, expr))
                .collect::<Vec<_>>();

            let multiplicity = symbolic_to_expr(&builder, &lookup.multiplicities_exprs[tuple_idx]);

            multiset.add(
                key,
                multiplicity,
                Location {
                    instance: instance_idx,
                    lookup: lookup_idx,
                    row,
                },
            );
        }
    }
}

struct MiniLookupBuilder<'a, F: Field> {
    main: VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>,
    preprocessed: Option<VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>>,
    public_values: &'a [F],
    permutation_challenges: &'a [F],
    row: usize,
    height: usize,
}

impl<'a, F: Field> AirBuilder for MiniLookupBuilder<'a, F> {
    type F = F;
    type Expr = F;
    type Var = F;
    type PublicVar = F;
    type M = RowWindow<'a, F>;

    fn main(&self) -> Self::M {
        RowWindow::new(self.main.top.values, self.main.bottom.values)
    }

    fn preprocessed(&self) -> Option<Self::M> {
        self.preprocessed
            .map(|p| RowWindow::new(p.top.values, p.bottom.values))
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }

    fn is_first_row(&self) -> Self::Expr {
        F::from_bool(self.row == 0)
    }

    fn is_last_row(&self) -> Self::Expr {
        F::from_bool(self.row + 1 == self.height)
    }

    fn is_transition(&self) -> Self::Expr {
        F::from_bool(self.row + 1 < self.height)
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, _x: I) {}
}

impl<'a, F: Field> p3_air::ExtensionBuilder for MiniLookupBuilder<'a, F> {
    type EF = F;
    type ExprEF = F;
    type VarEF = F;

    fn assert_zero_ext<I: Into<Self::ExprEF>>(&mut self, _x: I) {}
}

impl<'a, F: Field> PermutationAirBuilder for MiniLookupBuilder<'a, F> {
    type MP = RowWindow<'a, F>;
    type RandomVar = F;

    fn permutation(&self) -> Self::MP {
        // Empty slices; permutation columns are not needed for debug evals.
        RowWindow::new(&[], &[])
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.permutation_challenges
    }
}
