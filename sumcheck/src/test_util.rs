//! Random table/spec fixtures.

use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::BabyBear;
use p3_field::{Field, PackedValue};
use p3_multilinear_util::poly::Poly;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::layout::Table;
use crate::{PointSchedule, TableShape, TableSpec};

pub type F = BabyBear;

pub fn table_point_schedule(width: usize, extra_points: PointSchedule) -> PointSchedule {
    let mut point_schedule = Vec::with_capacity(extra_points.len() + 1);
    point_schedule.push((0..width).collect());
    point_schedule.extend(extra_points);
    point_schedule
}

pub fn random_point_schedule(rng: &mut SmallRng, width: usize, num_points: usize) -> PointSchedule {
    table_point_schedule(
        width,
        (1..num_points)
            .map(|_| {
                let polys = (0..width)
                    .filter(|_| rng.random_bool(0.5))
                    .collect::<Vec<_>>();
                if polys.is_empty() { vec![0] } else { polys }
            })
            .collect(),
    )
}

pub fn random_table_specs(rng: &mut SmallRng, folding: usize) -> Vec<TableSpec> {
    let packing_log = log2_strict_usize(<F as Field>::Packing::WIDTH);
    loop {
        let num_points = rng.random_range(1..=5);
        let num_tables = rng.random_range(1..=5);
        let specs = (0..num_tables)
            .map(|_| {
                let num_variables = rng.random_range(1..=12);
                let width = rng.random_range(1..=5);
                TableSpec::new(
                    TableShape::new(num_variables, width),
                    random_point_schedule(rng, width, num_points),
                )
            })
            .collect::<Vec<_>>();

        if stacked_num_variables(&specs, folding) >= folding + packing_log {
            return specs;
        }
    }
}

pub fn stacked_num_variables(specs: &[TableSpec], folding: usize) -> usize {
    let total_evals = specs
        .iter()
        .map(|spec| spec.shape().width() << spec.shape().num_variables().max(folding))
        .sum::<usize>();
    log2_ceil_usize(total_evals)
}

pub fn table_specs_to_tables(specs: &[TableSpec]) -> Vec<Table<F>> {
    let mut rng = SmallRng::seed_from_u64(3);
    specs
        .iter()
        .map(|spec| {
            let polys = (0..spec.shape().width())
                .map(|_| Poly::<F>::rand(&mut rng, spec.shape().num_variables()))
                .collect();
            Table::new(polys)
        })
        .collect()
}
