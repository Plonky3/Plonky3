//! A finished circuit: its tables and the traces of one run.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_batch_stark::{StarkGenericConfig, StarkInstance, Val};
use p3_field::{ExtensionField, Field, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use crate::builder::{Kind, Op};
use crate::tables::{
    AluAir, CircuitAir, PermuteAir, PublicAir, RecomposeAir, ValueAir, padded_matrix,
};
use crate::{CircuitBuilder, PermutationChip, Wire};

/// A circuit, as the tables that prove it, with the traces of the run that built it.
///
/// The preprocessed columns depend only on the circuit, so they fix its verifying key.
///
/// The main traces hold this run's witness.
#[derive(Debug)]
pub struct Circuit<F: Field, EF, P: PermutationChip<F>> {
    /// The tables, in proving order.
    pub airs: Vec<CircuitAir<F, EF, P>>,
    /// The main trace of every table.
    pub traces: Vec<RowMajorMatrix<F>>,
    /// The public inputs, in index order.
    pub publics: Vec<F>,
}

/// Rows of every table, as preprocessed and main values.
struct Rows<F> {
    preprocessed: Vec<F>,
    main: Vec<F>,
}

impl<F> Default for Rows<F> {
    fn default() -> Self {
        Self {
            preprocessed: Vec::new(),
            main: Vec::new(),
        }
    }
}

impl<F, EF, P> Circuit<F, EF, P>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
{
    /// The tables and traces of a finished builder.
    pub(crate) fn new(builder: CircuitBuilder<F, EF, P>) -> Self {
        let d = EF::DIMENSION;
        let reads = read_counts(&builder.ops, builder.values.len());
        let id = |wire: Wire| F::from_u32(wire.0);
        let writes = |wire: Wire| F::from_u32(reads[wire.index()]);
        let limbs = |wire: Wire| {
            builder.values[wire.index()]
                .as_basis_coefficients_slice()
                .to_vec()
        };

        let mut value = Rows::default();
        let mut base_alu = Rows::default();
        let mut ext_alu = Rows::default();
        let mut compose = Rows::default();
        let mut decompose = Rows::default();
        let mut permute_pre = Vec::new();
        let mut permute_inputs = Vec::new();
        let mut public_pre = vec![F::ZERO; 2 * builder.publics.len()];

        for op in &builder.ops {
            match op {
                Op::Const { out } | Op::Witness { out } => {
                    let fixed = matches!(op, Op::Const { .. });
                    let is_base = builder.kinds[out.index()] == Kind::Base;
                    value.preprocessed.extend([
                        id(*out),
                        writes(*out),
                        F::from_bool(is_base),
                        F::from_bool(fixed),
                    ]);
                    let values = limbs(*out);
                    value.preprocessed.extend(if fixed {
                        values.clone()
                    } else {
                        vec![F::ZERO; d]
                    });
                    value.main.extend(values);
                }
                Op::Public { out, index } => {
                    public_pre[2 * index] = id(*out);
                    public_pre[2 * index + 1] = writes(*out);
                }
                Op::Alu {
                    ext,
                    a,
                    b,
                    c,
                    out,
                    out_read,
                } => {
                    let rows = if *ext { &mut ext_alu } else { &mut base_alu };
                    let out_writes = if *out_read { F::ZERO } else { writes(*out) };
                    rows.preprocessed.extend([
                        F::ONE,
                        id(*a),
                        id(*b),
                        id(*c),
                        id(*out),
                        out_writes,
                        F::from_bool(*out_read),
                    ]);
                    for wire in [a, b, c, out] {
                        let values = limbs(*wire);
                        if *ext {
                            rows.main.extend(values);
                        } else {
                            rows.main.push(values[0]);
                        }
                    }
                }
                Op::Recompose {
                    ext,
                    limbs: limb_wires,
                    compose: composing,
                } => {
                    let rows = if *composing {
                        &mut compose
                    } else {
                        &mut decompose
                    };
                    rows.preprocessed.extend([F::ONE, id(*ext)]);
                    if *composing {
                        rows.preprocessed.push(writes(*ext));
                        rows.preprocessed.extend(limb_wires.iter().map(|&w| id(w)));
                    } else {
                        rows.preprocessed.extend(limb_wires.iter().map(|&w| id(w)));
                        rows.preprocessed
                            .extend(limb_wires.iter().map(|&w| writes(w)));
                    }
                    rows.main.extend(limbs(*ext));
                }
                Op::Permute { inputs, outputs } => {
                    permute_pre.push(F::ONE);
                    permute_pre.extend(inputs.iter().map(|&w| id(w)));
                    for output in outputs {
                        permute_pre.push(output.map_or(F::ZERO, id));
                    }
                    for output in outputs {
                        permute_pre.push(output.map_or(F::ZERO, writes));
                    }
                    permute_inputs.push(inputs.iter().map(|&w| limbs(w)[0]).collect());
                }
            }
        }

        let mut airs = Vec::new();
        let mut traces = Vec::new();
        let mut push = |air: CircuitAir<F, EF, P>, trace: RowMajorMatrix<F>| {
            airs.push(air);
            traces.push(trace);
        };

        let value_width = 4 + d;
        push(
            CircuitAir::Value(ValueAir {
                dimension: d,
                preprocessed: padded_matrix(value.preprocessed, value_width),
            }),
            padded_matrix(value.main, d),
        );
        if !builder.publics.is_empty() {
            let count = builder.publics.len();
            push(
                CircuitAir::Public(PublicAir {
                    dimension: d,
                    count,
                    preprocessed: RowMajorMatrix::new(public_pre, 2 * count),
                }),
                RowMajorMatrix::new(builder.publics.clone(), count),
            );
        }
        for (rows, ext) in [(base_alu, false), (ext_alu, true)] {
            if rows.main.is_empty() {
                continue;
            }
            let air = AluAir {
                ext,
                dimension: d,
                preprocessed: padded_matrix(rows.preprocessed, 7),
                _ef: PhantomData,
            };
            let width = 4 * air.limbs();
            let trace = padded_matrix(rows.main, width);
            push(
                if ext {
                    CircuitAir::ExtAlu(air)
                } else {
                    CircuitAir::BaseAlu(air)
                },
                trace,
            );
        }
        for (rows, composing) in [(compose, true), (decompose, false)] {
            if rows.main.is_empty() {
                continue;
            }
            let width = if composing { 3 + d } else { 2 + 2 * d };
            push(
                CircuitAir::Recompose(RecomposeAir {
                    compose: composing,
                    dimension: d,
                    preprocessed: padded_matrix(rows.preprocessed, width),
                }),
                padded_matrix(rows.main, d),
            );
        }
        if !permute_inputs.is_empty() {
            let trace = builder.chip.trace(permute_inputs);
            let width = 1 + 3 * P::WIDTH;
            let mut preprocessed = permute_pre;
            preprocessed.resize(trace.values.len() / trace.width * width, F::ZERO);
            push(
                CircuitAir::Permute(PermuteAir {
                    chip: builder.chip.clone(),
                    dimension: d,
                    preprocessed: RowMajorMatrix::new(preprocessed, width),
                }),
                trace,
            );
        }

        Self {
            airs,
            traces,
            publics: builder.publics,
        }
    }

    /// The public values of every table, in table order.
    pub fn public_values(&self) -> Vec<Vec<F>> {
        self.airs
            .iter()
            .map(|air| match air {
                CircuitAir::Public(_) => self.publics.clone(),
                _ => Vec::new(),
            })
            .collect()
    }

    /// One batch-STARK instance per table.
    pub fn instances<SC>(&self) -> Vec<StarkInstance<'_, SC, CircuitAir<F, EF, P>>>
    where
        SC: StarkGenericConfig<Challenge = EF>,
        SC::Pcs: p3_commit::Pcs<EF, SC::Challenger, Domain: p3_commit::PolynomialSpace<Val = F>>,
        Val<SC>: Field,
    {
        let publics = self.public_values();
        self.airs
            .iter()
            .zip(&self.traces)
            .zip(publics)
            .map(|((air, trace), public_values)| StarkInstance {
                air,
                trace,
                public_values,
            })
            .collect()
    }
}

/// How many times every wire is read.
fn read_counts(ops: &[Op], wires: usize) -> Vec<u32> {
    let mut reads = vec![0u32; wires];
    let mut read = |wire: &Wire| reads[wire.index()] += 1;
    for op in ops {
        match op {
            Op::Const { .. } | Op::Public { .. } | Op::Witness { .. } => {}
            Op::Alu {
                a,
                b,
                c,
                out,
                out_read,
                ..
            } => {
                [a, b, c].into_iter().for_each(&mut read);
                if *out_read {
                    read(out);
                }
            }
            Op::Recompose {
                ext,
                limbs,
                compose,
            } => {
                if *compose {
                    limbs.iter().for_each(&mut read);
                } else {
                    read(ext);
                }
            }
            Op::Permute { inputs, .. } => inputs.iter().for_each(&mut read),
        }
    }
    reads
}
