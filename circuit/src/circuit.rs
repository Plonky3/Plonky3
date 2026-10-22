//! A finished circuit: its tables and the traces of one run.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_batch_stark::{StarkGenericConfig, StarkInstance, Val};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::config::{Commitment, MultiStarkConfig, PcsProverError, ProverData};
use p3_multi_stark::folder::ProverAir;
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, ProvingError, ProvingKey, prove,
};
use p3_sumcheck::PrescribedPointPcs;
use p3_sumcheck::layout::Table;
use p3_util::log2_strict_usize;

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
        let mut permute_bits = Vec::new();
        let mut merkle_digest = None;
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
                    permute_pre.extend([F::ONE, F::ZERO, F::ZERO, F::ZERO]);
                    permute_pre.extend(inputs.iter().map(|&w| id(w)));
                    for output in outputs {
                        permute_pre.push(output.map_or(F::ZERO, id));
                    }
                    for output in outputs {
                        permute_pre.push(output.map_or(F::ZERO, writes));
                    }
                    permute_inputs.push(inputs.iter().map(|&w| limbs(w)[0]).collect());
                    permute_bits.push((F::ZERO, F::ZERO));
                }
                Op::PermuteIf {
                    inputs,
                    take,
                    outputs,
                } => {
                    let width = P::WIDTH;
                    let mut row = vec![F::ZERO; 4 + 3 * width];
                    row[..4].copy_from_slice(&[F::ONE, F::ZERO, F::ONE, id(*take)]);
                    for (i, &wire) in inputs.iter().enumerate() {
                        row[4 + i] = id(wire);
                    }
                    for (i, &wire) in outputs.iter().enumerate() {
                        row[4 + width + i] = id(wire);
                        row[4 + 2 * width + i] = writes(wire);
                    }
                    permute_pre.extend(row);
                    permute_inputs.push(inputs.iter().map(|&w| limbs(w)[0]).collect());
                    let take = limbs(*take)[0];
                    permute_bits.push((take, F::ONE - take));
                }
                Op::MerkleStep {
                    digest,
                    sibling,
                    bit,
                    outputs,
                } => {
                    let size = digest.len();
                    assert!(
                        merkle_digest
                            .replace(size)
                            .is_none_or(|previous| previous == size),
                        "every Merkle step of a circuit has one digest size"
                    );
                    let width = P::WIDTH;
                    let mut row = vec![F::ZERO; 4 + 3 * width];
                    row[..4].copy_from_slice(&[F::ONE, F::ONE, F::ZERO, id(*bit)]);
                    for (i, &wire) in digest.iter().enumerate() {
                        row[4 + i] = id(wire);
                    }
                    for (i, &wire) in outputs.iter().enumerate() {
                        row[4 + width + i] = id(wire);
                        row[4 + 2 * width + i] = writes(wire);
                    }
                    permute_pre.extend(row);

                    // The state is `left || right`, padded with zeros.
                    let digest: Vec<F> = digest.iter().map(|&w| limbs(w)[0]).collect();
                    let right = limbs(*bit)[0].is_one();
                    let (left, right_child) = if right {
                        (&sibling[..], &digest[..])
                    } else {
                        (&digest[..], &sibling[..])
                    };
                    let mut state = vec![F::ZERO; width];
                    state[..size].copy_from_slice(left);
                    state[size..2 * size].copy_from_slice(right_child);
                    permute_inputs.push(state);
                    permute_bits.push((F::from_bool(right), F::ZERO));
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
            // The chip's trace, with the bit and the keep flag of every row appended.
            let chip_trace = builder.chip.trace(permute_inputs);
            let height = chip_trace.values.len() / chip_trace.width;
            permute_bits.resize(height, (F::ZERO, F::ZERO));
            let main: Vec<F> = chip_trace
                .values
                .chunks(chip_trace.width)
                .zip(&permute_bits)
                .flat_map(|(row, &(bit, keep))| row.iter().copied().chain([bit, keep]))
                .collect();
            let width = 4 + 3 * P::WIDTH;
            let mut preprocessed = permute_pre;
            preprocessed.resize(height * width, F::ZERO);
            push(
                CircuitAir::Permute(PermuteAir {
                    chip: builder.chip.clone(),
                    dimension: d,
                    digest: merkle_digest.unwrap_or(0),
                    preprocessed: RowMajorMatrix::new(preprocessed, width),
                }),
                RowMajorMatrix::new(main, chip_trace.width + 2),
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

impl<F, EF, P> Circuit<F, EF, P>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
{
    /// Pad every table with inert rows to at least `2^log_height` rows.
    ///
    /// An all-zero row reads and writes nothing, and a permutation row permutes the zero state.
    pub fn pad(&mut self, log_height: usize) {
        let height = 1 << log_height;
        for (air, trace) in self.airs.iter_mut().zip(&mut self.traces) {
            let rows = trace.values.len() / trace.width;
            if rows >= height {
                continue;
            }
            let preprocessed = match air {
                CircuitAir::Value(air) => &mut air.preprocessed,
                CircuitAir::Public(air) => &mut air.preprocessed,
                CircuitAir::BaseAlu(air) | CircuitAir::ExtAlu(air) => &mut air.preprocessed,
                CircuitAir::Recompose(air) => &mut air.preprocessed,
                CircuitAir::Permute(air) => {
                    let padding = air.chip.trace(vec![vec![F::ZERO; P::WIDTH]; height - rows]);
                    for row in padding.values.chunks(padding.width).take(height - rows) {
                        trace.values.extend_from_slice(row);
                        trace.values.extend([F::ZERO; 2]);
                    }
                    &mut air.preprocessed
                }
            };
            trace.values.resize(height * trace.width, F::ZERO);
            preprocessed
                .values
                .resize(height * preprocessed.width, F::ZERO);
        }
    }

    /// Base-two logarithm of every table's height.
    pub fn log_heights(&self) -> Vec<usize> {
        self.traces
            .iter()
            .map(|trace| log2_strict_usize(trace.values.len() / trace.width))
            .collect()
    }

    /// A proof of the circuit as one multi-STARK batch, under the key `setup` gave for its tables.
    ///
    /// # Errors
    ///
    /// When the commitment scheme rejects a table or an opening.
    ///
    /// # Panics
    ///
    /// When a table is shorter than the configuration's minimum, see [`Self::pad`].
    pub fn prove_multi_stark<C>(
        &self,
        config: &C,
        key: &ProvingKey<C>,
        challenger: &mut C::Challenger,
    ) -> Result<MultiStarkProof<C>, ProvingError<PcsProverError<C>>>
    where
        C: MultiStarkConfig<Val = F, Challenge = EF>,
        C::Pcs: PrescribedPointPcs<EF, C::Challenger>,
        C::Challenger: Clone
            + FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<Commitment<C>>,
        Commitment<C>: Clone,
        ProverData<C>: Clone,
        CircuitAir<F, EF, P>: ProverAir<F, EF>,
        EF::ExtensionPacking: From<EF> + From<F::Packing>,
    {
        assert!(
            self.log_heights()
                .iter()
                .all(|&log_height| log_height >= config.min_num_variables()),
            "every table meets the configuration's minimum height"
        );
        let publics = self.public_values();
        let instances = self
            .airs
            .iter()
            .zip(&self.traces)
            .zip(&publics)
            .map(|((air, trace), publics)| {
                ProverInstance::new(air, Table::new(trace.transpose()), key, publics)
            })
            .collect();
        prove(config, ProverInstances::new(instances), 0, challenger)
    }
}

/// How many times every wire is read.
fn read_counts<F>(ops: &[Op<F>], wires: usize) -> Vec<u32> {
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
            Op::PermuteIf { inputs, take, .. } => {
                inputs.iter().for_each(&mut read);
                read(take);
            }
            Op::MerkleStep { digest, bit, .. } => {
                digest.iter().for_each(&mut read);
                read(bit);
            }
        }
    }
    reads
}
