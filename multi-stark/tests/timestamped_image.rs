//! A timestamped memory seeded from a public image, proved and verified over GF(2^128).
//!
//! The image is a periodic column of the boundary table, so nothing commits it.
//!
//! The verifier evaluates it at the zerocheck point in closed form, one term per word.

use std::borrow::Cow;
use std::collections::HashMap;

use p3_air::{Air, BaseAir, WindowAccess};
use p3_binary_field::{BinaryChallenger, BinaryField64, BinaryField128};
use p3_binary_pcs::{
    BinaryPcs, BinaryPcsConfig, BinaryPcsParams, BinaryPcsProverData, GroupedCodewordMmcs,
};
use p3_bus::{
    BusInteractionBuilder, ClockGap, ClockRangeAir, PublicImage, RangeRead, TimestampedAccess,
    TimestampedBoundaryAir, TimestampedMemory, TimestampedMemoryInteractionBuilder,
    TimestampedSeed,
};
use p3_challenger::HashChallenger;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::config::{MultiStarkConfig, PcsError};
use p3_multi_stark::zerocheck::ZerocheckError;
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, VerificationError, VerifierInstance,
    VerifierInstances, prove, setup, verify,
};
use p3_sumcheck::layout::{Layout, SuffixProver, Table, Witness};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_util::log2_ceil_usize;

type F = BinaryField128;
type C = BinaryField64;
type Memory = TimestampedMemory<C, F>;
type Hash = SerializingHasher<Keccak256Hash>;
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MerkleMmcs = p3_merkle_tree::MerkleTreeMmcs<F, u8, Hash, Compress, 2, 32>;
type Mmcs = GroupedCodewordMmcs<MerkleMmcs>;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

/// Base-two logarithm of the memory size.
const LOG_CELLS: usize = 3;

/// Rows of the machine table.
const ROWS: usize = 8;

/// Columns of the machine table: the clock, then one access.
const MACHINE_WIDTH: usize = 11;

/// The binary PCS over GF(2^128).
struct Config {
    /// Commitment scheme sized for the whole batch.
    pcs: BinaryPcs<F, F, Mmcs, Mmcs>,
}

impl MultiStarkConfig for Config {
    type Val = F;
    type Challenge = F;
    type Challenger = Challenger;
    type Pcs = BinaryPcs<F, F, Mmcs, Mmcs>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        Some(128)
    }

    fn min_num_variables(&self) -> usize {
        1
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
        SuffixProver::<F, F>::new_witness(tables, 0)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a BinaryPcsProverData<F, F, Mmcs>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// A configuration whose stacked polynomial has `num_variables` variables.
fn config(num_variables: usize) -> Config {
    let params = BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: 100,
    };
    let pcs_config = BinaryPcsConfig::try_new::<F, F>(num_variables, params)
        .unwrap()
        .try_with_folding(3)
        .unwrap();
    let merkle = MerkleMmcs::new(Hash::new(Keccak256Hash), Compress::new(Keccak256Hash), 0);
    let mmcs = Mmcs::for_folding(merkle, &pcs_config);
    Config {
        pcs: BinaryPcs::new(pcs_config, mmcs.clone(), mmcs).unwrap(),
    }
}

/// A fresh transcript.
fn challenger() -> Challenger {
    Challenger::from_hasher(b"p3-timestamped-image-test".to_vec(), Keccak256Hash)
}

/// The memory under test, one component per cell.
fn memory() -> Memory {
    Memory::new("tm-memory", "tm-low", "tm-high", 1).unwrap()
}

/// A distinguishable field element.
fn value(tag: u128) -> F {
    F::from_le_bytes(tag.to_le_bytes())
}

/// The image: cell 2 holds 11, cells 5 and 6 hold `word` and 9.
fn image(word: u128) -> PublicImage<F> {
    PublicImage::new(
        &memory(),
        LOG_CELLS,
        vec![(2, vec![value(11)]), (5, vec![value(word), value(9)])],
    )
    .unwrap()
}

/// A machine making one access per row, at the row's clock.
struct MachineAir {
    /// Memory the accesses go to.
    memory: Memory,
}

/// The three tables of one memory, so they fit one batch.
enum MemoryAir {
    /// The machine making the accesses.
    Machine(MachineAir),
    /// The seed and close block.
    Boundary(TimestampedBoundaryAir<C, F>),
    /// The seed and close block, as a verifier that never materializes the image sees it.
    SparseBoundary(TimestampedBoundaryAir<C, F>),
    /// Both range tables.
    Range(ClockRangeAir<C, F>),
}

impl BaseAir<F> for MemoryAir {
    fn width(&self) -> usize {
        match self {
            Self::Machine(_) => MACHINE_WIDTH,
            Self::Boundary(air) | Self::SparseBoundary(air) => air.width(),
            Self::Range(air) => BaseAir::<F>::width(air),
        }
    }

    fn num_periodic_columns(&self) -> usize {
        match self {
            Self::Boundary(air) | Self::SparseBoundary(air) => air.num_periodic_columns(),
            _ => 0,
        }
    }

    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
        match self {
            Self::Boundary(air) => air.periodic_columns(),
            Self::SparseBoundary(_) => panic!("the verifier materializes the image"),
            _ => Cow::Borrowed(&[]),
        }
    }

    fn periodic_periods(&self) -> Vec<usize> {
        match self {
            Self::Boundary(air) | Self::SparseBoundary(air) => air.periodic_periods(),
            _ => Vec::new(),
        }
    }

    fn periodic_evaluations<EF: ExtensionField<F>>(&self, point: &[EF]) -> Option<Vec<EF>> {
        match self {
            Self::Boundary(air) | Self::SparseBoundary(air) => air.periodic_evaluations(point),
            _ => None,
        }
    }
}

impl<AB: BusInteractionBuilder<F = F>> Air<AB> for MemoryAir {
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::Machine(air) => {
                let row: Vec<AB::Expr> = builder
                    .main()
                    .current_slice()
                    .iter()
                    .map(|&v| v.into())
                    .collect();
                let read = |offset: usize| RangeRead {
                    value: row[offset].clone(),
                    count: row[offset + 1].clone(),
                    count_inverse: row[offset + 2].clone(),
                };
                let access = TimestampedAccess {
                    address: row[1].clone(),
                    previous: row[2].clone(),
                    old: vec![row[3].clone()],
                    new: vec![row[4].clone()],
                    gap: ClockGap {
                        low: read(5),
                        high: read(8),
                    },
                };
                builder.timestamped_access(&air.memory, row[0].clone(), 0, access);
            }
            Self::Boundary(air) | Self::SparseBoundary(air) => air.eval(builder),
            Self::Range(air) => air.eval(builder),
        }
    }
}

/// Column-major table from row-major values.
fn table(values: Vec<F>, width: usize) -> Table<F> {
    Table::new(RowMajorMatrix::new(values, width).transpose())
}

/// Traces of the machine, boundary and range tables for a run from `starts`.
///
/// Each operation is a clock exponent, a cell, and an optional written value.
fn traces(starts: &[F], program: &[(u64, usize, Option<u128>)]) -> [Table<F>; 3] {
    let tick = Memory::tick();
    let height = ClockRangeAir::<C, F>::HEIGHT;
    let index = |entry: fn(usize) -> F| {
        (0..height)
            .map(|row| (entry(row), row))
            .collect::<HashMap<_, _>>()
    };
    let (low_index, high_index) = (
        index(ClockRangeAir::<C, F>::low),
        index(ClockRangeAir::<C, F>::high),
    );
    let mut reads = [vec![0u64; height], vec![0u64; height]];

    // The k-th read of a range entry holds count `G^k`.
    let mut count = |side: usize, factor: F| {
        let row = [&low_index, &high_index][side][&factor];
        let count = F::GENERATOR.exp_u64(reads[side][row]);
        reads[side][row] += 1;
        [factor, count, count.inverse()]
    };

    // Every cell starts at time `g^0` holding its starting value.
    let mut state = starts
        .iter()
        .map(|&start| (0u64, start))
        .collect::<Vec<_>>();
    let mut machine = Vec::new();
    for row in 0..ROWS {
        let (clock, address, previous, old, new, (low, high)) = match program.get(row) {
            Some(&(clock, cell, write)) => {
                let (last, old) = state[cell];
                let new = write.map_or(old, value);
                state[cell] = (clock, new);
                (
                    tick.exp_u64(clock),
                    TimestampedBoundaryAir::<C, F>::cell_address(cell),
                    tick.exp_u64(last),
                    old,
                    new,
                    Memory::gap_factors(clock - last).unwrap(),
                )
            }
            // Padding sits at time zero and reads the smallest range entries.
            None => (
                F::ZERO,
                F::ONE,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                (
                    ClockRangeAir::<C, F>::low(0),
                    ClockRangeAir::<C, F>::high(0),
                ),
            ),
        };
        machine.extend([clock, address, previous, old, new]);
        machine.extend(count(0, low));
        machine.extend(count(1, high));
    }

    let boundary = state
        .iter()
        .enumerate()
        .flat_map(|(cell, &(last, final_value))| {
            [
                TimestampedBoundaryAir::<C, F>::cell_address(cell),
                tick.exp_u64(last),
                final_value,
            ]
        })
        .collect();

    let range = (0..height)
        .flat_map(|row| {
            [
                ClockRangeAir::<C, F>::low(row),
                F::GENERATOR.exp_u64(reads[0][row]),
                ClockRangeAir::<C, F>::high(row),
                F::GENERATOR.exp_u64(reads[1][row]),
            ]
        })
        .collect();

    [
        table(machine, MACHINE_WIDTH),
        table(boundary, 3),
        table(range, 4),
    ]
}

/// The three AIRs, with the boundary seeded from `image`.
///
/// A sparse boundary panics if anything materializes its image.
fn airs(image: PublicImage<F>, sparse: bool) -> [MemoryAir; 3] {
    let boundary = TimestampedBoundaryAir::new(memory(), TimestampedSeed::Public(image)).unwrap();
    [
        MemoryAir::Machine(MachineAir { memory: memory() }),
        if sparse {
            MemoryAir::SparseBoundary(boundary)
        } else {
            MemoryAir::Boundary(boundary)
        },
        MemoryAir::Range(ClockRangeAir::new(memory())),
    ]
}

/// Reads of every image word, then a write over one of them.
const PROGRAM: [(u64, usize, Option<u128>); 4] =
    [(1, 5, None), (2, 2, None), (3, 5, Some(3)), (4, 6, None)];

/// Heights of the machine, boundary and range tables.
const LOG_HEIGHTS: [usize; 3] = [3, LOG_CELLS, 16];

/// Proves the honest run from the image with 7 in cell 5.
fn honest_proof(config: &Config) -> MultiStarkProof<Config> {
    let airs = airs(image(7), false);
    let refs = airs.each_ref();
    let (pk, _) = setup(config, &refs, &mut challenger()).unwrap();
    let tables = traces(&image(7).columns()[0], &PROGRAM);
    let instances = airs
        .iter()
        .zip(tables)
        .map(|(air, table)| ProverInstance::new(air, table, &pk, &[]))
        .collect();
    prove(
        config,
        ProverInstances::new(instances),
        0,
        &mut challenger(),
    )
    .unwrap()
}

/// Verifies `proof` against a statement whose image holds `word` in cell 5.
///
/// The verifier never materializes the image, so its cost is linear in the words.
fn verify_against(
    config: &Config,
    proof: &MultiStarkProof<Config>,
    word: u128,
) -> Result<(), VerificationError<PcsError<Config>>> {
    let airs = airs(image(word), true);
    let refs = airs.each_ref();
    let (_, vk) = setup(config, &refs, &mut challenger()).unwrap();
    let instances = airs
        .iter()
        .zip(LOG_HEIGHTS)
        .map(|(air, log_height)| VerifierInstance::new(air, &vk, log_height, &[]))
        .collect();
    verify(
        config,
        VerifierInstances::new(instances),
        proof,
        0,
        &mut challenger(),
    )
}

#[test]
fn a_proof_from_a_public_image_verifies_only_against_that_image() {
    let cells = [ROWS * MACHINE_WIDTH, 3 << LOG_CELLS, 4 << 16];
    let config = config(log2_ceil_usize(cells.iter().sum()));
    let proof = honest_proof(&config);

    // The honest statement accepts.
    verify_against(&config, &proof, 7).unwrap();

    // A statement whose image says 5 in cell 5 moves the seed's bus share, so the zerocheck fails.
    assert!(matches!(
        verify_against(&config, &proof, 5),
        Err(VerificationError::Zerocheck(
            ZerocheckError::FinalSumMismatch
        ))
    ));
}
