use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::num::NonZeroUsize;

use p3_air::symbolic::{AirLayout, BaseEntry, SymbolicExpr, SymbolicVariable};
use p3_air::{Air, BaseAir, WindowAccess, check_constraints};
use p3_binary_field::{BinaryChallenger, BinaryField128, Rijndael8b};
use p3_challenger::HashChallenger;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_security::bus::PRODUCT_GKR_BATCHING_LABEL;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoroshiro128Plus;

use super::*;
use crate::{
    BusActivation, BusDirection, BusPlanInput, BusSymbolicBuilder, SymbolicBusInteraction,
};

type F = BinaryField128;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

/// Builds one named bus whose declarations cover a fixed number of read rows.
fn bus_plan<T: Field>(payload_width: usize, log_height: usize) -> BusPlan {
    // Every payload slot reads a distinct current-row trace column.
    let fields = (0..payload_width)
        .map(|index| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, index).into())
        .collect::<Vec<_>>();
    let interactions =
        [BusDirection::Push, BusDirection::Pull].map(|direction| SymbolicBusInteraction::<T> {
            bus_name: "memory".to_string(),
            direction,
            fields: fields.clone(),
            activation: BusActivation::Always,
        });

    BusPlan::build(&[BusPlanInput {
        log_height,
        interactions: &interactions,
    }])
    .expect("the fixture uses a valid bus shape")
    .expect("the fixture contains two declarations")
}

/// Builds one named bus declaring an exact, possibly non-power-of-two number of read rows.
fn bus_plan_rows<T: Field>(payload_width: usize, rows: usize) -> BusPlan {
    // One single-row block per read keeps the declared total free of power-of-two rounding.
    let fields = (0..payload_width)
        .map(|index| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, index).into())
        .collect::<Vec<_>>();
    let interactions = (0..rows)
        .flat_map(|_| {
            [BusDirection::Pull, BusDirection::Push].map(|direction| SymbolicBusInteraction::<T> {
                bus_name: "memory".to_string(),
                direction,
                fields: fields.clone(),
                activation: BusActivation::Always,
            })
        })
        .collect::<Vec<_>>();

    BusPlan::build(&[BusPlanInput {
        log_height: 0,
        interactions: &interactions,
    }])
    .expect("the fixture uses a valid bus shape")
    .expect("the fixture contains at least one declaration")
}

/// Builds the fixed challenge pair the direct materialization tests share.
fn fixed_challenges() -> ReadOnlyMemoryChallenges<F> {
    // Direct materialization bypasses the reduction, so the challenges are chosen here.
    ReadOnlyMemoryChallenges {
        fingerprint: vec![F::GENERATOR.exp_u64(7), F::GENERATOR.exp_u64(9)],
        offset: F::GENERATOR.exp_u64(17),
    }
}

/// Builds one challenge pair from a single generator exponent.
fn challenges_at(exponent: u64) -> ReadOnlyMemoryChallenges<F> {
    let challenge = F::GENERATOR.exp_u64(exponent);
    ReadOnlyMemoryChallenges {
        fingerprint: vec![challenge, challenge.square()],
        offset: challenge.cube(),
    }
}

/// Builds a deterministic binary-field transcript.
fn challenger() -> Challenger {
    // An empty prefix leaves the product protocol separator first in the transcript.
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// Assembles borrowed columns from owned witness vectors.
fn memory_columns<'a>(
    table: &'a [&'a [F]],
    final_counts: &'a [F],
    addresses: &'a [F],
    counts: &'a [F],
    values: &'a [&'a [F]],
) -> ReadOnlyMemoryColumns<'a, F> {
    ReadOnlyMemoryColumns {
        table: MemoryTableValues(table),
        final_counts: MemoryFinalCounts(final_counts),
        read_addresses: MemoryReadAddresses(addresses),
        read_counts: MemoryReadCounts(counts),
        read_values: MemoryReadValues(values),
    }
}

/// Reports whether a witness is rejected under at least one of several fixed challenge pairs.
///
/// Tuple compression is statistical, so a single challenge could cancel a real difference.
fn rejected_somewhere(
    plan: &ReadOnlyMemoryPlan<F>,
    columns: ReadOnlyMemoryColumns<'_, F>,
    expected: &ReadOnlyMemoryError,
) -> bool {
    (2..8).any(|exponent| {
        let leaves = plan
            .materialize(columns, &challenges_at(exponent))
            .expect("the malformed witness still has the public shape");
        leaves.check_products().as_ref() == Err(expected)
    })
}

/// Owned columns for the honest read-only memory fixture.
struct HonestFixture {
    /// Named-bus layout shared by every memory tuple.
    bus: BusPlan,
    /// Verifier-derived memory statement.
    plan: ReadOnlyMemoryPlan<F>,
    /// Value columns of the seeded array.
    table: Vec<Vec<F>>,
    /// Final count of each seeded entry.
    final_counts: Vec<F>,
    /// Address column of the read events.
    addresses: Vec<F>,
    /// Pre-read count column.
    counts: Vec<F>,
    /// Value columns returned by the reads.
    values: Vec<Vec<F>>,
}

/// AIR that issues one array read per row.
struct ReadAir {
    /// Checked handle naming the array every row reads.
    bus: ReadOnlyMemoryBus<F>,
}

impl ReadAir {
    /// Builds the handle once so repeated evaluations reuse it.
    fn new() -> Self {
        Self {
            bus: ReadOnlyMemoryBus::new("memory").expect("a 128-bit count orbit is unreachable"),
        }
    }
}

impl BaseAir<F> for ReadAir {
    fn width(&self) -> usize {
        // Address, count, count inverse, and value each use one trace column.
        4
    }
}

impl<AB> Air<AB> for ReadAir
where
    AB: ReadOnlyMemoryInteractionBuilder<F = F>,
{
    fn eval(&self, builder: &mut AB) {
        // The handle is built once by the AIR, so an evaluation per point costs nothing extra.
        let row = builder.main();
        let cells = row.current_slice();
        builder.read_only_memory(
            &self.bus,
            cells[0].into(),
            cells[1].into(),
            cells[2].into(),
            [cells[3].into()],
        );
    }
}

/// Builds one honest three-entry array read twice.
fn honest_fixture() -> HonestFixture {
    // Tuple payload: address, count, one value component.
    // A one-variable trace declares exactly the two reads the statement covers.
    let bus = bus_plan::<F>(3, 1);
    let plan = ReadOnlyMemoryPlan::new(&bus, "memory", 3, 2)
        .expect("two reads cannot wrap a 128-bit generator orbit");

    // Array entries live at addresses 1, g, and g squared.
    let table = vec![vec![
        F::GENERATOR.exp_u64(11),
        F::GENERATOR.exp_u64(13),
        F::GENERATOR.exp_u64(17),
    ]];

    // Both reads visit entry one, so its count advances twice.
    let addresses = vec![F::GENERATOR, F::GENERATOR];
    let counts = vec![F::ONE, F::GENERATOR];
    let values = vec![vec![F::GENERATOR.exp_u64(13), F::GENERATOR.exp_u64(13)]];
    let final_counts = vec![F::ONE, F::GENERATOR.square(), F::ONE];

    HonestFixture {
        bus,
        plan,
        table,
        final_counts,
        addresses,
        counts,
        values,
    }
}

#[test]
fn honest_memory_reduces_to_three_authenticated_claims() {
    // Fixture state: three entries, two reads, and one value component.
    let HonestFixture {
        bus: _bus,
        plan,
        table,
        final_counts,
        addresses,
        counts,
        values,
    } = honest_fixture();
    let table_refs = table.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let value_refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let columns = memory_columns(&table_refs, &final_counts, &addresses, &counts, &value_refs);
    let leaves = plan
        .materialize(columns, &fixed_challenges())
        .expect("all witness dimensions match the statement");

    // Seeds plus reads give five factors on each direction.
    // Only the two pre-read counts enter the nonzero product.
    assert_eq!(leaves.product_inputs().map(<[_]>::len), [5, 5, 2]);
    leaves
        .check_products()
        .expect("the honest memory transcript balances");

    // The reduction binds the equal bus roots and the independent count root.
    let mut prover_challenger = challenger();
    let (proof, prover_claims) = plan
        .prove::<F, _>(columns, &mut prover_challenger)
        .expect("the honest memory statement proves");
    let mut verifier_challenger = challenger();
    let claims = plan
        .verify(&proof, &mut verifier_challenger)
        .expect("the honest memory reduction verifies");

    // The adapter checks roots but leaves commitment authentication to composition.
    assert_eq!(claims, prover_claims);
    assert_eq!(claims.point.len(), plan.product_shape.log_height());
}

#[test]
fn air_helper_emits_one_paired_count_transition() {
    // Evaluate the declaration path once over symbolic trace variables.
    let air = ReadAir::new();
    let profile = BusSymbolicBuilder::<F, F>::from_air(&air, AirLayout::from_air(&air));
    let interactions = profile.interactions();

    // One logical read becomes a pull followed by a generator-advanced push.
    assert_eq!(interactions.len(), 2);
    assert_eq!(interactions[0].direction, BusDirection::Pull);
    assert_eq!(interactions[1].direction, BusDirection::Push);
    assert_eq!(interactions[0].fields.len(), 3);
    assert_eq!(interactions[1].fields.len(), 3);
    assert!(matches!(
        interactions[1].fields[1],
        SymbolicExpr::Mul { .. }
    ));

    // Both directions must read the same address and the same value expression.
    // A push reading another column would leave every honest trace unbalanced.
    assert_eq!(
        format!("{:?}", interactions[0].fields[0]),
        format!("{:?}", interactions[1].fields[0]),
    );
    assert_eq!(
        format!("{:?}", interactions[0].fields[2]),
        format!("{:?}", interactions[1].fields[2]),
    );

    // An unconditional read declares no selector, so the count inverse is the only constraint.
    assert_eq!(profile.base_constraints().len(), 1);
}

#[test]
fn the_statement_must_cover_exactly_the_declared_reads() {
    // The fixture bus declares two read rows on each direction.
    let bus = bus_plan::<F>(3, 1);
    assert!(ReadOnlyMemoryPlan::<F>::new(&bus, "memory", 3, 2).is_ok());

    // A statement covering fewer reads than the AIR declares is refused, not silently combined.
    assert_eq!(
        ReadOnlyMemoryPlan::<F>::new(&bus, "memory", 3, 1),
        Err(ReadOnlyMemoryError::DeclaredReadCountMismatch {
            name: "memory".to_string(),
            expected: 1,
            actual: 2,
        })
    );

    // A statement with no reads at all is refused by the same check rather than panicking.
    assert_eq!(
        ReadOnlyMemoryPlan::<F>::new(&bus, "memory", 3, 0),
        Err(ReadOnlyMemoryError::DeclaredReadCountMismatch {
            name: "memory".to_string(),
            expected: 0,
            actual: 2,
        })
    );
}

#[test]
fn wrong_values_and_missing_entries_break_bus_balance() {
    // Start from an honest statement and alter one returned value.
    let HonestFixture {
        bus: _bus,
        plan,
        table,
        final_counts,
        addresses,
        counts,
        mut values,
    } = honest_fixture();
    values[0][0] += F::ONE;
    let table_refs = table.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let value_refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let columns = memory_columns(&table_refs, &final_counts, &addresses, &counts, &value_refs);

    // The altered pair has no matching seed or finalization boundary.
    assert!(rejected_somewhere(
        &plan,
        columns,
        &ReadOnlyMemoryError::UnbalancedProducts
    ));
}

#[test]
fn zero_count_rejects_a_self_cancelling_invalid_read() {
    // One valid entry seeds and finalizes at the same count.
    let bus = bus_plan::<F>(3, 0);
    let plan = ReadOnlyMemoryPlan::new(&bus, "memory", 1, 1).unwrap();
    let table = [F::GENERATOR.exp_u64(11)];
    let invalid_value = [F::GENERATOR.exp_u64(99)];
    let invalid_address = [F::GENERATOR.exp_u64(77)];
    let zero_count = [F::ZERO];
    let final_counts = [F::ONE];
    let table_columns = [&table[..]];
    let read_columns = [&invalid_value[..]];
    let leaves = plan
        .materialize(
            memory_columns(
                &table_columns,
                &final_counts,
                &invalid_address,
                &zero_count,
                &read_columns,
            ),
            &fixed_challenges(),
        )
        .unwrap();

    // In characteristic two, multiplying zero by the generator stays zero.
    // The invalid read therefore contributes the same tuple to both directions.
    assert_eq!(
        leaves.product_inputs()[0].iter().copied().product::<F>(),
        leaves.product_inputs()[1].iter().copied().product::<F>(),
    );

    // The independent count product closes this otherwise reachable attack.
    assert_eq!(
        leaves.check_products(),
        Err(ReadOnlyMemoryError::ZeroCountProduct)
    );
}

#[test]
fn a_forged_address_with_a_nonzero_count_breaks_the_orbit_argument() {
    // One valid entry sits at address one, and the single read invents another address.
    let bus = bus_plan::<F>(3, 0);
    let plan = ReadOnlyMemoryPlan::new(&bus, "memory", 1, 1).unwrap();
    let table = [F::GENERATOR.exp_u64(11)];
    let forged_value = [F::GENERATOR.exp_u64(99)];
    let forged_address = [F::GENERATOR.exp_u64(77)];
    let honest_count = [F::ONE];
    let final_counts = [F::GENERATOR];
    let table_columns = [&table[..]];
    let read_columns = [&forged_value[..]];
    let columns = memory_columns(
        &table_columns,
        &final_counts,
        &forged_address,
        &honest_count,
        &read_columns,
    );
    let leaves = plan.materialize(columns, &fixed_challenges()).unwrap();

    // The count tree passes, so nothing but the orbit argument can reject this read.
    assert_ne!(
        leaves.product_inputs()[2].iter().copied().product::<F>(),
        F::ZERO,
    );

    // Reading a value that was never seeded leaves the two sides unbalanced.
    assert!(rejected_somewhere(
        &plan,
        columns,
        &ReadOnlyMemoryError::UnbalancedProducts
    ));
}

#[test]
fn a_zero_final_count_cannot_appear_on_the_push_side() {
    // Start from an honest statement and finalize one entry at zero.
    let HonestFixture {
        bus: _bus,
        plan,
        table,
        mut final_counts,
        addresses,
        counts,
        values,
    } = honest_fixture();
    final_counts[0] = F::ZERO;
    let table_refs = table.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let value_refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let columns = memory_columns(&table_refs, &final_counts, &addresses, &counts, &value_refs);
    let leaves = plan.materialize(columns, &fixed_challenges()).unwrap();

    // Every read count stays nonzero, so the count tree cannot be what rejects this.
    assert_ne!(
        leaves.product_inputs()[2].iter().copied().product::<F>(),
        F::ZERO,
    );

    // Seeds start at one and reads produce generator multiples, so zero is unreachable on the push side.
    assert!(rejected_somewhere(
        &plan,
        columns,
        &ReadOnlyMemoryError::UnbalancedProducts
    ));
}

#[test]
fn a_replayed_read_tuple_breaks_chain_rigidity() {
    // Two reads of one entry must consume consecutive orbit elements.
    let bus = bus_plan::<F>(3, 1);
    let plan = ReadOnlyMemoryPlan::new(&bus, "memory", 1, 2).unwrap();
    let value = F::GENERATOR.exp_u64(11);
    let table = [value];
    let read_values = [value, value];
    let addresses = [F::ONE, F::ONE];

    // Both rows replay the same count instead of advancing.
    let replayed_counts = [F::ONE, F::ONE];
    let final_counts = [F::GENERATOR.square()];
    let table_columns = [&table[..]];
    let read_columns = [&read_values[..]];
    let columns = memory_columns(
        &table_columns,
        &final_counts,
        &addresses,
        &replayed_counts,
        &read_columns,
    );
    let leaves = plan.materialize(columns, &fixed_challenges()).unwrap();

    // Both replayed counts are nonzero, so the count tree passes.
    assert_ne!(
        leaves.product_inputs()[2].iter().copied().product::<F>(),
        F::ZERO,
    );

    // The consumed counts must form one unbroken chain, so a repeat is rejected.
    assert!(rejected_somewhere(
        &plan,
        columns,
        &ReadOnlyMemoryError::UnbalancedProducts
    ));
}

#[test]
fn the_smallest_statement_reduces_without_panicking() {
    // One entry and one read give the smallest legal memory statement.
    let bus = bus_plan::<F>(3, 0);
    let plan = ReadOnlyMemoryPlan::new(&bus, "memory", 1, 1).unwrap();
    let value = F::GENERATOR.exp_u64(11);
    let table = [value];
    let read_values = [value];
    let addresses = [F::ONE];
    let counts = [F::ONE];
    let final_counts = [F::GENERATOR];
    let table_columns = [&table[..]];
    let read_columns = [&read_values[..]];
    let columns = memory_columns(
        &table_columns,
        &final_counts,
        &addresses,
        &counts,
        &read_columns,
    );

    // Two factors on each side round up to a one-variable reduction.
    assert_eq!(plan.product_shape.log_height(), 1);
    let mut prover_challenger = challenger();
    let (proof, _) = plan.prove::<F, _>(columns, &mut prover_challenger).unwrap();
    let mut verifier_challenger = challenger();
    let claims = plan.verify(&proof, &mut verifier_challenger).unwrap();
    assert_eq!(claims.prefix_lens, [2, 2, 1]);
    assert_eq!(claims.read_offset, 1);
}

#[test]
fn a_valueless_array_keeps_its_address_and_count_columns() {
    // A payload of exactly two slots leaves no room for any value component.
    let bus = bus_plan::<F>(2, 0);
    let plan = ReadOnlyMemoryPlan::new(&bus, "memory", 1, 1).unwrap();
    assert_eq!(plan.value_width(), 0);

    // Both height loops over value columns are vacuous, so only the metadata columns are checked.
    let addresses = [F::ONE];
    let counts = [F::ONE];
    let final_counts = [F::GENERATOR];
    let mut prover_challenger = challenger();
    let (proof, _) = plan
        .prove::<F, _>(
            memory_columns(&[], &final_counts, &addresses, &counts, &[]),
            &mut prover_challenger,
        )
        .unwrap();
    let mut verifier_challenger = challenger();
    assert!(plan.verify(&proof, &mut verifier_challenger).is_ok());

    // A metadata column of the wrong height is still rejected in the degenerate shape.
    let long_counts = [F::ONE, F::ONE];
    assert_eq!(
        plan.materialize(
            memory_columns(&[], &final_counts, &addresses, &long_counts, &[]),
            &fixed_challenges(),
        ),
        Err(ReadOnlyMemoryError::CountHeightMismatch {
            expected: 1,
            actual: 2,
        })
    );
}

#[test]
fn the_memory_dimensions_separate_two_statements_of_one_product_shape() {
    // Five factors and eight factors both round up to a three-variable reduction.
    let small_bus = bus_plan::<F>(3, 1);
    let small = ReadOnlyMemoryPlan::<F>::new(&small_bus, "memory", 3, 2).unwrap();
    let large_bus = bus_plan::<F>(3, 2);
    let large = ReadOnlyMemoryPlan::<F>::new(&large_bus, "memory", 4, 4).unwrap();

    // The product statement alone cannot tell the two apart.
    assert_eq!(small.product_shape, large.product_shape);

    // Prove the smaller statement honestly.
    let HonestFixture {
        table,
        final_counts,
        addresses,
        counts,
        values,
        ..
    } = honest_fixture();
    let table_refs = table.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let value_refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let columns = memory_columns(&table_refs, &final_counts, &addresses, &counts, &value_refs);
    let mut prover_challenger = challenger();
    let (proof, _) = small
        .prove::<F, _>(columns, &mut prover_challenger)
        .unwrap();

    // The same proof must not replay under a statement of different dimensions.
    let mut honest_challenger = challenger();
    assert!(small.verify(&proof, &mut honest_challenger).is_ok());
    let mut swapped_challenger = challenger();
    assert!(large.verify(&proof, &mut swapped_challenger).is_err());
}

#[test]
fn generator_orbits_bound_addresses_and_read_cycles() {
    // The AES field has 255 nonzero elements in one generator orbit.
    let bus = bus_plan::<Rijndael8b>(3, 0);

    // Every orbit element may name one table entry exactly once.
    assert!(ReadOnlyMemoryPlan::<Rijndael8b>::new(&bus, "memory", 255, 1).is_ok());
    assert_eq!(
        ReadOnlyMemoryPlan::<Rijndael8b>::new(&bus, "memory", 256, 1),
        Err(ReadOnlyMemoryError::AddressOrbitTooShort {
            table_len: 256,
            orbit_len: 255,
        })
    );

    // Exactly one orbit of reads already suffices to forge a self-cancelling cycle.
    let full_bus = bus_plan_rows::<Rijndael8b>(3, 255);
    assert_eq!(
        ReadOnlyMemoryPlan::<Rijndael8b>::new(&full_bus, "memory", 1, 255),
        Err(ReadOnlyMemoryError::CountOrbitTooShort {
            read_len: 255,
            orbit_len: 255,
        })
    );

    // One read fewer cannot close the cycle, so the bound is strict rather than inclusive.
    let short_bus = bus_plan_rows::<Rijndael8b>(3, 254);
    assert!(ReadOnlyMemoryPlan::<Rijndael8b>::new(&short_bus, "memory", 1, 254).is_ok());
}

#[test]
fn a_reachable_count_orbit_cannot_be_named_by_a_declaration() {
    // A declaration cannot see the trace heights, so only an unreachable orbit is admissible.
    assert_eq!(
        ReadOnlyMemoryBus::<Rijndael8b>::new("memory").err(),
        Some(ReadOnlyMemoryError::CountOrbitReachable { orbit_len: 255 })
    );

    // The orbit of a 128-bit field exceeds every machine-word row count.
    assert!(ReadOnlyMemoryBus::<F>::new("memory").is_ok());
}

#[test]
fn the_read_helper_constrains_every_count_against_its_inverse() {
    // One honest row reads value 11 at address 1 holding count g, whose inverse is g inverted.
    let count = F::GENERATOR;
    let honest = RowMajorMatrix::new(
        vec![F::ONE, count, count.inverse(), F::GENERATOR.exp_u64(11)],
        4,
    );
    check_constraints(&ReadAir::new(), &honest, &[]);
}

#[test]
#[should_panic(expected = "constraint")]
fn a_zero_count_row_is_refused_by_the_read_helper() {
    // A zero count has no inverse, so the self-cancelling read cannot be declared at all.
    let forged = RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO, F::GENERATOR.exp_u64(11)], 4);
    check_constraints(&ReadAir::new(), &forged, &[]);
}

#[test]
fn malformed_columns_and_outputs_are_rejected_without_indexing() {
    // A one-component plan rejects a missing read-value column first.
    let HonestFixture {
        bus: _bus,
        plan,
        table,
        final_counts,
        addresses,
        counts,
        values: _values,
    } = honest_fixture();
    let table_refs = table.iter().map(Vec::as_slice).collect::<Vec<_>>();
    assert_eq!(
        plan.materialize(
            memory_columns(&table_refs, &final_counts, &addresses, &counts, &[]),
            &fixed_challenges(),
        ),
        Err(ReadOnlyMemoryError::ReadWidthMismatch {
            expected: 1,
            actual: 0,
        })
    );

    // A malformed reduction cannot use empty root and claim vectors to trigger indexing.
    let malformed = ProductGkrOutput {
        roots: Vec::<F>::new(),
        point: vec![F::ZERO; plan.product_shape.log_height()],
        values: Vec::new(),
    };
    assert_eq!(
        plan.claims(fixed_challenges(), malformed),
        Err(ReadOnlyMemoryError::RootCountMismatch {
            expected: 3,
            actual: 0,
        })
    );

    // A structurally balanced output still fails when its count product vanishes.
    let zero_count = ProductGkrOutput {
        roots: vec![F::ONE, F::ONE, F::ZERO],
        point: vec![F::ZERO; plan.product_shape.log_height()],
        values: vec![F::ONE; 3],
    };
    assert_eq!(
        plan.claims(fixed_challenges(), zero_count),
        Err(ReadOnlyMemoryError::ZeroCountProduct)
    );
}

#[test]
fn security_uses_the_three_tree_schedule() {
    // Three entries plus two reads produce a height-three product tree.
    let HonestFixture { plan, .. } = honest_fixture();
    let components = plan.security_components(NonZeroUsize::new(128).unwrap());
    let batching = components
        .iter()
        .find(|term| term.label == PRODUCT_GKR_BATCHING_LABEL)
        .expect("a nontrivial product schedule has a batching term");

    // Two layers batch three trees, giving numerator 2 * (3 - 1) = 4.
    assert_eq!(batching.bits.bits(), 126.0);

    // Tuple compression charges two variables against the five factors of the larger side.
    // Understating that input to the two read factors would report 128 - log2(4) bits instead.
    let fingerprint = components
        .iter()
        .find(|term| term.label == p3_security::bus::BUS_FINGERPRINT_LABEL)
        .expect("tuple compression is always charged");
    assert!(close(fingerprint.bits.bits(), 124.678_071_905_112_63));

    // The four numerators 10, 5, 4 and 3 union to 22 over the same field order.
    let combined = plan.security_term(NonZeroUsize::new(128).unwrap());
    assert_eq!(combined.label, p3_security::bus::BINARY_BUS_LABEL);
    assert!(close(combined.bits.bits(), 123.540_568_381_362_7));
}

/// Compares two soundness bit counts up to floating-point rounding.
fn close(actual: f64, expected: f64) -> bool {
    // The reference values come from the union-bound formula, not from the implementation.
    let difference = actual - expected;
    difference < 1e-9 && difference > -1e-9
}

/// Evaluates the multilinear extension of a table padded with ones outside one block.
fn padded_evaluation(leaves: &[F], offset: usize, log_height: usize, point: &[F]) -> F {
    // Coordinates run from the most significant index bit to the least significant bit.
    let mut table = vec![F::ONE; 1usize << log_height];
    table[offset..offset + leaves.len()].copy_from_slice(leaves);
    for &coordinate in point.iter().rev() {
        let half = table.len() / 2;
        for row in 0..half {
            table[row] = table[2 * row] + coordinate * (table[2 * row + 1] - table[2 * row]);
        }
        table.truncate(half);
    }

    table[0]
}

#[test]
fn leaf_claims_authenticate_only_under_their_own_prefix() {
    // Fixture state: three entries, two reads, and one value component.
    let HonestFixture {
        bus: _bus,
        plan,
        table,
        final_counts,
        addresses,
        counts,
        values,
    } = honest_fixture();
    let table_refs = table.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let value_refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let columns = memory_columns(&table_refs, &final_counts, &addresses, &counts, &value_refs);
    let mut prover_challenger = challenger();
    let (proof, _) = plan.prove::<F, _>(columns, &mut prover_challenger).unwrap();
    let mut verifier_challenger = challenger();
    let claims = plan
        .verify(&proof, &mut verifier_challenger)
        .expect("the honest reduction is valid");

    // Reconstructing the leaves needs the challenges the reduction itself drew.
    let leaves = plan
        .materialize(columns, &claims.challenges)
        .expect("all witness dimensions match the statement");

    // Bus factors fill five leaves while only the two read counts fill the count tree.
    assert_eq!(claims.prefix_lens, [5, 5, 2]);
    assert_eq!(claims.read_offset, 3);

    // Each claim reproduces only when its own table is padded from its own prefix.
    let log_height = plan.product_shape.log_height();
    let [pushes, pulls, count_leaves] = leaves.product_inputs();
    assert_eq!(
        padded_evaluation(pushes, 0, log_height, &claims.point),
        claims.push,
    );
    assert_eq!(
        padded_evaluation(pulls, 0, log_height, &claims.point),
        claims.pull,
    );
    assert_eq!(
        padded_evaluation(count_leaves, 0, log_height, &claims.point),
        claims.counts,
    );

    // A composer reusing the bus-side layout would place the counts at the read offset.
    assert_ne!(
        padded_evaluation(count_leaves, claims.read_offset, log_height, &claims.point),
        claims.counts,
    );
}

/// Owned witness columns for one randomly generated honest schedule.
struct RandomSchedule {
    /// Value of each array entry.
    table: Vec<F>,
    /// Final count of each array entry.
    final_counts: Vec<F>,
    /// Address of each read event.
    addresses: Vec<F>,
    /// Count held by each read event before it advances.
    counts: Vec<F>,
    /// Value returned by each read event.
    values: Vec<F>,
}

impl RandomSchedule {
    /// Draws one honest schedule of reads over a random array.
    fn new(rng: &mut Xoroshiro128Plus, table_len: usize, read_len: usize) -> Self {
        let table = (0..table_len)
            .map(|_| rng.random::<F>())
            .collect::<Vec<_>>();
        let mut seen = vec![0u64; table_len];
        let mut addresses = Vec::with_capacity(read_len);
        let mut counts = Vec::with_capacity(read_len);
        let mut values = Vec::with_capacity(read_len);
        for _ in 0..read_len {
            let entry = rng.random_range(0..table_len);
            addresses.push(F::GENERATOR.exp_u64(entry as u64));
            counts.push(F::GENERATOR.exp_u64(seen[entry]));
            values.push(table[entry]);
            seen[entry] += 1;
        }

        Self {
            table,
            final_counts: seen.iter().map(|&c| F::GENERATOR.exp_u64(c)).collect(),
            addresses,
            counts,
            values,
        }
    }

    /// Reports whether the schedule satisfies both root obligations at one challenge pair.
    fn balances(&self, plan: &ReadOnlyMemoryPlan<F>, exponent: u64) -> bool {
        let table_columns = [&self.table[..]];
        let read_columns = [&self.values[..]];
        plan.materialize(
            memory_columns(
                &table_columns,
                &self.final_counts,
                &self.addresses,
                &self.counts,
                &read_columns,
            ),
            &challenges_at(exponent),
        )
        .expect("the schedule has the public shape")
        .check_products()
        .is_ok()
    }
}

#[test]
fn one_mutation_of_a_random_honest_schedule_is_always_caught() {
    let mut rng = Xoroshiro128Plus::seed_from_u64(0x0FF1_CE01);
    for trial in 0..24u64 {
        // Trace heights are powers of two, so the declared read count is too.
        let log_reads = (trial % 4) as usize;
        let read_len = 1usize << log_reads;
        let table_len = rng.random_range(1..8usize);
        let bus = bus_plan::<F>(3, log_reads);
        let plan = ReadOnlyMemoryPlan::<F>::new(&bus, "memory", table_len, read_len).unwrap();
        let honest = RandomSchedule::new(&mut rng, table_len, read_len);
        assert!(honest.balances(&plan, 2));

        // Apply exactly one mutation, chosen so it always changes the statement.
        let mut broken = honest;
        let read = rng.random_range(0..read_len);
        let entry = rng.random_range(0..table_len);
        match trial % 5 {
            0 => broken.values[read] += F::ONE,
            1 => broken.addresses[read] *= F::GENERATOR,
            2 => broken.counts[read] *= F::GENERATOR,
            3 => broken.final_counts[entry] *= F::GENERATOR,
            _ => broken.counts[read] = F::ZERO,
        }

        // Tuple compression is statistical, so a rejection at one challenge pair is enough.
        assert!(
            (2..8).any(|exponent| !broken.balances(&plan, exponent)),
            "mutation {} of trial {trial} survived every challenge",
            trial % 5,
        );
    }
}

#[test]
fn swapping_two_read_counts_across_addresses_is_caught() {
    // Two reads of two different entries each consume the first orbit element.
    let bus = bus_plan::<F>(3, 1);
    let plan = ReadOnlyMemoryPlan::<F>::new(&bus, "memory", 2, 2).unwrap();
    let table = [F::GENERATOR.exp_u64(11), F::GENERATOR.exp_u64(13)];
    let read_values = [table[0], table[0]];

    // One entry is read twice, so its two counts differ and can be swapped onto other rows.
    let addresses = [F::ONE, F::ONE];
    let swapped_counts = [F::GENERATOR, F::ONE];
    let final_counts = [F::GENERATOR.square(), F::ONE];
    let table_columns = [&table[..]];
    let read_columns = [&read_values[..]];
    let honest_counts = [F::ONE, F::GENERATOR];
    let honest = memory_columns(
        &table_columns,
        &final_counts,
        &addresses,
        &honest_counts,
        &read_columns,
    );
    assert!(
        plan.materialize(honest, &fixed_challenges())
            .unwrap()
            .check_products()
            .is_ok()
    );

    // Reordering the two counts at one address is only a row permutation, so it still balances.
    let permuted = memory_columns(
        &table_columns,
        &final_counts,
        &addresses,
        &swapped_counts,
        &read_columns,
    );
    assert!(
        plan.materialize(permuted, &fixed_challenges())
            .unwrap()
            .check_products()
            .is_ok()
    );

    // Moving one of those counts onto the other address does break the chain.
    let moved_addresses = [F::ONE, F::GENERATOR];
    let moved_values = [table[0], table[1]];
    let moved_columns = [&moved_values[..]];
    let moved_finals = [F::GENERATOR, F::GENERATOR];
    assert!(rejected_somewhere(
        &plan,
        memory_columns(
            &table_columns,
            &moved_finals,
            &moved_addresses,
            &swapped_counts,
            &moved_columns,
        ),
        &ReadOnlyMemoryError::UnbalancedProducts
    ));
}
