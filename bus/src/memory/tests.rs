use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::num::NonZeroUsize;

use p3_air::symbolic::{AirLayout, BaseEntry, SymbolicExpr, SymbolicVariable};
use p3_air::{Air, BaseAir, WindowAccess};
use p3_binary_field::{BinaryChallenger, BinaryField128, Rijndael8b};
use p3_challenger::HashChallenger;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_security::bus::PRODUCT_GKR_BATCHING_LABEL;

use super::*;
use crate::{
    BusActivation, BusDirection, BusPlanInput, BusSymbolicBuilder, ProductGkrProof,
    SymbolicBusInteraction,
};

type F = BinaryField128;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

/// Builds one named bus with a fixed payload width.
fn bus_plan<T: Field>(payload_width: usize) -> BusPlan {
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

    // Four rows are enough to establish the symbolic statement shape.
    BusPlan::build(&[BusPlanInput {
        log_height: 2,
        interactions: &interactions,
    }])
    .expect("the fixture uses a valid bus shape")
    .expect("the fixture contains two declarations")
}

/// Builds a deterministic binary-field transcript.
fn challenger() -> Challenger {
    // An empty prefix leaves the product protocol separator first in the transcript.
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// Owned columns for the honest read-only memory fixture.
struct HonestFixture {
    /// Named-bus layout shared by every memory tuple.
    bus: BusPlan,
    /// Verifier-derived memory statement.
    plan: ReadOnlyMemoryPlan<F>,
    /// Value columns of the seeded array.
    table: Vec<Vec<F>>,
    /// Address column of the read events.
    addresses: Vec<F>,
    /// Pre-read count column.
    counts: Vec<F>,
    /// Value columns returned by the reads.
    values: Vec<Vec<F>>,
    /// Final count of each seeded entry.
    final_counts: Vec<F>,
}

/// AIR that issues one conditionally active array read.
struct ReadAir;

impl BaseAir<F> for ReadAir {
    fn width(&self) -> usize {
        // Address, count, and value each use one trace column.
        3
    }
}

impl<AB> Air<AB> for ReadAir
where
    AB: ReadOnlyMemoryInteractionBuilder<F = F>,
{
    fn eval(&self, builder: &mut AB) {
        // Every row issues one unconditional read.
        let row = builder.main();
        let cells = row.current_slice();
        builder.read_only_memory(
            "memory",
            cells[0].into(),
            cells[1].into(),
            [cells[2].into()],
        );
    }
}

/// Builds one honest two-entry array with three reads.
fn honest_fixture() -> HonestFixture {
    // Tuple payload: address, count, one value component.
    let bus = bus_plan::<F>(3);
    let plan = ReadOnlyMemoryPlan::new(&bus, "memory", 2, 3)
        .expect("three reads cannot wrap a 128-bit generator orbit");

    // Array entries live at addresses 1 and g.
    let table = vec![vec![F::GENERATOR.exp_u64(11), F::GENERATOR.exp_u64(13)]];
    let addresses = vec![F::GENERATOR, F::ONE, F::GENERATOR];

    // Reads visit entry one, entry zero, then entry one again.
    let counts = vec![F::ONE, F::ONE, F::GENERATOR];
    let values = vec![vec![
        F::GENERATOR.exp_u64(13),
        F::GENERATOR.exp_u64(11),
        F::GENERATOR.exp_u64(13),
    ]];
    let final_counts = vec![F::GENERATOR, F::GENERATOR.square()];

    HonestFixture {
        bus,
        plan,
        table,
        addresses,
        counts,
        values,
        final_counts,
    }
}

#[test]
fn honest_memory_reduces_to_three_authenticated_claims() {
    // Fixture state: two entries, three reads, and one value component.
    let HonestFixture {
        bus: _bus,
        plan,
        table,
        addresses,
        counts,
        values,
        final_counts,
    } = honest_fixture();
    let table_refs = table.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let value_refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let columns = ReadOnlyMemoryColumns {
        table: &table_refs,
        read_addresses: &addresses,
        read_counts: &counts,
        read_values: &value_refs,
        final_counts: &final_counts,
    };
    let point = [F::GENERATOR.exp_u64(7), F::GENERATOR.exp_u64(9)];
    let leaves = plan
        .materialize(columns, &point, F::GENERATOR.exp_u64(17))
        .expect("all witness dimensions match the statement");

    // Seeds plus reads give five factors on each direction.
    // Only the three pre-read counts enter the nonzero product.
    assert_eq!(leaves.product_inputs().map(<[_]>::len), [5, 5, 3]);
    leaves
        .check_products()
        .expect("the honest memory transcript balances");

    // Product GKR binds the equal bus roots and the independent count root.
    let mut prover_challenger = challenger();
    let (proof, prover_output) = ProductGkrProof::prove::<F, _>(
        &leaves.product_inputs(),
        plan.product_shape(),
        &mut prover_challenger,
    );
    let mut verifier_challenger = challenger();
    let verifier_output = proof
        .verify::<F, _>(plan.product_shape(), &mut verifier_challenger)
        .expect("the honest product reduction verifies");
    assert_eq!(verifier_output, prover_output);

    // The adapter checks roots but leaves commitment authentication to composition.
    let claims = plan
        .claims(verifier_output)
        .expect("balanced roots and nonzero counts satisfy offline memory");
    assert_eq!(claims.point.len(), plan.product_shape().log_height());
}

#[test]
fn air_helper_emits_one_paired_count_transition() {
    // Evaluate the declaration path once over symbolic trace variables.
    let air = ReadAir;
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

    // An unconditional read declares no selector, so no Booleanity constraint appears.
    assert_eq!(profile.base_constraints().len(), 0);
}

#[test]
fn wrong_values_and_missing_entries_break_bus_balance() {
    // Start from an honest statement and alter one returned value.
    let HonestFixture {
        bus: _bus,
        plan,
        table,
        addresses,
        counts,
        mut values,
        final_counts,
    } = honest_fixture();
    values[0][0] += F::ONE;
    let table_refs = table.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let value_refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let columns = ReadOnlyMemoryColumns {
        table: &table_refs,
        read_addresses: &addresses,
        read_counts: &counts,
        read_values: &value_refs,
        final_counts: &final_counts,
    };

    // The altered pair has no matching seed or finalization boundary.
    // Tuple compression is statistical, so exercise several fixed challenge pairs.
    let rejected = (2..8).any(|exponent| {
        let challenge = F::GENERATOR.exp_u64(exponent);
        let point = [challenge, challenge.square()];
        let leaves = plan
            .materialize(columns, &point, challenge.cube())
            .expect("the malformed witness still has the public shape");
        leaves.check_products() == Err(ReadOnlyMemoryError::UnbalancedProducts)
    });
    assert!(rejected);
}

#[test]
fn zero_count_rejects_a_self_cancelling_invalid_read() {
    // One valid entry seeds and finalizes at the same count.
    let bus = bus_plan::<F>(3);
    let plan = ReadOnlyMemoryPlan::new(&bus, "memory", 1, 1).unwrap();
    let table = [F::GENERATOR.exp_u64(11)];
    let invalid_value = [F::GENERATOR.exp_u64(99)];
    let invalid_address = [F::GENERATOR.exp_u64(77)];
    let zero_count = [F::ZERO];
    let final_counts = [F::ONE];
    let table_columns = [&table[..]];
    let read_columns = [&invalid_value[..]];
    let columns = ReadOnlyMemoryColumns {
        table: &table_columns,
        read_addresses: &invalid_address,
        read_counts: &zero_count,
        read_values: &read_columns,
        final_counts: &final_counts,
    };
    let leaves = plan
        .materialize(
            columns,
            &[F::GENERATOR.exp_u64(7), F::GENERATOR.exp_u64(9)],
            F::GENERATOR.exp_u64(17),
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
fn generator_orbits_bound_addresses_and_read_cycles() {
    // The AES field has 255 nonzero elements in one generator orbit.
    let bus = bus_plan::<Rijndael8b>(3);

    // Every orbit element may name one table entry exactly once.
    assert!(ReadOnlyMemoryPlan::<Rijndael8b>::new(&bus, "memory", 255, 0).is_ok());
    assert_eq!(
        ReadOnlyMemoryPlan::<Rijndael8b>::new(&bus, "memory", 256, 0),
        Err(ReadOnlyMemoryError::AddressOrbitTooShort {
            table_len: 256,
            orbit_len: 255,
        })
    );

    // A read multiset of size 255 could contain one complete forged orbit.
    assert_eq!(
        ReadOnlyMemoryPlan::<Rijndael8b>::new(&bus, "memory", 1, 255),
        Err(ReadOnlyMemoryError::CountOrbitTooShort {
            read_len: 255,
            orbit_len: 255,
        })
    );
}

#[test]
fn malformed_columns_and_outputs_are_rejected_without_indexing() {
    // A one-component plan rejects a missing read-value column first.
    let HonestFixture {
        bus: _bus,
        plan,
        table,
        addresses,
        counts,
        values: _values,
        final_counts,
    } = honest_fixture();
    let table_refs = table.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let columns = ReadOnlyMemoryColumns {
        table: &table_refs,
        read_addresses: &addresses,
        read_counts: &counts,
        read_values: &[],
        final_counts: &final_counts,
    };
    assert_eq!(
        plan.materialize(
            columns,
            &[F::GENERATOR.exp_u64(7), F::GENERATOR.exp_u64(9)],
            F::GENERATOR.exp_u64(17),
        ),
        Err(ReadOnlyMemoryError::ReadWidthMismatch {
            expected: 1,
            actual: 0,
        })
    );

    // A malformed reduction cannot use empty root and claim vectors to trigger indexing.
    let malformed = ProductGkrOutput {
        roots: Vec::<F>::new(),
        point: vec![F::ZERO; plan.product_shape().log_height()],
        values: Vec::new(),
    };
    assert_eq!(
        plan.claims(malformed),
        Err(ReadOnlyMemoryError::RootCountMismatch {
            expected: 3,
            actual: 0,
        })
    );

    // A structurally balanced output still fails when its count product vanishes.
    let zero_count = ProductGkrOutput {
        roots: vec![F::ONE, F::ONE, F::ZERO],
        point: vec![F::ZERO; plan.product_shape().log_height()],
        values: vec![F::ONE; 3],
    };
    assert_eq!(
        plan.claims(zero_count),
        Err(ReadOnlyMemoryError::ZeroCountProduct)
    );
}

#[test]
fn security_uses_the_three_tree_schedule() {
    // Two entries plus three reads produce a height-three product tree.
    let HonestFixture { plan, .. } = honest_fixture();
    let components = plan.security_components(NonZeroUsize::new(128).unwrap());
    let batching = components
        .iter()
        .find(|term| term.label == PRODUCT_GKR_BATCHING_LABEL)
        .expect("a nontrivial product schedule has a batching term");

    // Two layers batch three trees, giving numerator 2 * (3 - 1) = 4.
    assert_eq!(batching.bits.bits(), 126.0);

    // Tuple compression charges two variables against the five factors of the larger side.
    // Understating that input to the three read factors would report 128 - log2(6) bits instead.
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
    // Fixture state: two entries, three reads, and one value component.
    let HonestFixture {
        bus: _bus,
        plan,
        table,
        addresses,
        counts,
        values,
        final_counts,
    } = honest_fixture();
    let table_refs = table.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let value_refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let columns = ReadOnlyMemoryColumns {
        table: &table_refs,
        read_addresses: &addresses,
        read_counts: &counts,
        read_values: &value_refs,
        final_counts: &final_counts,
    };
    let point = [F::GENERATOR.exp_u64(7), F::GENERATOR.exp_u64(9)];
    let leaves = plan
        .materialize(columns, &point, F::GENERATOR.exp_u64(17))
        .expect("all witness dimensions match the statement");
    let mut prover_challenger = challenger();
    let (proof, _) = ProductGkrProof::prove::<F, _>(
        &leaves.product_inputs(),
        plan.product_shape(),
        &mut prover_challenger,
    );
    let mut verifier_challenger = challenger();
    let output = proof
        .verify::<F, _>(plan.product_shape(), &mut verifier_challenger)
        .expect("the honest product reduction verifies");
    let claims = plan.claims(output).expect("the honest reduction is valid");

    // Bus factors fill five leaves while only the three read counts fill the count tree.
    assert_eq!(claims.prefix_lens, [5, 5, 3]);
    assert_eq!(claims.read_offset, 2);

    // Each claim reproduces only when its own table is padded from its own prefix.
    let log_height = plan.product_shape().log_height();
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
